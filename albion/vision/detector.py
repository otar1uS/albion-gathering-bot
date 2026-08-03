"""
Running the model over a frame and turning boxes into things the bot can act on.

Two decisions here were paid for in a long evening of the bot walking into trees it could
not touch, and both look like details:

The model's own suppression threshold is opened right up and the filtering is done here
instead. yolov5 drops everything under `model.conf` before handing anything back, and its
default of 0.25 quietly became the real threshold of the old bot: a resource asking for
less was never given the boxes to decide on, and a node could not be followed at a lower
bar than it was picked at, however low it asked.

Inference is augmented. It costs 157ms a frame against 69ms, which is still three times
what the bot needs, and it is the difference between trees being usable and not: over the
42 frames of the farming zone in images/dataset, frames holding a tree the bot would act
on went from 2 to 14 at 0.5 confidence, and the median best score from 0.16 to 0.36.
"""

import pathlib
from dataclasses import dataclass
from platform import system

import cv2 as cv
import numpy as np

from .. import logs
from ..game import resources

log = logs.get("vision.detector")


@dataclass
class Detection:
    """One thing the model found."""

    profile: object
    label: str
    confidence: float

    # In the model's square.
    box: tuple

    # Centre of the box on screen, used for distances.
    screen: tuple

    # Where to actually put the cursor, on screen. For a tall tree this is the trunk
    # rather than the canopy, because the canopy is leaves with gaps and the game only
    # answers a cursor resting on the node itself.
    click: tuple

    # Pixels from the character, on screen.
    distance: float

    @property
    def gatherable(self):
        return self.profile.gatherable


def process(picture, image_size):
    """
    Turn a screenshot into the image the model expects.

    The screenshot arrives as BGRA and the model wants three channels the same way round
    the training frames had them. The conversion looks wrong and is not: the colours of
    a debug window drawn from this are swapped, and correcting that stops the model
    detecting anything, because it is what every frame it learned from looked like.

    :param picture: BGRA screenshot.
    :param image_size: Side of the square to produce.
    :return: BGR image, image_size by image_size.
    """
    if picture is None or picture.size == 0:
        raise ValueError("empty screenshot")

    if picture.shape[2] == 4:
        picture = picture[:, :, :3]

    return cv.resize(cv.cvtColor(picture, cv.COLOR_RGB2BGR), (image_size, image_size))


class Detector:
    """The model, and what it is allowed to say."""

    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.model = self.__load()
        self.classes = self.__classes()

        for class_id, label in self.classes.items():
            profile = resources.resolve(label)
            log.info("class %s %r -> %s at %.2f", class_id, label, profile,
                     profile.threshold(self.config.vision.confidence))

    def __load(self):
        """
        Load the weights through the vendored yolov5.

        :return: Autoshape model.
        """
        import torch

        weights = self.config.weights_path
        yolov5 = self.config.yolov5_path

        if not yolov5.joinpath("hubconf.py").exists():
            raise FileNotFoundError(
                f"yolov5 is missing from {yolov5}, get it with: "
                f"git submodule update --init --recursive")

        if not weights.exists():
            raise FileNotFoundError(f"the weights are missing from {weights}")

        # Weights trained on Linux pickle their paths as PosixPath and Windows refuses to
        # build one, so loading them fails before the first layer is read. The swap is
        # put back immediately, so nothing else in the process sees it.
        posix = pathlib.PosixPath

        if system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            model = torch.hub.load(str(yolov5), "custom", path=str(weights),
                                   source="local", device="cpu", _verbose=False)
        finally:
            pathlib.PosixPath = posix

        model.conf = self.config.vision.model_floor

        log.info("loaded %s, suppression floor %.2f, augmented %s",
                 weights.name, model.conf, self.config.vision.augment)

        return model

    def __classes(self):
        """
        Labels of the model, whatever shape the checkpoint stored them in.

        :return: Dictionary of class id to label.
        """
        names = self.model.names

        return names if isinstance(names, dict) else dict(enumerate(names))

    def character(self):
        """
        Where the character is on screen.

        It is always in the middle of the window, a little above centre because the
        camera looks down at it and the feet are what the world is measured from.

        :return: (x, y) on screen.
        """
        rect = self.capture.rect

        return rect.left + rect.width // 2, rect.top + rect.height // 2 - 60

    def look(self, factor=1.0):
        """
        Take a picture and resolve everything worth mentioning in it.

        Everything is returned, not only what is being gathered: the roaming needs the
        monsters to walk around them, and following a node needs it found again whatever
        was asked for.

        :param factor: Multiplier on every threshold, under 1 to accept boxes that would
                       not have been picked as a target. Following an already found node
                       uses it.
        :return: (detections nearest first, the frame they were found in).
        """
        image_size = self.config.vision.image_size
        frame = process(self.capture.grab(), image_size)
        raw = self.model(frame, augment=self.config.vision.augment).xyxy[0].tolist()

        character = self.character()
        found = []

        for x1, y1, x2, y2, confidence, class_id in raw:
            label = self.classes[int(class_id)]
            profile = resources.resolve(label)
            threshold = profile.threshold(self.config.vision.confidence) * factor

            if confidence < threshold:
                continue

            box = (int(x1), int(y1), int(x2), int(y2))
            screen = self.capture.rect.to_screen((x1 + x2) / 2, (y1 + y2) / 2, image_size)
            click = self.capture.rect.to_screen(
                (x1 + x2) / 2, y1 + (y2 - y1) * profile.aim_depth, image_size)

            found.append(Detection(
                profile=profile,
                label=label,
                confidence=float(confidence),
                box=box,
                screen=screen,
                click=click,
                distance=float(np.hypot(screen[0] - character[0],
                                        screen[1] - character[1])),
            ))

        found.sort(key=lambda detection: detection.distance)

        return found, frame

    def draw(self, frame, detections):
        """
        Mark up a frame for the preview window.

        :param frame: Frame the detections were found in.
        :param detections: What to draw.
        :return: A copy with the boxes on it.
        """
        frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            colour = (0, 200, 0) if detection.gatherable else (0, 0, 220)

            cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv.putText(frame, f"{detection.label} {detection.confidence:.2f}",
                       (x1, max(y1 - 6, 12)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                       cv.LINE_AA)

        return frame
