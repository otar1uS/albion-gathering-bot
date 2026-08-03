import pathlib

import cv2 as cv
import torch
from Application import game, paths
from Application.Albion import resources, screen
from Application.Capture.Factory import CaptureFactory
from time import time
from math import sqrt
from pathlib import Path
from platform import system


class AlbionDetection:
    MODEL_NAME = paths.MODEL
    IMG_SIZE = screen.IMG_SIZE
    CONFIDENCE = 0.5

    # Test time augmentation: the model is run over the frame again flipped and rescaled
    # and the boxes are merged. It costs 157ms a frame against 69ms, which is 6.4 FPS
    # where the bot asks for 2, and it is what makes the trees usable at all. Measured
    # over the 42 frames of this zone in images/dataset, counting the frames holding a
    # tree the bot would act on: 2 without it against 14 with it at 0.5 confidence, 12
    # against 24 at 0.3, and the median best score in a frame goes 0.16 -> 0.36. Trees
    # are the thinnest class in the training set, 560 boxes against the 11650 of the ore,
    # and this buys back most of what that costs.
    AUGMENT = True

    # yolov5 drops anything under model.conf during its own non maximum suppression,
    # and that default is 0.25, which quietly became the real threshold of the bot: a
    # profile asking for less than that was never given the boxes to decide on, and
    # find_node could not follow a node at a lower bar than it was picked at, however
    # low it asked. The model is therefore opened up here and the bot does its own
    # filtering in scan, per class. Boxes between this and a threshold are not acted
    # upon, they only keep a node the bot is already standing at from disappearing.
    MODEL_FLOOR = 0.10

    def __init__(self,
                 model_name=MODEL_NAME,
                 debug=False,
                 confidence=CONFIDENCE,
                 window_name=game.DEFAULT_WINDOW_NAME,
                 targets=None,
                 preview=None
                 ):
        """
        Initialize the AlbionDetection object.

        :param model_name: Path of the YOLOv5 weights.
        :param debug: Flag to enable debug mode.
        :param confidence: Lowest confidence of a detection to keep.
        :param window_name: Title of the game window.
        :param targets: Resource names to detect, None to detect all of them.
        :param preview: Flag to show the window drawing the detections, follows debug
                        when left to None. The interface logs without that window.
        """
        self.model_name = model_name
        self.debug = debug
        self.preview = debug if preview is None else preview
        self.model = self._load_model()
        self.classes = self._load_classes()
        self.targets = resources.resolve_targets(targets)
        self.target_ids = self._load_target_ids()
        self.window_capture = self._load_capture(window_name)
        self.confidence = confidence
        self.character_position_X = self.IMG_SIZE / 2
        self.character_position_Y = self.IMG_SIZE / 2 - 60

        # Whatever the model recognised on the last look without any profile claiming
        # it, monsters above all. The roaming reads it to walk around them instead of
        # into them, and it is filled here so a single pass of the model serves both.
        self.last_hazards = []

    def _process_image(self, img):
        """
        Preprocess the image, see Application.Albion.screen.

        :param img: Input image.
        :return: Processed image.
        """
        return screen.process(img)

    def _load_capture(self, window_name):
        """
        Find the Albion window.

        :param window_name: Title of the game window.
        :return: Capture bound to the window.
        """
        capture = CaptureFactory(window_name).capture

        if capture is None:
            raise Exception(f"Failed to capture the window {window_name}")

        return capture

    def _load_classes(self):
        """
        Load class information.

        :return: Dictionary containing class information.
        """
        names = self.model.names

        # Depending on the yolov5 version used to train the weights, names is
        # either a dict of id to label or a plain list of labels.
        if not isinstance(names, dict):
            names = dict(enumerate(names))

        classes = {}
        for k, v in names.items():
            classes[k] = {
                "label": v,
                "color": (0, 255, k * 10)
            }
        return classes

    def _load_target_ids(self):
        """
        Find the class ids of the wanted resources.

        :return: Set of class ids to keep during a prediction.
        """
        ids = set()

        for k, v in self.classes.items():
            profile = resources.resolve(v["label"])

            if profile in self.targets:
                ids.add(k)

            if self.debug:
                # Classes matching no resource, monsters for example, are never clicked.
                print(f"Class {k} {v['label']!r} -> {profile}{'' if k in ids else ' (ignored)'}")

        if len(ids) == 0:
            raise Exception(
                f"None of the targets {[str(t) for t in self.targets]} is known by the model, "
                f"available classes are {[v['label'] for v in self.classes.values()]}"
            )

        return ids

    def profile_of(self, class_id):
        """
        Find the gathering profile of a detected class.

        :param class_id: Class id given by the model.
        :return: Matching ResourceProfile.
        """
        return resources.resolve(self.classes[int(class_id)]["label"])

    def _confidence_of(self, class_id):
        """
        Lowest confidence to accept for one class.

        A single threshold for every class suits the model badly, because the classes
        are not learned equally well: the resources it has seen thousands of times are
        sure of themselves, while the trees are recognised correctly but scored low
        simply for being rare in the training set. A profile may therefore ask for its
        own bar, and only the classes that need it pay for it.

        :param class_id: Class id given by the model.
        :return: Confidence under which a box of that class is dropped.
        """
        confidence = self.profile_of(class_id).confidence

        return self.confidence if confidence is None else confidence

    def draw_boxes(self, img, coordinates):
        """
        Draw bounding boxes on the image.

        :param img: Input image.
        :param coordinates: Bounding box coordinates.
        """
        for coord in coordinates:
            x1, y1, x2, y2 = (int(value) for value in coord[:4])
            confidence, class_id = float(coord[4]), self.classes[int(coord[5])]
            cv.rectangle(img, (x1, y1), (x2, y2), class_id["color"], 2)
            label = f"{class_id['label']} {confidence:.2f}"
            cv.putText(img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_id["color"], 2)

        self.__cross_line(img)

        cv.drawMarker(img, (int(self.character_position_X), int(self.character_position_Y)), (255, 255, 255),
                      cv.MARKER_DIAMOND, 10, 2)
        self.__marker_closest(img, coordinates)
        # cv.imshow("Boxes", img)

    def __cross_line(self, img):
        cv.line(img, (0, int(self.character_position_Y)), (self.IMG_SIZE, int(self.character_position_Y)),
                (255, 255, 255), 2)
        cv.line(img, (int(self.character_position_X), 0), (int(self.character_position_X), self.IMG_SIZE),
                (255, 255, 255), 2)

    def __marker_closest(self, img, coordinates):
        closest = self.closest_point(coordinates)

        if closest is not None:
            cv.drawMarker(img, (int(closest[0]), int(closest[1])), (37, 150, 190), cv.MARKER_CROSS, 10, 2)
            cv.putText(img, "Closest", (int(closest[0]), int(closest[1]) + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)

    def centers(self, coordinates):
        """
        Compute the center of every box, from the closest to the character to the
        farthest.

        :param coordinates: Bounding box coordinates.
        :return: List of (center_x, center_y, class_id) in image coordinates.
        """
        centers = []

        for coord in coordinates:
            x1, y1, x2, y2 = (int(value) for value in coord[:4])

            center_x, center_y = ((x2 - x1) / 2) + x1, ((y2 - y1) / 2) + y1

            distance = sqrt(
                abs(self.character_position_X - center_x) ** 2 + abs(self.character_position_Y - center_y) ** 2)

            centers.append((distance, center_x, center_y, int(coord[5])))

        centers.sort(key=lambda center: center[0])

        return [(center_x, center_y, class_id) for _, center_x, center_y, class_id in centers]

    def closest_point(self, coordinates):
        """
        Find the box the closest to the character.

        :param coordinates: Bounding box coordinates.
        :return: (center_x, center_y, class_id) in image coordinates, None when
                 there is no box.
        """
        centers = self.centers(coordinates)

        if len(centers) == 0:
            return None

        return centers[0]

    def __convert_coordinates_to_screen_position(self, center_x, center_y):

        center_x = ((center_x * self.window_capture.window.width) / self.IMG_SIZE) + self.window_capture.window.left
        center_y = ((center_y * self.window_capture.window.height) / self.IMG_SIZE) + self.window_capture.window.top

        # The mouse is driven in whole pixels, and a torch value would leak here otherwise.
        return int(center_x), int(center_y)

    def image_position(self, screen_x, screen_y):
        """
        Turn a position on the screen back into one in the 640x640 image the model works
        with, the other way round from __convert_coordinates_to_screen_position.

        :param screen_x: Position on the screen.
        :param screen_y: Position on the screen.
        :return: (x, y) in image coordinates.
        """
        window = self.window_capture.window

        return (int((screen_x - window.left) * self.IMG_SIZE / window.width),
                int((screen_y - window.top) * self.IMG_SIZE / window.height))

    def character_screen_position(self):
        """
        Position of the character on the screen.

        :return: (x, y) in screen coordinates.
        """
        return self.__convert_coordinates_to_screen_position(self.character_position_X, self.character_position_Y)

    def _load_model(self):
        """
        Load the YOLOv5 model.

        :return: Loaded YOLOv5 model.
        """
        if not paths.YOLOV5.joinpath("hubconf.py").exists():
            raise Exception(
                f"The yolov5 submodule is missing in {paths.YOLOV5}, "
                f"get it with: git submodule update --init --recursive"
            )

        if not Path(self.model_name).exists():
            raise Exception(
                f"The trained weights are missing in {self.model_name}, "
                f"train them with yolov5 and drop the best.pt file there"
            )

        # Weights trained on Linux, on Google Colab for example, pickle the paths of
        # the machine they were trained on as PosixPath, and Windows refuses to build
        # one of those, so loading them fails before the first layer is read. Pointing
        # PosixPath at WindowsPath while the checkpoint is unpickled is enough, those
        # paths are never looked at again once the weights are in memory. It is put
        # back right after, so nothing else in the process sees the swap.
        posix = None

        if system() == "Windows":
            posix = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath

        try:
            model = torch.hub.load(str(paths.YOLOV5), 'custom', path=str(self.model_name), source="local",
                                   force_reload=True, verbose=True)

        except Exception as e:
            raise Exception(f"Failed to load the model: {e}")

        finally:
            if posix is not None:
                pathlib.PosixPath = posix

        model.conf = self.MODEL_FLOOR

        return model

    def scan(self, confidence_factor=1.0):
        """
        Take a picture of the game and resolve everything the model is sure enough of.

        Every box is kept here, and not only the wanted resources, because the caller
        needs the rest too: the roaming avoids the monsters, and following a node while
        it is being gathered needs it found again whatever was asked for.

        :param confidence_factor: Multiplier applied to every threshold, under 1 to
                                  accept boxes the bot would not have picked a node
                                  from. Following an already found node uses it, see
                                  Interaction.find_node.
        :return: (detections, image, boxes), detections being (screen_x, screen_y,
                 profile) ordered from the closest to the character to the farthest.
        """
        img = self._process_image(self.window_capture.screenshot())
        boxes = [box for box in self.model(img, augment=self.AUGMENT).xyxy[0]
                 if float(box[4]) > self._confidence_of(box[5]) * confidence_factor]

        detections = []

        for center_x, center_y, class_id in self.centers(boxes):
            screen_x, screen_y = self.__convert_coordinates_to_screen_position(center_x, center_y)
            detections.append((screen_x, screen_y, self.profile_of(class_id)))

        return detections, img, boxes

    def predict(self, ignore=None):
        """
        Make predictions using the YOLOv5 model.

        :param ignore: Callable taking a screen position and returning True when the
                       resource has to be skipped, None to keep all of them.
        :return: Screen position, profile of the closest resource and the image,
                 position and profile are None when nothing has been detected.
        """
        loop_time = time()
        detections, img, boxes = self.scan()

        self.last_hazards = [(x, y) for x, y, profile in detections if profile is resources.DEFAULT]

        if self.preview:
            self.draw_boxes(img, [box for box in boxes if int(box[5]) in self.target_ids])
            cv.putText(img, f'FPS {1 / (time() - loop_time):.1f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            cv.imshow(game.PREVIEW_WINDOW_TITLE, img)

        for screen_x, screen_y, profile in detections:
            if profile not in self.targets:
                continue

            if ignore is not None and ignore(screen_x, screen_y):
                continue

            return screen_x, screen_y, profile, img

        return None, None, None, img
