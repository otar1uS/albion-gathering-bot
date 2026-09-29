"""
The bar the game shows while the character gathers, the only sign the bot has that a
node is being gathered, and that it is over.

It is calibrated once: a screenshot is taken while gathering, the user draws a box around
the bar, and that picture is looked for, a bit around where it was, from then on.
"""

import json

import cv2 as cv
import numpy as np

from albion_bot import geometry, paths

# Margin around the box the bar is looked for in, as a fraction of its size. The bar
# moves a bit with the camera and the size of the character.
SEARCH_MARGIN = 0.6

# Spread of the grey levels under which a picture is flat, see usable.
FLATNESS = 6.0


def _grey(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image


class GatherBar:

    def __init__(self, template, box, frame_height, threshold=0.7):
        """
        :param template: Grey picture of the bar.
        :param box: Where the bar was, fractions of the frame.
        :param frame_height: Height of the frame the picture was taken from.
        :param threshold: Score over which the bar is on screen.
        """
        self.template = template
        self.box = tuple(box)
        self.frame_height = frame_height
        self.threshold = threshold
        self.search = geometry.expand(self.box, SEARCH_MARGIN)
        self._scaled = {}

    @classmethod
    def calibrate(cls, frame, box, threshold=0.7):
        """
        :param frame: BGR frame taken while gathering.
        :param box: Box drawn around the bar, fractions of the frame.
        """
        template = _grey(geometry.crop(frame, box)).copy()

        if not usable(template):
            raise ValueError("the box holds a flat picture, the character was most likely not gathering, "
                             "or the box misses the bar")

        return cls(template, box, frame.shape[0], threshold)

    def save(self):
        paths.DATA.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(paths.GATHER_BAR), self.template)
        paths.GATHER_BAR_INFO.write_text(json.dumps({"box": list(self.box), "frame_height": self.frame_height}))

    @classmethod
    def load(cls, threshold=0.7):
        """
        :return: The calibrated bar, None when it has not been calibrated.
        """
        template = cv.imread(str(paths.GATHER_BAR), cv.IMREAD_GRAYSCALE)

        try:
            info = json.loads(paths.GATHER_BAR_INFO.read_text())
        except (OSError, ValueError):
            return None

        if template is None or not geometry.valid_region(info.get("box")) or not info.get("frame_height"):
            return None

        return cls(template, info["box"], int(info["frame_height"]), threshold)

    def _template_for(self, frame_height):
        """
        :return: The picture of the bar at the scale of a frame, the game window may have
                 been resized since the calibration.
        """
        if frame_height not in self._scaled:
            scale = frame_height / self.frame_height
            template = self.template

            if abs(scale - 1) > 0.01:
                size = (max(int(template.shape[1] * scale), 4), max(int(template.shape[0] * scale), 4))
                template = cv.resize(template, size, interpolation=cv.INTER_AREA)

            self._scaled[frame_height] = template

        return self._scaled[frame_height]

    def score(self, frame):
        """
        :return: How much the bar is on screen, between -1 and 1.
        """
        area = _grey(geometry.crop(frame, self.search))
        template = self._template_for(frame.shape[0])

        if area.shape[0] < template.shape[0] or area.shape[1] < template.shape[1]:
            return -1.0

        result = cv.matchTemplate(area, template, cv.TM_CCOEFF_NORMED)
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return float(result.max())

    def visible(self, frame):
        return self.score(frame) >= self.threshold


def usable(template):
    """
    A picture with nothing in it matches noise, the bot would either never see the bar or
    never see it go.
    """
    return float(template.std()) >= FLATNESS
