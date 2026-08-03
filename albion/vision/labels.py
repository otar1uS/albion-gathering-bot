"""
Reading the name Albion writes over the node under the cursor.

This runs on the window's own pixels rather than on the 640 pixel square the model works
in. At that scale the writing is four pixels tall and is mush; at native resolution it is
crisp enough to read straight off, measured at 0.86 confidence with no upscaling at all.

It is only ever asked once per node, during the hover that already happens to tell a real
resource from scenery, so its 600ms costs nothing: the walk to that node is twenty times
longer. The engine is RapidOCR, which is ONNX and pip installable with no system binary to
chase, and which is also where this project is heading for packaging.

The enchantment is not in the text. Albion writes an enchanted name in the rarity colour
of its level, so it is read from the colour of the brightest strokes rather than from the
words.
"""

import cv2 as cv
import numpy as np

from .. import logs
from ..game import tiers

log = logs.get("vision.labels")


class LabelReader:
    """Reads the tooltip the game draws over whatever the cursor is on."""

    def __init__(self, config, capture):
        self.config = config
        self.capture = capture
        self.engine = None
        self.broken = False

    def __load(self):
        """
        Bring up the OCR engine, once, and only if it is ever actually wanted.

        :return: The engine, or None when it is not installed.
        """
        if self.engine is not None or self.broken:
            return self.engine

        try:
            from rapidocr_onnxruntime import RapidOCR

            self.engine = RapidOCR()
            log.info("label reading is on, tier and enchantment filters will be applied")
        except Exception as error:
            # Not fatal. Without it the bot gathers everything it finds, which is what it
            # did before tiers existed, and says so once rather than every node.
            self.broken = True
            log.warning("no OCR engine (%s), so tiers cannot be read and every tier will "
                        "be gathered. Install it with: pip install rapidocr_onnxruntime",
                        error)

        return self.engine

    def crop(self, screen_point):
        """
        Cut the piece of the window the tooltip is drawn in, at native resolution.

        :param screen_point: (x, y) on screen the cursor is resting on.
        :return: BGR crop, or None when it falls outside the window.
        """
        picture = self.capture.grab()

        if picture is None or picture.size == 0:
            return None

        settings = self.config.labels
        rect = self.capture.rect
        x, y = screen_point[0] - rect.left, screen_point[1] - rect.top
        height, width = picture.shape[:2]

        left, right = max(x - settings.across, 0), min(x + settings.across, width)
        top, bottom = max(y - settings.above, 0), min(y + settings.below, height)

        if right <= left or bottom <= top:
            return None

        return np.ascontiguousarray(picture[top:bottom, left:right, :3])

    def read(self, screen_point):
        """
        Read whatever the game has written near a point.

        :param screen_point: (x, y) on screen the cursor is resting on.
        :return: tiers.Named, with resource None when nothing could be read.
        """
        engine = self.__load()

        if engine is None:
            return tiers.Named(raw="")

        picture = self.crop(screen_point)

        if picture is None:
            return tiers.Named(raw="")

        try:
            result, _ = engine(picture)
        except Exception as error:
            log.debug("the OCR engine failed on this crop: %s", error)
            return tiers.Named(raw="")

        if not result:
            return tiers.Named(raw="")

        # The tooltip is one short line. Anything else in the crop is a player name or a
        # chat message that happened to be near, so the longest confident line wins.
        lines = [(text, float(score), box) for box, text, score in result
                 if float(score) >= self.config.labels.confidence]

        if not lines:
            return tiers.Named(raw="")

        text, score, box = max(lines, key=lambda line: len(line[0]) * line[1])
        enchantment = self.__enchantment(picture, box)
        found = tiers.identify(text, enchantment)

        log.debug("read %r at %.2f -> %s", text, score, found)

        return found

    def __enchantment(self, picture, box):
        """
        Work out the enchantment level from the colour the name is written in.

        :param picture: The crop the name was read from.
        :param box: Corner points of the text, as the engine gave them.
        :return: Enchantment level, 0 for a plain resource.
        """
        try:
            points = np.array(box, dtype=np.int32).reshape(-1, 2)
            left, top = points.min(axis=0)
            right, bottom = points.max(axis=0)

            height, width = picture.shape[:2]
            left, top = max(int(left), 0), max(int(top), 0)
            right, bottom = min(int(right), width), min(int(bottom), height)

            if right <= left or bottom <= top:
                return 0

            writing = picture[top:bottom, left:right]

            # Only the strokes themselves carry the colour. The tooltip behind them is
            # nearly black, and averaging the whole box would drag every colour towards
            # it and make every enchantment look like every other one.
            grey = cv.cvtColor(writing, cv.COLOR_BGR2GRAY)
            strokes = writing[grey >= max(int(grey.max()) - 40, 120)]

            if strokes.size == 0:
                return 0

            colour = strokes.reshape(-1, 3).mean(axis=0)
        except Exception:
            return 0

        # Nearest rarity colour, in plain distance. The writing is anti aliased over
        # whatever is behind it, so an exact match is not on offer.
        level, best = 0, None

        for candidate, reference in tiers.ENCHANTMENT_COLOURS.items():
            gap = float(np.linalg.norm(colour - np.array(reference, dtype=float)))

            if best is None or gap < best:
                level, best = candidate, gap

        # White is much closer to every other colour than they are to each other, so a
        # reading has to be clearly nearer an enchantment colour than to white to count.
        if level:
            plain = float(np.linalg.norm(
                colour - np.array(tiers.ENCHANTMENT_COLOURS[0], dtype=float)))

            if plain - best < self.config.labels.enchantment_margin:
                return 0

        return level
