"""
Reading the character's position off the minimap.

Until this file existed the bot had no idea where it was. It could see what was in front
of it and nothing else, so "walk somewhere new" meant an expanding square drawn blind and
"am I stuck" meant asking whether the picture had changed lately. Both are guesses, and
both are why the character looks like it is wandering: it is.

The tools that do this well read the game's network stream and are handed world
coordinates for everything. This project does not do that, and the minimap is the nearest
honest substitute. It is drawn in a fixed corner, it does not rotate with the camera, and
the character is a bright blue arrow on it. That gives a position in the zone, which is
not a world coordinate but is enough for every question the bot actually asks: have I
moved, have I been here before, and how do I get back to the spot the trees were in.

Measured on a 2560x1440 client: the arrow masks out as a single blob of 72 pixels, the
next largest thing passing the same filter is 19, and the whole read costs about a
millisecond against the 157ms the model already spends.
"""

import cv2 as cv
import numpy as np

from .. import logs
from ..geometry import distance

log = logs.get("nav.minimap")


class Minimap:
    """Where the character is standing, as a fraction of the zone map."""

    def __init__(self, config, capture):
        self.config = config
        self.capture = capture

        # Complained about once rather than every pass. A minimap that cannot be read is
        # worth knowing about, but it is not worth a line a second for an hour.
        self.warned = False

    def box(self, picture):
        """
        The part of the picture the minimap is drawn in.

        Held as fractions of the window rather than pixels so it survives a different
        resolution, which pixels would not: this was written against a 2560x1440 client
        and the client is whatever the user runs it at.

        :param picture: The captured frame.
        :return: (left, top, right, bottom) in picture pixels.
        """
        settings = self.config.minimap
        height, width = picture.shape[:2]

        return (int(width * settings.left), int(height * settings.top),
                int(width * settings.right), int(height * settings.bottom))

    def where(self, picture=None):
        """
        Find the character on the minimap.

        :param picture: A frame already captured, or None to take one.
        :return: (x, y) as fractions of the minimap, or None when the arrow is not there.
        """
        if not self.config.minimap.enabled:
            return None

        if picture is None:
            picture = self.capture.grab()

        if picture is None or picture.size == 0:
            return None

        settings = self.config.minimap
        left, top, right, bottom = self.box(picture)

        if right <= left or bottom <= top:
            return None

        region = picture[top:bottom, left:right, :3]
        hsv = cv.cvtColor(region, cv.COLOR_BGR2HSV)

        mask = cv.inRange(
            hsv,
            np.array([settings.hue_low, settings.saturation, settings.value]),
            np.array([settings.hue_high, 255, 255]))

        count, _, stats, centres = cv.connectedComponentsWithStats(mask, 8)

        # The arrow is the biggest thing of that colour by a wide margin. Water on the
        # map passes the hue but not the brightness, and what little of it does comes
        # through as specks a fifth the size.
        best, area = None, 0

        for index in range(1, count):
            size = int(stats[index, cv.CC_STAT_AREA])

            if size > area:
                best, area = index, size

        if best is None or area < settings.smallest:
            if not self.warned:
                self.warned = True
                log.warning("cannot find the character arrow on the minimap. Either the "
                            "minimap is hidden, or it is not where minimap.left/top/"
                            "right/bottom say it is. Position tracking is off until it "
                            "comes back.")

            return None

        self.warned = False
        across, down = right - left, bottom - top

        return (float(centres[best][0]) / across, float(centres[best][1]) / down)

    def travelled(self, before, after):
        """
        How far the character has moved, as a fraction of the zone.

        :param before: Position from where(), or None.
        :param after: Position from where(), or None.
        :return: Distance, or None when either reading is missing.
        """
        if before is None or after is None:
            return None

        return distance(before, after)
