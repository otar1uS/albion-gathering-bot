"""
Positions shared by the whole bot.

Everything the bot looks at is measured in pixels of the captured frame, whatever the
size of the window, and everything configured by the user is a fraction of the frame, so
changing the resolution of the game keeps the settings right.
"""

from dataclasses import dataclass

import numpy as np

# left, top, width, height, as fractions of the frame.
Region = tuple[float, float, float, float]


@dataclass(frozen=True)
class Rect:
    """A rectangle on the screen, in physical pixels."""

    left: int
    top: int
    width: int
    height: int

    def __str__(self):
        return f"{self.width}x{self.height} at ({self.left}, {self.top})"


def region_pixels(shape, region: Region):
    """
    :param shape: Shape of the frame.
    :param region: Fractions of the frame.
    :return: (x1, y1, x2, y2) in pixels, clamped to the frame.
    """
    height, width = shape[:2]
    left, top, region_width, region_height = region

    x1, y1 = int(round(width * left)), int(round(height * top))
    x2, y2 = int(round(width * (left + region_width))), int(round(height * (top + region_height)))

    return max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)


def crop(frame, region: Region, minimum=4):
    """
    :return: The part of the frame inside the region.
    """
    x1, y1, x2, y2 = region_pixels(frame.shape, region)

    if x2 - x1 < minimum or y2 - y1 < minimum:
        raise ValueError(f"the region {region} holds nothing of a {frame.shape[1]}x{frame.shape[0]} frame")

    return frame[y1:y2, x1:x2]


def expand(region: Region, margin: float) -> Region:
    """
    :param margin: Added on every side, as a fraction of the size of the region.
    :return: The larger region, kept inside the frame.
    """
    left, top, width, height = region
    x1, y1 = max(left - width * margin, 0.0), max(top - height * margin, 0.0)
    x2, y2 = min(left + width * (1 + margin), 1.0), min(top + height * (1 + margin), 1.0)

    return x1, y1, x2 - x1, y2 - y1


def valid_region(value) -> bool:
    try:
        left, top, width, height = (float(v) for v in value)
    except (TypeError, ValueError):
        return False

    return 0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1 \
        and left + width <= 1.0001 and top + height <= 1.0001


def point(shape, fractions):
    """
    :return: (x, y) in pixels of a point given as fractions of the frame.
    """
    height, width = shape[:2]
    return width * fractions[0], height * fractions[1]


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
