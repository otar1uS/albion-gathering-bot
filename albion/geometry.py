"""
Rectangles and the two coordinate systems the bot lives in.

Everything the model says is in the 640x640 square the window is squeezed into, and
everything the mouse does is in screen pixels, and mixing the two up silently sends
clicks a thousand pixels from where they were meant to go. Both conversions live here so
there is one place to be right.
"""

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Rect:
    """Where the game is on the screen, decoration already taken off."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return self.left + self.width // 2, self.top + self.height // 2

    def contains(self, x, y):
        return self.left <= x < self.right and self.top <= y < self.bottom

    def fraction(self, fx, fy):
        """
        A point given as a fraction of the window, in screen pixels.

        :param fx: Fraction across, 0 at the left edge.
        :param fy: Fraction down, 0 at the top edge.
        :return: (x, y) on screen.
        """
        return int(self.left + self.width * fx), int(self.top + self.height * fy)

    def to_screen(self, image_x, image_y, image_size):
        """
        Turn a position in the model's square into one on the screen.

        :param image_x: Position across the model image.
        :param image_y: Position down the model image.
        :param image_size: Side of the square the window was squeezed into.
        :return: (x, y) on screen, whole pixels because the mouse is driven in them.
        """
        return (int(image_x * self.width / image_size + self.left),
                int(image_y * self.height / image_size + self.top))

    def to_image(self, screen_x, screen_y, image_size):
        """
        Turn a position on the screen back into one in the model's square.

        :param screen_x: Position on the screen.
        :param screen_y: Position on the screen.
        :param image_size: Side of the square the window was squeezed into.
        :return: (x, y) in the model image.
        """
        return (int((screen_x - self.left) * image_size / self.width),
                int((screen_y - self.top) * image_size / self.height))

    def __str__(self):
        return f"{self.width}x{self.height} at ({self.left}, {self.top})"


def distance(a, b):
    """
    How far apart two points are.

    :param a: (x, y).
    :param b: (x, y).
    :return: Distance in the units the points were given in.
    """
    return hypot(a[0] - b[0], a[1] - b[1])
