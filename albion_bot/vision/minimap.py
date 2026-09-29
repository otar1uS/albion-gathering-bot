"""
Where the character is, read from the minimap.

The minimap of Albion follows the character: the arrow stays in its middle and the land
slides under it. A piece of the minimap saved at some place is found again in a later
minimap, shifted by the way walked since, and that shift says where that place is from
the character now. A route is a string of such pieces, see albion_bot.navigation.route.
"""

import cv2 as cv
import numpy as np

from albion_bot import geometry

# Size of the piece of minimap saved at each waypoint, as a fraction of the minimap. The
# rest of the minimap is the margin it can be found in, so a waypoint is found as long
# as the character is closer to it than (1 - PATCH) / 2 of the minimap.
PATCH = 0.5

# Radius of the arrow of the character, as a fraction of the minimap. It is the same in
# every picture, so it is painted over, or every match would pull towards no move.
ARROW = 0.07

# Match score under which a waypoint is considered not on the minimap.
LOST_SCORE = 0.45

# The minimap is resized to this width before anything, so the numbers above work the
# same on every screen and matching stays cheap.
WIDTH = 240


def crop(frame, region):
    """
    Cut the minimap out of a frame of the game window.

    :param frame: BGR frame of the game window.
    :param region: (left, top, width, height) as fractions of the frame.
    :return: BGR minimap, resized to WIDTH.
    """
    minimap = geometry.crop(frame, region, minimum=8)

    return cv.resize(minimap, (WIDTH, max(int(WIDTH * minimap.shape[0] / minimap.shape[1]), 8)))


def read(frame, region):
    """
    :return: Minimap of a frame, prepared for matching.
    """
    return prepare(crop(frame, region))


def prepare(minimap):
    """
    Turn a minimap into what is matched: grey, contrast stretched, arrow painted over.

    :param minimap: BGR minimap given by crop.
    :return: Grey image.
    """
    grey = cv.cvtColor(minimap, cv.COLOR_BGR2GRAY)
    grey = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(grey)

    height, width = grey.shape
    mask = np.zeros_like(grey)
    cv.circle(mask, (width // 2, height // 2), int(min(width, height) * ARROW), 255, -1)

    return cv.inpaint(grey, mask, 3, cv.INPAINT_TELEA)


def patch(prepared):
    """
    :param prepared: Minimap given by prepare.
    :return: Middle of the minimap, what a waypoint remembers.
    """
    height, width = prepared.shape
    size = int(min(width, height) * PATCH)
    top, left = (height - size) // 2, (width - size) // 2

    return prepared[top:top + size, left:left + size].copy()


def locate(waypoint, prepared):
    """
    Find where a waypoint is, from the character.

    :param waypoint: Patch saved at the waypoint.
    :param prepared: Current minimap given by prepare.
    :return: (dx, dy, score), the way from the character to the waypoint in minimap
             pixels, and how sure the match is, between -1 and 1.
    """
    result = cv.matchTemplate(prepared, waypoint, cv.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, score, _, (x, y) = cv.minMaxLoc(result)

    height, width = prepared.shape
    size_y, size_x = waypoint.shape

    return x + size_x / 2 - width / 2, y + size_y / 2 - height / 2, float(score)


def arrive_distance(prepared):
    """
    :return: Distance in minimap pixels under which a waypoint is reached.
    """
    return min(prepared.shape) * 0.04


def step_distance(prepared):
    """
    :return: Distance in minimap pixels between two waypoints of a recording, well
             inside the margin a waypoint can be found in.
    """
    return min(prepared.shape) * (1 - PATCH) / 2 * 0.5
