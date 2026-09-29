"""
How far the camera moved, so what the bot remembers about places on screen follows the
world when the character walks. The camera follows the character, every step it takes
slides the whole world the other way on screen.
"""

import cv2 as cv
import numpy as np

from albion_bot import geometry

# Width frames are shrunk to before being compared, plenty for the ground to line up.
WIDTH = 320

# Part of the frame compared: the interface of the game sits on the edges and never
# moves, it would pull every answer towards no move at all.
AREA = (0.12, 0.08, 0.76, 0.70)

# Radius of the character, painted over for the same reason, fraction of the height.
CHARACTER_RADIUS = 0.09

# Response of the correlation under which the answer is not trusted.
MIN_RESPONSE = 0.08

# Match score over which a node found again is taken for the same one.
REIDENTIFY_SCORE = 0.35


def _prepare(frame, character):
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = WIDTH / grey.shape[1]
    small = cv.resize(grey, (WIDTH, max(int(grey.shape[0] * scale), 8)), interpolation=cv.INTER_AREA)

    mask = np.zeros_like(small)
    center = (int(character[0] * scale), int(character[1] * scale))
    cv.circle(mask, center, int(small.shape[0] * CHARACTER_RADIUS), 255, -1)
    small = cv.inpaint(small, mask, 3, cv.INPAINT_TELEA)

    x1, y1, x2, y2 = geometry.region_pixels(small.shape, AREA)
    return np.float32(small[y1:y2, x1:x2]), scale


def camera_shift(previous, current, character):
    """
    :param previous: Earlier BGR frame.
    :param current: Later BGR frame, same size.
    :param character: (x, y) of the character in the frames.
    :return: (dx, dy, response): the world moved by (dx, dy) frame pixels on screen, and
             how clear the answer is, trust it over MIN_RESPONSE.
    """
    if previous.shape != current.shape:
        return 0.0, 0.0, 0.0

    a, scale = _prepare(previous, character)
    b, _ = _prepare(current, character)
    window = cv.createHanningWindow((a.shape[1], a.shape[0]), cv.CV_32F)
    (dx, dy), response = cv.phaseCorrelate(a, b, window)

    return dx / scale, dy / scale, float(response)


def node_picture(frame, detection):
    """
    :return: Grey picture of a detected node, to find it again later, None when empty.
    """
    x1, y1, x2, y2 = detection.box()
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    return cv.cvtColor(frame[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)


def reidentify(picture, candidates, frame, near=None, max_distance=None):
    """
    Find a node again among the detections of a later frame, by the way it looks.

    :param picture: Given by node_picture.
    :param candidates: Detections to pick from, of the same kind as the node.
    :param near: (x, y) the node has to be close to, None for anywhere.
    :param max_distance: How close, in frame pixels.
    :return: (detection, score), None when none looks like it.
    """
    if picture is None:
        return None

    best = None

    for candidate in candidates:
        if near is not None and max_distance is not None and geometry.distance(candidate.center, near) > max_distance:
            continue

        other = node_picture(frame, candidate)

        if other is None:
            continue

        other = cv.resize(other, (picture.shape[1], picture.shape[0]), interpolation=cv.INTER_AREA)
        score = float(np.nan_to_num(cv.matchTemplate(other, picture, cv.TM_CCOEFF_NORMED).max(), nan=-1.0))

        if best is None or score > best[1]:
            best = (candidate, score)

    if best is None or best[1] < REIDENTIFY_SCORE:
        return None

    return best
