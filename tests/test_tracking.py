import cv2 as cv
import pytest

from albion_bot.vision import resources, tracking
from albion_bot.vision.detector import Detection

H, W = 540, 960
CHARACTER = (480, 221)


def view(world, x, y):
    frame = world[y:y + H, x:x + W].copy()
    cv.rectangle(frame, (0, 0), (W, 40), (40, 40, 40), -1)  # interface, never moves
    cv.circle(frame, CHARACTER, 30, (0, 0, 200), -1)  # character, never moves
    return frame


@pytest.mark.parametrize("move", [(0, 0), (8, -5), (60, 40), (-150, 90), (250, -120)])
def test_camera_shift(world, move):
    dx, dy, response = tracking.camera_shift(view(world, 1000, 1000),
                                             view(world, 1000 + move[0], 1000 + move[1]), CHARACTER)

    assert response > tracking.MIN_RESPONSE
    assert (dx, dy) == pytest.approx((-move[0], -move[1]), abs=2)


def test_reidentify_picks_the_same_node(world):
    before = view(world, 1000, 1000)
    after = view(world, 1100, 1050)

    def box(x, y):
        return Detection(x - 25, y - 25, x + 25, y + 25, 0.9, "tree", resources.TREE)

    target = box(700, 300)
    picture = tracking.node_picture(before, target)
    same, other = box(600, 250), box(300, 400)

    found, score = tracking.reidentify(picture, [other, same], after)
    assert found is same
    assert score > 0.9
