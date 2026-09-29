import cv2 as cv
import numpy as np
import pytest

from albion_bot.vision.gather_bar import GatherBar


def frame(width=960, height=540, bar=True, seed=0):
    random = np.random.default_rng(seed)
    image = (random.random((height, width, 3)) * 80).astype(np.uint8)

    if bar:
        scale = height / 540
        x1, y1, x2, y2 = (int(v * scale) for v in (430, 290, 530, 304))
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv.rectangle(image, (x1 + 2, y1 + 2), (x1 + (x2 - x1) * 2 // 3, y2 - 2), (0, 180, 0), -1)

    return image


BOX = (425 / 960, 286 / 540, 110 / 960, 22 / 540)


def test_seen_when_there_and_not_otherwise():
    bar = GatherBar.calibrate(frame(), BOX)
    assert bar.score(frame(seed=1)) > 0.9
    assert not bar.visible(frame(bar=False, seed=2))


def test_survives_a_resized_window():
    bar = GatherBar.calibrate(frame(), BOX)
    assert bar.visible(frame(1280, 720, seed=3))
    assert not bar.visible(frame(1280, 720, bar=False, seed=4))


def test_flat_box_is_refused():
    flat = np.full((540, 960, 3), 50, np.uint8)

    with pytest.raises(ValueError):
        GatherBar.calibrate(flat, BOX)


def test_save_and_load():
    GatherBar.calibrate(frame(), BOX).save()
    loaded = GatherBar.load()
    assert loaded is not None and loaded.visible(frame(seed=5))
