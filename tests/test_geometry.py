import numpy as np
import pytest

from albion_bot import geometry


def test_crop_and_expand():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert geometry.crop(frame, (0.5, 0.5, 0.25, 0.1)).shape == (10, 50, 3)
    assert geometry.expand((0.0, 0.0, 0.2, 0.2), 0.5) == pytest.approx((0.0, 0.0, 0.3, 0.3))

    with pytest.raises(ValueError):
        geometry.crop(frame, (0.99, 0.99, 0.001, 0.001))


def test_valid_region():
    assert geometry.valid_region([0.1, 0.1, 0.5, 0.5])
    assert not geometry.valid_region([0.8, 0.1, 0.5, 0.5])
    assert not geometry.valid_region("nope")
