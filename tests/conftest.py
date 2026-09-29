import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def textured_world(width=4000, height=3000, seed=0):
    """Ground to walk on: noise and blobs, something the camera tracking can lock onto."""
    random = np.random.default_rng(seed)
    world = cv.resize((random.random((height // 10, width // 10)) * 255).astype(np.uint8), (width, height),
                      interpolation=cv.INTER_CUBIC)

    for _ in range(width * height // 6000):
        center = tuple(int(v) for v in random.integers(0, [width, height]))
        cv.circle(world, center, int(random.integers(4, 30)), int(random.integers(0, 255)), -1)

    return cv.cvtColor(cv.GaussianBlur(world, (5, 5), 0), cv.COLOR_GRAY2BGR)


@pytest.fixture(scope="session")
def world():
    return textured_world()


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Nothing a test writes lands in the real data folder."""
    from albion_bot import paths

    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(paths, "SETTINGS", tmp_path / "data" / "settings.json")
    monkeypatch.setattr(paths, "ROUTES", tmp_path / "data" / "routes")
    monkeypatch.setattr(paths, "GATHER_BAR", tmp_path / "data" / "gather_bar.png")
    monkeypatch.setattr(paths, "GATHER_BAR_INFO", tmp_path / "data" / "gather_bar.json")
    monkeypatch.setattr(paths, "MODELS", tmp_path / "models")
    monkeypatch.setattr(paths, "MODEL", tmp_path / "models" / "best.pt")
    monkeypatch.setattr(paths, "MODEL_OPENVINO", tmp_path / "models" / "best_openvino_model")
