"""
Capture the name the game writes over a resource, at full resolution.

The verification only has to answer "did anything light up", which the 640 pixel frame the
model works in is plenty for. Reading *what* the game wrote is a different job: at that
scale the text is four pixels tall and is mush. So this takes the crop from the window's
own pixels before anything is downscaled, and writes it out big, so the question of how
to read it can be answered by looking at the thing rather than by guessing.

What matters is whether Albion prints the tier as a short marker like "T4" or only as the
resource name, because recognising two characters and recognising a vocabulary of forty
names are very different problems.

    .venv\\Scripts\\python.exe training\\probe_label.py
"""

import sys
from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albion import logs
from albion.config import Config
from albion.control import input as input_module
from albion.game import resources
from albion.nav.navigator import Navigator
from albion.vision import capture as capture_module
from albion.vision.detector import Detector
from albion.vision.verify import Verifier

# How much of the window around the cursor to keep, in native pixels. The name is drawn
# above the node and can be a good deal wider than the trunk it belongs to.
ACROSS = 260
ABOVE = 150
BELOW = 60


def native_crop(capture, screen_point):
    """
    Cut a piece of the window around a point, at the window's own resolution.

    :param capture: Capture.
    :param screen_point: (x, y) on screen.
    :return: BGR crop, or None when the point is outside the window.
    """
    picture = capture.grab()

    if picture is None or picture.size == 0:
        return None

    rect = capture.rect
    x = screen_point[0] - rect.left
    y = screen_point[1] - rect.top

    height, width = picture.shape[:2]

    left, right = max(x - ACROSS, 0), min(x + ACROSS, width)
    top, bottom = max(y - ABOVE, 0), min(y + BELOW, height)

    if right <= left or bottom <= top:
        return None

    return np.ascontiguousarray(picture[top:bottom, left:right, :3])


def main():
    config = Config.load()
    logs.setup("INFO")
    log = logs.get("probe_label")

    capture = capture_module.build(config)
    controller = input_module.build(config)
    detector = Detector(config, capture)
    verifier = Verifier(config, capture, controller)
    navigator = Navigator(config, capture, controller, detector)

    out = Path("images/label_probe")
    out.mkdir(parents=True, exist_ok=True)

    found = None

    for attempt in range(30):
        capture.focus()
        capture.refresh()
        controller.park(capture.rect)
        sleep(0.4)

        detections, clean = detector.look()
        candidates = [d for d in detections if d.gatherable]
        candidates.sort(key=lambda d: -d.confidence)

        for candidate in candidates:
            if verifier.confirm(candidate, clean):
                found = candidate
                break

        if found is not None:
            break

        log.info("nothing the game will name yet, roaming (%d/30)", attempt + 1)
        navigator.roam(detections)
        controller.wait(config.navigation.wait)

    if found is None:
        sys.exit("no gatherable node found after roaming")

    log.info("%s at %.2f, hovering it again to photograph the name",
             found.label, found.confidence)

    # Parked first so the "before" crop is the same piece of world without the name on it,
    # which is what tells the writing apart from the tree behind it.
    controller.park(capture.rect)
    sleep(0.6)
    before = native_crop(capture, found.click)

    controller.move(*found.click)
    controller.wait(config.verify.settle)
    after = native_crop(capture, found.click)

    controller.park(capture.rect)

    if before is None or after is None:
        sys.exit("the crop fell outside the window")

    cv.imwrite(str(out / "before.png"), before)
    cv.imwrite(str(out / "after.png"), after)

    # Upscaled with nearest neighbour on purpose: smoothing invents strokes that are not
    # there, and the question is what the glyphs really look like.
    big = cv.resize(after, None, fx=3, fy=3, interpolation=cv.INTER_NEAREST)
    cv.imwrite(str(out / "after_big.png"), big)

    difference = cv.absdiff(after, before)
    cv.imwrite(str(out / "difference.png"),
               cv.resize(cv.convertScaleAbs(difference, alpha=3), None, fx=3, fy=3,
                         interpolation=cv.INTER_NEAREST))

    # Where the writing actually is, so a sensible crop can be chosen rather than guessed.
    grey = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    lit = np.argwhere(grey > 40)

    log.info("crop is %dx%d native pixels", after.shape[1], after.shape[0])

    if lit.size:
        top, left = lit.min(axis=0)
        bottom, right = lit.max(axis=0)
        log.info("the name appears at x %d..%d, y %d..%d within the crop "
                 "(cursor sits at x %d, y %d)", left, right, top, bottom,
                 min(ACROSS, after.shape[1] // 2), ABOVE)
        log.info("so relative to the cursor: %d..%d across, %d..%d above",
                 left - ACROSS, right - ACROSS, ABOVE - bottom, ABOVE - top)

        pad = 6
        name = after[max(top - pad, 0):bottom + pad, max(left - pad, 0):right + pad]
        cv.imwrite(str(out / "name.png"),
                   cv.resize(name, None, fx=4, fy=4, interpolation=cv.INTER_NEAREST))
        log.info("wrote name.png, the writing alone at 4x")
    else:
        log.warning("nothing changed between the two crops, the name was not captured")

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
