"""
Watch one hover happen, step by step, and keep the pictures.

The verification reported the identical bright pixel count before and after every hover,
for a dozen trees in a row including ones the model was very sure of. Identical is the
suspicious part: two grabs of a live game a second apart are never exactly equal, so
either the cursor never went where it was told, or the two frames being compared are the
same frame. This prints where the cursor actually ended up and writes both frames out so
the difference can be looked at rather than summarised.

    .venv\\Scripts\\python.exe training\\probe_hover.py
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
from albion.vision import capture as capture_module
from albion.vision.detector import Detector, process


def main():
    config = Config.load()
    logs.setup("INFO")

    capture = capture_module.build(config)
    controller = input_module.build(config)
    detector = Detector(config, capture)

    out = Path("images/hover_probe")
    out.mkdir(parents=True, exist_ok=True)

    capture.focus()
    sleep(1.0)

    controller.park(capture.rect)
    sleep(0.6)

    from albion.nav.navigator import Navigator

    navigator = Navigator(config, capture, controller, detector)
    trees, clean = [], None

    # The character rarely happens to be looking at a tree when this is started, so it
    # is walked about until one turns up, the same way the bot does it.
    for attempt in range(20):
        capture.focus()
        capture.refresh()
        controller.park(capture.rect)
        sleep(0.5)

        detections, clean = detector.look()
        trees = [d for d in detections if d.profile is resources.TREE]

        if trees:
            break

        print(f"  no tree in sight, roaming ({attempt + 1}/20)")
        navigator.roam(detections)
        controller.wait(config.navigation.wait)

    if not trees:
        sys.exit("no tree found after roaming, move the character into a forest")

    target = max(trees, key=lambda d: d.confidence)
    print(f"tree at {target.confidence:.2f}, screen {target.screen}, box {target.box}")
    print(f"cursor parked at {controller.position()}")

    cv.imwrite(str(out / "clean.png"), clean)

    controller.move(*target.screen)
    sleep(config.verify.settle)

    where = controller.position()
    print(f"cursor asked for {target.screen}, actually at {tuple(where)}")

    hovered = process(capture.grab(), config.vision.image_size)
    cv.imwrite(str(out / "hovered.png"), hovered)

    change = float(np.abs(hovered.astype(int) - clean.astype(int)).mean())
    print(f"whole frame changed by {change:.3f} between the two grabs")

    if change == 0.0:
        print("  -> the two grabs are the SAME IMAGE. The capture is handing back a "
              "stale frame, which makes every comparison meaningless.")

    image_size = config.vision.image_size
    x, y = capture.rect.to_image(*target.screen, image_size)
    radius = config.verify.radius
    left, top = max(x - radius, 0), max(y - radius, 0)
    right, bottom = min(x + radius, image_size), min(y + radius, image_size)

    print(f"looking in the box x {left}..{right}, y {top}..{bottom} of the 640 frame")

    for tag, frame in (("clean", clean), ("hovered", hovered)):
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        region = grey[top:bottom, left:right]
        print(f"  {tag:<8} bright>{config.verify.bright}: {int((region > config.verify.bright).sum()):>5}"
              f"   mean {region.mean():6.1f}")

    pair = cv.hconcat([clean[top:bottom, left:right], hovered[top:bottom, left:right]])
    cv.imwrite(str(out / "side.png"),
               cv.resize(pair, None, fx=3, fy=3, interpolation=cv.INTER_NEAREST))

    # The whole frame too, since the name may well be drawn outside the box being looked
    # at, which would be a different bug with the same symptom.
    cv.imwrite(str(out / "diff.png"),
               cv.convertScaleAbs(cv.absdiff(hovered, clean), alpha=4))

    controller.park(capture.rect)
    print(f"wrote {out}/side.png, diff.png, clean.png, hovered.png")


if __name__ == "__main__":
    main()
