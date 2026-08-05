"""
One question: does the minimap say where the character is, and does it keep up?

Reading a position off a still frame proves nothing. What matters is whether it tracks:
whether walking moves the reading, whether standing still leaves it alone, and whether it
survives the arrow crossing water, grass and the edge of the map. So this walks the
character in a square and prints what the minimap said after each leg.

    .venv\\Scripts\\python.exe -m training.probe_minimap

Nothing is gathered and nothing is clicked except empty ground to walk to.
"""

import sys
from pathlib import Path
from time import sleep, time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from albion import logs
from albion.config import Config
from albion.control.input import build as build_controller
from albion.nav.minimap import Minimap
from albion.vision.capture import build as build_capture

LEGS = 6
WALK_SECONDS = 4.0


def main():
    logs.setup("INFO")
    config = Config.load()

    capture = build_capture(config)
    capture.refresh()
    capture.focus()

    minimap = Minimap(config, capture)
    controller = build_controller(config)

    picture = capture.grab()
    print(f"window {capture.rect}, picture {picture.shape[1]}x{picture.shape[0]}")

    left, top, right, bottom = minimap.box(picture)
    print(f"minimap box in the picture: ({left}, {top}) to ({right}, {bottom})")

    start = minimap.where(picture)

    if start is None:
        print("\nThe arrow was not found at all. Save the frame and look at it: the box "
              "above is probably not over the minimap.")
        return 1

    print(f"\nstanding still, the character reads {start[0]:.4f}, {start[1]:.4f}")

    # Still first, so the noise floor of the reading is known before anything moves. A
    # tracker that wanders while the character does not is worse than none.
    sleep(2.0)
    again = minimap.where()
    print(f"two seconds later      {again[0]:.4f}, {again[1]:.4f}  "
          f"(drifted {minimap.travelled(start, again):.4f})")

    rect = capture.rect
    corners = [(0.35, 0.30), (0.62, 0.30), (0.62, 0.48), (0.35, 0.48)]
    previous = again

    print("\nwalking a square, one leg at a time")

    for leg in range(LEGS):
        fx, fy = corners[leg % len(corners)]
        controller.click(*rect.fraction(fx, fy))

        deadline = time() + WALK_SECONDS

        while time() < deadline:
            sleep(0.3)

        now = minimap.where()

        if now is None:
            print(f"  leg {leg + 1}: the arrow was lost")
            continue

        print(f"  leg {leg + 1}: {now[0]:.4f}, {now[1]:.4f}  "
              f"moved {minimap.travelled(previous, now):.4f}")
        previous = now

    print(f"\nover the whole square the character moved "
          f"{minimap.travelled(start, previous):.4f} of the map from where it started.")
    print("A number near zero after six legs means the reading is not tracking. A number "
          "that grew leg by leg means it is.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
