"""
Find which point inside a detection box the game treats as the resource.

A box around a tree covers the crown as much as the trunk, and the middle of it is
often a gap between two branches with the ground showing through. Clicking there walks
the character over and leaves it standing next to a tree it never touched, which looks
exactly like a node that refuses to be gathered.

The game outlines a resource and names it when the cursor is over it, so the answer can
be had without clicking anything: the cursor is put on a few points down the box in turn
and the frame is compared against the same frame with the cursor parked away. The point
that changes the picture the most is the one the game accepts.

    .venv\\Scripts\\python.exe training\\probe_aim.py --resource tree
"""

import argparse
import sys
from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.Albion import resources
from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction

# Fractions down the box to try, 0 being its top edge and 1 its bottom.
DEPTHS = (0.30, 0.50, 0.65, 0.80, 0.92)

# How long to leave the cursor somewhere before looking, the outline fades in.
HOVER = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--roam", type=int, default=15)
    parser.add_argument("--out", default="images/aim_probe")
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)

    model.window_capture.focus()
    sleep(1.5)

    box = None

    for attempt in range(args.roam):
        model.window_capture.focus()
        _, _, boxes = model.scan()
        mine = [b for b in boxes if model.profile_of(b[5]) is profile]

        if mine:
            box = max(mine, key=lambda b: float(b[4]))
            break

        print(f"  no {profile} in sight, roaming ({attempt + 1}/{args.roam})")
        interaction._Interaction__roam()
        sleep(interaction.ROAM_WAIT)

    if box is None:
        sys.exit(f"No {profile} found after roaming")

    x1, y1, x2, y2 = (int(v) for v in box[:4])
    window = model.window_capture.window

    print(f"{profile} at {float(box[4]):.2f}, box {x1},{y1} to {x2},{y2} "
          f"({x2 - x1}x{y2 - y1} of the 640 frame)")

    def to_screen(ix, iy):
        return (int(ix * window.width / model.IMG_SIZE + window.left),
                int(iy * window.height / model.IMG_SIZE + window.top))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def look():
        return cv.cvtColor(model._process_image(model.window_capture.screenshot()),
                           cv.COLOR_BGR2GRAY)

    # The cursor parked out of the way, which is what the box looks like untouched.
    interaction._Interaction__park()
    sleep(HOVER)
    clean = look()

    # Only the box is compared, and a little around it, so a cloud drifting somewhere
    # else does not outscore the outline.
    pad = 12
    top, bottom = max(y1 - pad, 0), min(y2 + pad, model.IMG_SIZE)
    left, right = max(x1 - pad, 0), min(x2 + pad, model.IMG_SIZE)

    results = []

    for depth in DEPTHS:
        ix = (x1 + x2) / 2
        iy = y1 + (y2 - y1) * depth

        model.window_capture.focus()
        interaction.pointer.move(*to_screen(ix, iy))
        sleep(HOVER)

        hovered = look()
        changed = float(np.abs(hovered[top:bottom, left:right].astype(int)
                               - clean[top:bottom, left:right].astype(int)).mean())
        results.append((depth, changed))

        print(f"  depth {depth:.2f}  at {int(ix)},{int(iy)} in frame  ->  change {changed:6.2f}")

        cv.imwrite(str(out / f"hover_{int(depth * 100):02d}.png"), hovered)

        interaction._Interaction__park()
        sleep(0.3)

    cv.imwrite(str(out / "clean.png"), clean)

    best = max(results, key=lambda r: r[1])
    quiet = min(r[1] for r in results)

    print(f"\nStrongest response at depth {best[0]:.2f} ({best[1]:.2f} against {quiet:.2f} "
          f"at the quietest point)")

    if best[1] - quiet < 0.5:
        print("Nothing responded anywhere in the box, so the game does not see a resource "
              "there at all: that detection is not a gatherable node.")
    else:
        print(f"Aim {best[0]:.0%} down the box rather than at its middle.")


if __name__ == "__main__":
    main()
