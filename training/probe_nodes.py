"""
Ask the game which of the trees the model found can actually be gathered.

A forest is full of trees that are scenery, and the model was never taught to tell them
from the ones holding wood, because every dataset it learned from only ever labelled
the gatherable kind and the decorative ones simply were not marked. So it finds both and
the bot walks to both, and half the time it ends up standing next to something it cannot
touch, which is what "it starts chopping and then stops" looks like from the outside.

The game knows the difference and says so: hovering a real node draws its name over it in
near white text and outlines it, and hovering scenery does nothing at all. This puts the
cursor on every tree found and reports what came back, which says whether that signal is
strong enough to filter on before any of it is built into the bot.

    .venv\\Scripts\\python.exe training\\probe_nodes.py --rounds 4
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

# Grey level over which a pixel counts as part of the label the game draws.
BRIGHT = 200

# How long the cursor rests on a node before the frame is read.
HOVER = 0.8

# Pixels around the hovered point that the label is looked for in. The name is drawn
# above the node and can be wider than the node itself.
LOOK = 90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--out", default="images/node_probe")
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model.window_capture.focus()
    sleep(1.5)

    window = model.window_capture.window
    seen = 0

    def look():
        return cv.cvtColor(model._process_image(model.window_capture.screenshot()),
                           cv.COLOR_BGR2GRAY)

    for round_number in range(args.rounds):
        model.window_capture.focus()
        _, _, boxes = model.scan()
        mine = [b for b in boxes if model.profile_of(b[5]) is profile]

        if not mine:
            print(f"round {round_number + 1}: nothing in sight, roaming")
            interaction._Interaction__roam()
            sleep(interaction.ROAM_WAIT)
            continue

        interaction._Interaction__park()
        sleep(HOVER)
        clean = look()

        print(f"round {round_number + 1}: {len(mine)} {profile} in sight")

        for box in sorted(mine, key=lambda b: -float(b[4])):
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            ix, iy = (x1 + x2) / 2, (y1 + y2) / 2

            model.window_capture.focus()
            interaction.pointer.move(
                int(ix * window.width / model.IMG_SIZE + window.left),
                int(iy * window.height / model.IMG_SIZE + window.top))
            sleep(HOVER)

            hovered = look()

            top, bottom = max(int(iy) - LOOK, 0), min(int(iy) + LOOK, model.IMG_SIZE)
            left, right = max(int(ix) - LOOK, 0), min(int(ix) + LOOK, model.IMG_SIZE)

            before = int((clean[top:bottom, left:right] > BRIGHT).sum())
            after = int((hovered[top:bottom, left:right] > BRIGHT).sum())
            moved = float(np.abs(hovered[top:bottom, left:right].astype(int)
                                 - clean[top:bottom, left:right].astype(int)).mean())

            seen += 1
            print(f"   conf {float(box[4]):.2f}  box {x2 - x1:>3}x{y2 - y1:<3}  "
                  f"bright {before:>4} -> {after:<4} ({after - before:+5})   change {moved:5.2f}")

            cv.imwrite(str(out / f"{seen:02d}_conf{int(float(box[4]) * 100)}.png"), hovered)

            interaction._Interaction__park()
            sleep(0.4)

        cv.imwrite(str(out / f"round{round_number + 1}_clean.png"), clean)

        interaction._Interaction__roam()
        sleep(interaction.ROAM_WAIT)

    print(f"\n{seen} trees probed, frames in {out}")
    print("A gatherable node should show a clear jump in bright pixels, scenery none.")


if __name__ == "__main__":
    main()
