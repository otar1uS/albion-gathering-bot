"""
Click one node and lay the seconds that follow out as a strip of pictures.

Every measurement of the gathering bar so far has been an inference from statistics, and
each one has been wrong in a different way, so this makes no claim at all: it clicks a
node and shows what happens, as a contact sheet of the frames with the time on each. It
answers whether the character walks, whether it reaches the node, and whether the game
draws a bar, by showing them rather than by scoring them.

    .venv\\Scripts\\python.exe training\\watch_click.py --resource tree
"""

import argparse
import sys
from pathlib import Path
from time import sleep, time

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.Albion import resources
from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--seconds", type=float, default=24)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--roam", type=int, default=15)
    parser.add_argument("--out", default="images/click_watch")
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)

    model.window_capture.focus()
    sleep(1.5)

    nodes, confidences = [], []

    for attempt in range(args.roam):
        model.window_capture.focus()
        detections, _, boxes = model.scan()
        nodes = [(x, y) for x, y, found in detections if found is profile]
        confidences = [round(float(b[4]), 2) for b in boxes
                       if model.profile_of(b[5]) is profile]

        if nodes:
            break

        print(f"  no {profile} in sight, roaming ({attempt + 1}/{args.roam})")
        interaction._Interaction__roam()
        sleep(interaction.ROAM_WAIT)

    if not nodes:
        sys.exit(f"No {profile} found after roaming")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    before = model._process_image(model.window_capture.screenshot())
    cv.imwrite(str(out / "00_before.png"), before)

    print(f"{len(nodes)} {profile} in sight at {confidences}, clicking {nodes[0]}")
    interaction.pointer.left_click(*nodes[0])
    interaction._Interaction__park()

    start = time()
    step = args.seconds / args.shots
    shots = []

    for i in range(args.shots):
        while time() - start < i * step:
            sleep(0.05)

        model.window_capture.focus()
        frame = model._process_image(model.window_capture.screenshot())
        stamp = time() - start

        cv.putText(frame, f"t+{stamp:.1f}s", (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 255), 2, cv.LINE_AA)
        shots.append(frame)
        cv.imwrite(str(out / f"{i + 1:02d}_t{stamp:04.1f}s.png"), frame)

    # A contact sheet, halved, so the whole run can be taken in at once.
    half = [cv.resize(s, None, fx=0.5, fy=0.5) for s in shots]
    rows = [cv.hconcat(half[i:i + 4]) for i in range(0, len(half), 4)]
    width = max(r.shape[1] for r in rows)
    rows = [cv.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv.BORDER_CONSTANT) for r in rows]

    cv.imwrite(str(out / "strip.png"), cv.vconcat(rows))
    print(f"Wrote {out}/strip.png and {len(shots)} frames")


if __name__ == "__main__":
    main()
