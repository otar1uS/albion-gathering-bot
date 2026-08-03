"""
Find where the game draws the gathering bars, by watching a node being worked on.

The bars sit above the head of the character and how far above depends on the camera
zoom, so their place in the frame is not something that can be assumed once and kept.
Rather than guess, this clicks a node and films the whole thing: the character walks
over, works on it, and stops. The frames from while it was working and the frames from
after it stopped are taken from the same camera position, so subtracting the two leaves
the bars and almost nothing else.

Writes the region it found and a picture of it, and leaves the frames in images/bar_hunt
to look at by hand when the answer is surprising.

    .venv\\Scripts\\python.exe training\\find_bar.py --resource tree
"""

import argparse
import sys
from pathlib import Path
from time import sleep, time

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.Albion import resources, screen
from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction

# A frame differing from the last one by more than this on average is one the camera
# moved between, so it cannot be subtracted from it cleanly.
STILL = 3.0

# Grey levels a pixel has to change by to count as part of the bars rather than as the
# grass moving or the light shifting.
CHANGED = 28


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--out", default="images/bar_hunt")
    parser.add_argument("--roam", type=int, default=15,
                        help="How many times to walk somewhere else looking for a node.")
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)

    model.window_capture.focus()
    sleep(1.5)

    # The character rarely happens to be standing next to a node when this is started,
    # so it is walked around until one shows up, the same way the bot does it.
    nodes = []

    for attempt in range(args.roam):
        detections, _, _ = model.scan()
        nodes = [(x, y) for x, y, found in detections if found is profile]

        if nodes:
            break

        print(f"  no {profile} in sight, roaming ({attempt + 1}/{args.roam})")
        interaction._Interaction__roam()
        sleep(interaction.ROAM_WAIT)

    if not nodes:
        sys.exit(f"No {profile} found after roaming, move the character to a forest")

    print(f"Clicking {profile} at {nodes[0]}, filming for {args.seconds}s")
    interaction.pointer.left_click(*nodes[0])
    interaction._Interaction__park()

    frames, stamps = [], []
    start = time()

    while time() - start < args.seconds:
        frames.append(cv.cvtColor(model._process_image(model.window_capture.screenshot()),
                                  cv.COLOR_BGR2GRAY))
        stamps.append(time() - start)
        sleep(0.4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    last = frames[-1]

    # Frames the camera did not move between, so the only thing left after subtracting
    # is what the interface drew on top.
    still = [(t, f) for t, f in zip(stamps, frames)
             if float(np.abs(f.astype(int) - last.astype(int)).mean()) < STILL]

    print(f"{len(frames)} frames, {len(still)} of them from the same camera position")

    if len(still) < 3:
        sys.exit("The camera never settled, the character may not have reached the node")

    # The bars flicker and deplete, so the strongest evidence is the pixel that changed
    # the most in any of those frames, not the average.
    peak = np.zeros_like(last, dtype=np.uint8)

    for _, f in still[:-2]:
        peak = np.maximum(peak, cv.absdiff(f, last))

    mask = (peak > CHANGED).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 9), np.uint8))

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        sys.exit("Nothing changed between working and idle, no bar was drawn")

    # The bars are drawn over the character, which the capture puts in the middle of the
    # frame, so the change nearest that is them rather than a cloud moving in a corner.
    centre = screen.IMG_SIZE / 2
    boxes = [cv.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= 8 and b[3] >= 3]
    boxes.sort(key=lambda b: (b[0] + b[2] / 2 - centre) ** 2 + (b[1] + b[3] / 2 - centre) ** 2)

    print("\nChanged regions, nearest the character first:")
    for x, y, w, h in boxes[:6]:
        print(f"  x {x}..{x + w}  y {y}..{y + h}   ({w}x{h})")

    x, y, w, h = boxes[0]
    print(f"\nBest guess for the bars: TOP_X, TOP_Y = {x}, {y}   "
          f"BOTTOM_X, BOTTOM_Y = {x + w}, {y + h}")
    print(f"bar.py currently has  TOP_X, TOP_Y = 304, 205   BOTTOM_X, BOTTOM_Y = 340, 217")

    cv.imwrite(str(out / "mask.png"), mask)
    cv.imwrite(str(out / "idle.png"), last)

    working = still[0][1]
    cv.imwrite(str(out / "working.png"), working)

    # A wide crop around the change, blown up, so the bars can be seen for what they are.
    pad = 40
    crop = working[max(y - pad, 0):min(y + h + pad, screen.IMG_SIZE),
                   max(x - pad, 0):min(x + w + pad, screen.IMG_SIZE)]
    cv.imwrite(str(out / "zoom.png"), cv.resize(crop, None, fx=4, fy=4,
                                                interpolation=cv.INTER_NEAREST))
    print(f"\nWrote {out}/zoom.png, working.png, idle.png and mask.png")


if __name__ == "__main__":
    main()
