"""
Print what the model sees in the running game, without touching the mouse.

This is the step between "the weights score well on saved frames" and "let the bot
drive": the saved frames in images/dataset were captured once, under one camera angle
and one time of day, and the model reading them correctly does not prove it reads the
window live. Nothing here clicks, moves or presses anything, so it is safe to run while
the character is standing still.

    .venv\\Scripts\\python.exe testing\\live_scan.py --seconds 15
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from time import sleep, time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Application.Albion import resources
from Application.Albion.detection import AlbionDetection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    # No preview window: it needs a waitKey pump to redraw and would only hang here.
    # targets covers every resource so the scan reports what the model sees rather than
    # what one profile asked for.
    model = AlbionDetection(debug=False, preview=False)

    # The capture reads the screen where the game window is, so anything drawn over it
    # is what the model gets handed: run this from a terminal covering the game and it
    # scans the terminal. Interaction.loop raises the window for the same reason, and
    # the pause gives the compositor time to actually paint it before the first frame.
    raised = model.window_capture.focus()
    sleep(1.5)

    print(f"Window: {model.window_capture.window}")
    print(f"Raised to the front: {raised}")
    print("Per class confidence:")
    for class_id, info in model.classes.items():
        print(f"  {info['label']:<9} -> {model.profile_of(class_id).name:<8} "
              f"@ {model._confidence_of(class_id)}")
    print(f"\nScanning for {args.seconds}s, nothing is clicked.\n")

    totals = defaultdict(int)
    seen = defaultdict(int)
    frames = 0
    end = time() + args.seconds

    while time() < end:
        detections, _, boxes = model.scan()
        frames += 1

        found = defaultdict(list)
        for box in boxes:
            found[model.classes[int(box[5])]["label"]].append(float(box[4]))

        for label, confs in found.items():
            totals[label] += len(confs)
            seen[label] += 1

        summary = ", ".join(
            f"{label} x{len(confs)} ({max(confs):.2f})" for label, confs in sorted(found.items())
        )
        closest = detections[0] if detections else None
        print(f"  frame {frames:>3}: {summary or 'nothing'}"
              + (f"  | closest {closest[2].name} at {closest[0]},{closest[1]}" if closest else ""))

        sleep(args.interval)

    print(f"\n{frames} frames scanned.")
    for label in sorted(totals):
        print(f"  {label:<9} {totals[label]:>3} boxes over {seen[label]:>3}/{frames} frames "
              f"-> {resources.resolve(label).name}")

    if not totals:
        print("  nothing detected at all — check the character is outside and facing resources")


if __name__ == "__main__":
    main()
