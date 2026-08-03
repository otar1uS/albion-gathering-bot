"""
Click a node and watch the gathering bar, to tell whether the saved picture of it still
matches what the game draws.

The picture is cropped from a fixed region of the frame and only holds for one interface
scale and one camera zoom, so it goes stale silently: the bot keeps running and every
node ends on "Mooving timed out", because it is waiting for a bar it can no longer
recognise. This walks the character to a real node and prints what the match is worth
while it works on it, which is the only moment the answer means anything.

    .venv\\Scripts\\python.exe training\\check_bar.py --resource tree
"""

import argparse
import sys
from pathlib import Path
from time import sleep, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.Albion import bar, resources
from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--seconds", type=float, default=25)
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)
    template = interaction.img_border_resource

    if template is None:
        sys.exit("No saved picture of the bar, calibrate it first")

    print(f"Template {template.shape[1]}x{template.shape[0]}, "
          f"usable {bar.usable(template)}, threshold {bar.CONFIDENCE}")

    model.window_capture.focus()
    sleep(1.5)

    idle = bar.confidence(model.window_capture, template)
    print(f"Idle match: {idle:.3f}  (has to stay well under {bar.CONFIDENCE})")

    detections, _, _ = model.scan()
    nodes = [(x, y) for x, y, found in detections if found is profile]

    if not nodes:
        sys.exit(f"No {profile} in sight to walk to, move the character near one")

    print(f"Clicking the closest of {len(nodes)} {profile} at {nodes[0]}")
    interaction.pointer.left_click(*nodes[0])

    # The cursor sits on the node otherwise and the tooltip the game draws under it
    # covers the very bars this is trying to read.
    interaction._Interaction__park()

    start = time()
    best = idle
    reached = None

    while time() - start < args.seconds:
        found = bar.confidence(model.window_capture, template)
        best = max(best, found)

        if found >= bar.CONFIDENCE and reached is None:
            reached = time() - start

        print(f"  t+{time() - start:5.1f}s  {found:.3f}"
              f"{'  GATHERING' if found >= bar.CONFIDENCE else ''}")

        sleep(0.5)

    print(f"\nBest match while working: {best:.3f}, threshold {bar.CONFIDENCE}")

    if reached is None:
        print("The bar was never recognised. Either the character never reached the node,"
              " or the saved picture no longer matches what the game draws and has to be"
              " calibrated again at this zoom.")
    else:
        print(f"Recognised {reached:.1f}s after the click, the template is good.")


if __name__ == "__main__":
    main()
