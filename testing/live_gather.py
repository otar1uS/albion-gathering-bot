"""
Run the real gathering loop for a fixed time, then stop by itself.

Interaction.loop only ends when a human stops it, which is right for a bot left to
farm but wrong for checking whether a change works: a test has to end on its own and
say what happened. This starts the same loop against the same model and trips
stop_requested once the clock runs out, so the character is left alone afterwards.

The mouse is driven for real while this runs. Throwing the pointer into the top left
corner of the screen stops it early, as does ctrl+c.

    .venv\\Scripts\\python.exe testing\\live_gather.py --seconds 90
"""

import argparse
import sys
from pathlib import Path
from threading import Timer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--targets", default="tree",
                        help="Comma separated resource names, empty for all of them.")
    parser.add_argument("--no-bar", action="store_true",
                        help="Ignore the saved gathering bar and follow nodes with the "
                             "model instead, which is what the bot does when the picture "
                             "was never calibrated. Use this to tell a stale bar template "
                             "apart from a character that is not walking at all.")
    args = parser.parse_args()

    targets = [t for t in args.targets.split(",") if t] or None

    # preview stays off deliberately. The debug window is drawn over the game, and the
    # capture reads the screen, so the bot would end up being shown its own preview.
    model = AlbionDetection(debug=True, preview=False, targets=targets)
    interaction = Interaction(model)

    if args.no_bar:
        interaction.img_border_resource = None

    print(f"Targets: {[str(t) for t in model.targets]}")
    print(f"Gathering bar: "
          f"{'ignored, following nodes with the model' if interaction.img_border_resource is None else 'used'}")
    print(f"Confidence per class: "
          f"{ {v['label']: model._confidence_of(k) for k, v in model.classes.items()} }")
    print(f"Running the real loop for {args.seconds}s, the mouse is driven.\n")

    stop = Timer(args.seconds, lambda: setattr(interaction, "stop_requested", True))
    stop.daemon = True
    stop.start()

    try:
        interaction.loop()
    finally:
        stop.cancel()

    print(f"\nStopped. Empty scans in a row at the end: {interaction.empty_scans}")
    print(f"Nodes gathered and put on cooldown: {len(interaction.depleted)}")


if __name__ == "__main__":
    main()
