"""
Collects training pictures of the game, to teach the model resources it does not know.

Run it, then play normally. A picture of the game is taken every few seconds and
dropped in images/dataset, ready to be uploaded to Roboflow and labelled. Only frames
that differ enough from the last kept one are saved, so standing still does not fill
the folder with the same picture a hundred times.

    python training/collect_screenshots.py [seconds] [interval]
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time import sleep, time

import cv2 as cv
import numpy as np

from Application import game, paths
from Application.Albion import screen
from Application.Capture.Factory import CaptureFactory

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

# Mean pixel difference under which a frame is considered the same view as the last one
# kept. Walking changes a frame far more than this, standing still far less.
DIFFERENCE = 12.0

capture = CaptureFactory(game.DEFAULT_WINDOW_NAME).capture
capture.focus()

folder = paths.IMAGES / "dataset"
folder.mkdir(parents=True, exist_ok=True)

print(f"saving to {folder}, every {INTERVAL}s for {SECONDS}s, play normally")

previous = None
kept = 0
start = time()

while time() - start < SECONDS:
    frame = screen.grab(capture)

    if previous is None or float(np.mean(cv.absdiff(frame, previous))) >= DIFFERENCE:
        name = folder / f"albion_{datetime.now().strftime('%H%M%S_%f')[:-3]}.png"
        cv.imwrite(str(name), frame)
        previous = frame
        kept += 1
        print(f"  {kept:4d}  {name.name}")

    sleep(INTERVAL)

print(f"\n{kept} pictures in {folder}")
print("Upload that folder to Roboflow, label the resources, and retrain with the notebook.")
