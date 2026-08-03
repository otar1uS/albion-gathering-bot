"""
Clicks the closest node and photographs what happens, so a gathering that is really
happening can be told apart from a bot clicking at nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time import sleep, time

import cv2 as cv

from Application import paths
from Application.Albion import screen
from Application.Albion.detection import AlbionDetection
from Application.Interaction import pointer

model = AlbionDetection(debug=False, preview=False, confidence=0.60, targets=["tree", "stone", "ore"])
model.window_capture.focus()

mouse = pointer.create()
shots = paths.IMAGES / "gather"
shots.mkdir(exist_ok=True)

detections, _, _ = model.scan()
targets = [d for d in detections if str(d[2]) != "unknown"]

if not targets:
    print("nothing to click")
    sys.exit(1)

x, y, profile = targets[0]
print(f"clicking {profile} at ({x}, {y})")

cv.imwrite(str(shots / "00_before.png"), screen.grab(model.window_capture))

mouse.left_click(x, y)
window = model.window_capture.window
mouse.move(int(window.left + window.width * 0.03), int(window.top + window.height * 0.10))

start = time()

for step in range(1, 13):
    sleep(2.5)
    frame = screen.grab(model.window_capture)
    cv.imwrite(str(shots / f"{step:02d}_t{int(time() - start):03d}s.png"), frame)

    found, _, _ = model.scan()
    same = [d for d in found if d[2] is profile]
    print(f"  t+{int(time() - start):3d}s  {len(found)} detections, {len(same)} {profile}")

print(f"frames in {shots}")
