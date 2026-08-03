"""
Grabs the game window the way the bot does and runs the model at a low threshold.

Everything found is drawn and written to images/live_test.png, so what the bot is
looking at can be compared with what it recognises.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv

from Application import game, paths
from Application.Albion import bar, resources, screen
from Application.Albion.detection import AlbionDetection
from Application.Capture.Factory import CaptureFactory
from training.focus_game import focus

THRESHOLD = 0.25


class OnlyTheWeights:
    model_name = paths.MODEL


capture = CaptureFactory(game.DEFAULT_WINDOW_NAME).capture
print(f"window: {capture.window}")

# In front and grabbed in the same process, otherwise whatever ran the script is back
# on top by the time the screenshot is taken, and the model is shown a terminal.
focus()

image = screen.grab(capture)

model = AlbionDetection._load_model(OnlyTheWeights())
names = model.names
names = names if isinstance(names, dict) else dict(enumerate(names))

detections = [d for d in model(image).xyxy[0] if float(d[4]) >= THRESHOLD]

print(f"\n{len(detections)} detections over {THRESHOLD}")
for d in sorted(detections, key=lambda d: -float(d[4])):
    label = names[int(d[5])]
    x1, y1, x2, y2 = (int(v) for v in d[:4])
    profile = resources.resolve(label)
    print(f"  {label:12} {float(d[4]):.2f}  {x2 - x1}x{y2 - y1}px  -> {profile}")

    colour = (0, 255, 0) if str(profile) == "tree" else (255, 160, 0)
    cv.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    cv.putText(image, f"{label} {float(d[4]):.2f}", (x1, max(y1 - 4, 10)),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

# The area the gathering bars are looked for in, to check it still lines up.
left, top, right, bottom = bar.searched_region()
cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)

cv.imwrite(str(paths.IMAGES / "live_test.png"), image)
print(f"\nwrote {paths.IMAGES / 'live_test.png'}")
