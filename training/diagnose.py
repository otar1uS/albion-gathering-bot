"""Runs the model on the frame the bot captured, both ways round, at low threshold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv

from Application import paths
from Application.Albion.detection import AlbionDetection


class OnlyTheWeights:
    model_name = paths.MODEL


model = AlbionDetection._load_model(OnlyTheWeights())
names = model.names
names = names if isinstance(names, dict) else dict(enumerate(names))

frame = cv.imread(str(paths.RESOURCE_FRAME))

# The saved frame already went through screen.process, so it is what the model is fed.
# The swapped version is tried too, in case the channel order is the thing that is off.
variants = {
    "as the bot feeds it": frame,
    "channels swapped back": cv.cvtColor(frame, cv.COLOR_BGR2RGB),
}

for label, image in variants.items():
    detections = model(image).xyxy[0]
    print(f"\n{label}: {len(detections)} detections")
    for d in sorted(detections, key=lambda d: -float(d[4]))[:10]:
        name = names[int(d[5])]
        w, h = int(d[2]) - int(d[0]), int(d[3]) - int(d[1])
        print(f"   {name:12} {float(d[4]):.2f}  {w}x{h}px")
