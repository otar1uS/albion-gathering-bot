"""
Loads best.pt through the bot's own loading code and runs it on a picture.

It goes through AlbionDetection._load_model on purpose, rather than calling
torch.hub itself, so what is proven here is the path the bot really takes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Application import paths
from Application.Albion import resources
from Application.Albion.detection import AlbionDetection


class OnlyTheWeights:
    """
    Enough of an AlbionDetection for _load_model, which reads nothing else.

    The real class looks for the game window as it is built, and the point here is
    to check the weights without needing Albion to be running.
    """

    model_name = paths.MODEL


print(f"loading {paths.MODEL} ({paths.MODEL.stat().st_size / 1e6:.1f} MB)")

model = AlbionDetection._load_model(OnlyTheWeights())

names = model.names
names = names if isinstance(names, dict) else dict(enumerate(names))

print("\nclasses the model knows:")
for index, label in names.items():
    profile = resources.resolve(label)
    print(f"  {index}: {label:12} -> {'IGNORED' if str(profile) == 'unknown' else str(profile).upper()}")

trees = [label for label in names.values() if str(resources.resolve(label)) == "tree"]
print(f"\ntree classes the bot will click: {trees or 'NONE, the bot would never chop anything'}")

# Inference on a real picture, so a model that loads but predicts nothing is caught.
sample = paths.ROOT / "ressources" / "prediction.jpg"

if sample.exists():
    import cv2 as cv

    image = cv.resize(cv.imread(str(sample)), (640, 640))
    detections = model(image).xyxy[0]

    print(f"\ninference on {sample.name}: {len(detections)} detections")
    for detection in detections[:8]:
        print(f"  {names[int(detection[5])]:12} {float(detection[4]):.2f}")

print("\nMODEL OK")
