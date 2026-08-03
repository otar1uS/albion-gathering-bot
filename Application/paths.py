from pathlib import Path

# Every path used at runtime is absolute, so the bot can be started from any
# directory and not only from the repository root.
ROOT = Path(__file__).resolve().parent.parent

YOLOV5 = ROOT / "yolov5"

# best_merged.pt was trained on the six merged datasets and replaced the original
# best.pt, which is kept next to it to fall back on. The old weights label the tier 4
# resources ("rought log", "travertine") rather than the resource itself, and on the 42
# Forgotten Vigils frames in images/dataset they drew logs over bare ground while
# missing the trees standing in the same picture, which is the reason gathering failed
# in the forests. Compare the two with training/compare_models.py before swapping back.
MODEL = ROOT / "best_merged.pt"
IMAGES = ROOT / "images"
RESOURCE_BAR = IMAGES / "cropped_bar_resource.png"
RESOURCE_FRAME = IMAGES / "640x640_resource.png"
