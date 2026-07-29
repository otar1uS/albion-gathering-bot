from pathlib import Path

# Every path used at runtime is absolute, so the bot can be started from any
# directory and not only from the repository root.
ROOT = Path(__file__).resolve().parent.parent

YOLOV5 = ROOT / "yolov5"
MODEL = ROOT / "best.pt"
IMAGES = ROOT / "images"
RESOURCE_BAR = IMAGES / "cropped_bar_resource.png"
RESOURCE_FRAME = IMAGES / "640x640_resource.png"
