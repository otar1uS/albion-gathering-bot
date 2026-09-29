from pathlib import Path

# Every path used at runtime is absolute, so the bot can be started from any directory.
ROOT = Path(__file__).resolve().parent.parent

# What each machine builds or records for itself, none of it is committed.
DATA = ROOT / "data"
SETTINGS = DATA / "settings.json"
ROUTES = DATA / "routes"
GATHER_BAR = DATA / "gather_bar.png"
GATHER_BAR_INFO = DATA / "gather_bar.json"

MODELS = ROOT / "models"
MODEL = MODELS / "best.pt"
# The same weights exported by tools/train.py --openvino, preferred when present.
MODEL_OPENVINO = MODELS / "best_openvino_model"

DATASETS = ROOT / "datasets"
DATASET_RAW = DATASETS / "raw"
DATASET_CAPTURED = DATASETS / "captured"
DATASET_MERGED = DATASETS / "albion"
RUNS = ROOT / "runs"
