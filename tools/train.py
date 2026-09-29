"""
Train the model on the dataset built by tools/dataset.py merge, and put the weights
where the bot looks for them.

    python tools/train.py                    # picks the Intel Arc, Nvidia GPU or CPU
    python tools/train.py --openvino         # also exports for fast Intel inference
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from albion_bot import paths
from albion_bot.vision.detector import torch_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo26s.pt",
                        help="starting weights, yolo26n.pt is faster, yolo26s.pt more accurate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    parser.add_argument("--workers", type=int, default=4, help="0 if Windows complains about dataloader workers")
    parser.add_argument("--device", default=None, help="xpu, 0 for Nvidia, cpu, found on its own by default")
    parser.add_argument("--openvino", action="store_true", help="export for OpenVINO once trained")
    arguments = parser.parse_args()

    data = paths.DATASET_MERGED / "data.yaml"

    if not data.exists():
        sys.exit(f"{data} is missing, run: python tools/dataset.py download, then merge")

    from ultralytics import YOLO

    chosen = arguments.device or torch_device()
    print(f"Training {arguments.model} on {chosen}")

    model = YOLO(arguments.model)
    model.train(data=str(data), epochs=arguments.epochs, imgsz=arguments.imgsz, batch=arguments.batch,
                device=chosen, workers=arguments.workers, project=str(paths.RUNS), name="albion", exist_ok=True,
                # Left and right look alike in an isometric game, up and down do not.
                fliplr=0.5, flipud=0.0, patience=25)

    best = Path(model.trainer.best)
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, paths.MODEL)
    print(f"Weights copied to {paths.MODEL}")

    if arguments.openvino:
        # Cleared first, the export is written next to the weights, right there.
        if paths.MODEL_OPENVINO.exists():
            shutil.rmtree(paths.MODEL_OPENVINO)

        exported = Path(YOLO(str(paths.MODEL)).export(format="openvino", imgsz=arguments.imgsz, half=True))

        if exported.resolve() != paths.MODEL_OPENVINO.resolve():
            shutil.move(str(exported), paths.MODEL_OPENVINO)

        print(f"OpenVINO model in {paths.MODEL_OPENVINO}, the bot uses it on its own")
    elif paths.MODEL_OPENVINO.exists():
        # The bot prefers the export, which now holds the previous weights.
        shutil.rmtree(paths.MODEL_OPENVINO)
        print(f"Removed the outdated {paths.MODEL_OPENVINO}, add --openvino to export again")


if __name__ == "__main__":
    main()
