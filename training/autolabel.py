"""
Collect frames from the farming zone and pre-label them, so correcting is quicker than
drawing from scratch.

The tree class is the weak one and it is weak for a countable reason: 560 boxes across
the whole training set against 11650 for ore, twenty one times thinner. Augmented
inference buys most of that back but nothing fixes it properly except trees from the zone
the bot actually works in.

Labelling several hundred frames by hand is a day's work, so the model labels them first
and the job becomes correcting boxes rather than drawing them. Two things make that safe:
the threshold is dropped well below what the bot would act on, because a box in roughly
the right place is quicker to nudge than to draw and a wrong one is quick to delete; and
nothing here is treated as ground truth, it is a starting point for a person.

    collect frames while you play:
        .venv\\Scripts\\python.exe training\\autolabel.py collect --minutes 20

    label whatever is in the folder:
        .venv\\Scripts\\python.exe training\\autolabel.py label

Then open the folder in a labelling tool, fix the boxes, and train on it.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from time import sleep, time

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from albion import logs
from albion.config import Config
from albion.vision import capture as capture_module
from albion.vision.detector import Detector, process

log = logs.get("autolabel")

# Frames are only kept when they differ from the last kept one by this much, so standing
# still for a minute does not produce a hundred copies of the same picture. A dataset of
# near duplicates teaches the model that one tree very well and nothing else.
DIFFERENT_ENOUGH = 6.0


def collect(config, out, minutes, interval):
    """
    Save frames from the game while somebody plays.

    :param config: Config.
    :param out: Folder to write to.
    :param minutes: How long to record for.
    :param interval: Seconds between frames.
    :return: How many frames were kept.
    """
    import numpy as np

    capture = capture_module.build(config)
    out.mkdir(parents=True, exist_ok=True)

    log.info("recording for %.0f minutes into %s. Play normally, walk around the zone "
             "you farm and look at plenty of different trees.", minutes, out)

    end = time() + minutes * 60
    previous = None
    kept = 0

    while time() < end:
        frame = process(capture.grab(), config.vision.image_size)

        if previous is not None:
            change = float(np.abs(frame.astype(np.int16)
                                  - previous.astype(np.int16)).mean())

            if change < DIFFERENT_ENOUGH:
                sleep(interval)
                continue

        cv.imwrite(str(out / f"frame_{int(time() * 1000)}.png"), frame)
        previous = frame
        kept += 1

        if kept % 25 == 0:
            log.info("%d frames so far", kept)

        sleep(interval)

    log.info("kept %d frames in %s", kept, out)

    return kept


def label(config, folder, threshold):
    """
    Write a YOLO label file beside every frame.

    :param config: Config.
    :param folder: Folder holding the frames.
    :param threshold: Lowest confidence to write a box for.
    :return: How many boxes were written.
    """
    frames = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

    if not frames:
        sys.exit(f"No frames in {folder}, run the collect step first")

    capture = capture_module.build(config)
    detector = Detector(config, capture)
    names = [detector.classes[key] for key in sorted(detector.classes)]

    size = config.vision.image_size
    counts = Counter()
    empty = 0
    boxes = 0

    for frame_path in frames:
        image = cv.imread(str(frame_path))
        raw = detector.model(image, augment=config.vision.augment).xyxy[0].tolist()

        lines = []

        for x1, y1, x2, y2, confidence, class_id in raw:
            if confidence < threshold:
                continue

            # YOLO wants the centre and the size, each as a fraction of the image.
            cx, cy = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
            width, height = (x2 - x1) / size, (y2 - y1) / size

            lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            counts[names[int(class_id)]] += 1
            boxes += 1

        if not lines:
            empty += 1

        frame_path.with_suffix(".txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")

    (folder / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    (folder / "data.yaml").write_text(
        "train: images\nval: images\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n", encoding="utf-8")

    log.info("labelled %d frames at a %.2f threshold, %d boxes", len(frames), threshold,
             boxes)

    for name, count in counts.most_common():
        log.info("  %-8s %d", name, count)

    if empty:
        log.info("  %d frames got no boxes at all, which is where the model is blind and "
                 "where hand drawn boxes are worth the most", empty)

    log.info("Open %s in a labelling tool, fix the boxes, and keep the classes in the "
             "order in classes.txt so the weights stay compatible.", folder)

    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("collect", "label", "both"))
    parser.add_argument("--folder", default="datasets/zone_capture")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Deliberately low. A box roughly in the right place is "
                             "quicker to nudge than to draw, and a wrong one is quick to "
                             "delete.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    logs.setup(config.log_level)

    folder = Path(args.folder)

    if args.step in ("collect", "both"):
        collect(config, folder, args.minutes, args.interval)

    if args.step in ("label", "both"):
        label(config, folder, args.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())
