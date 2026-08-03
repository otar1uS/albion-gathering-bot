"""
Score two sets of weights against the same frames and print the difference.

The 42 frames in images/dataset are unlabelled, so this cannot compute mAP and does
not pretend to. What it measures is what actually matters for the bot: on a real
Forgotten Woods screen, does the model see a tree at all, and how sure is it. A model
that finds nothing there is useless however good its validation numbers look.

Counts alone can be gamed by a model that hallucinates, so every frame is also written
out annotated, side by side, into images/model_compare for a human to look at. Trust
the pictures over the table.

Frames are fed exactly as the bot feeds them: images/dataset holds 640x640 images that
were already put through Application.Albion.screen.process, so they go to the model
untouched, in the same colour space the bot uses.

    .venv\\Scripts\\python.exe training\\compare_models.py
"""

import argparse
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from platform import system

import cv2 as cv
import torch

ROOT = Path(__file__).resolve().parents[1]

# Every detection is collected at this threshold, and the report then splits at the
# bot's own 0.5. A tree found at 0.42 is invisible to the bot today but is a very
# different situation from a tree that was never found, and lowering the confidence is
# a one-line fix, so the two cases are worth telling apart.
SWEEP_CONFIDENCE = 0.20
BOT_CONFIDENCE = 0.50


def load(weights):
    """
    Load weights through the vendored yolov5, on the CPU the bot uses.

    Weights trained on Linux pickle their paths as PosixPath and Windows refuses to
    build one, so PosixPath is aimed at WindowsPath for the unpickling and put back
    afterwards. This is the same workaround AlbionDetection._load_model already uses,
    and it is needed here because the old weights came off Colab.

    :param weights: Path of the .pt file.
    :return: yolov5 autoshape model.
    """
    posix = pathlib.PosixPath

    if system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        model = torch.hub.load(
            str(ROOT / "yolov5"),
            "custom",
            path=str(weights),
            source="local",
            device="cpu",
            _verbose=False,
        )
    finally:
        pathlib.PosixPath = posix

    model.conf = SWEEP_CONFIDENCE
    return model


def names_of(model):
    """
    Class labels of a model as a dict, whatever form the checkpoint stored them in.

    :param model: Loaded model.
    :return: Dictionary of class id to label.
    """
    names = model.names
    return names if isinstance(names, dict) else dict(enumerate(names))


def detect(model, frames):
    """
    Run a model over every frame.

    :param model: Loaded model.
    :param frames: List of image paths.
    :return: Dictionary of frame name to list of (label, confidence, box) tuples.
    """
    names = names_of(model)
    out = {}

    for frame in frames:
        img = cv.imread(str(frame))
        rows = model(img).xyxy[0].tolist()
        out[frame.name] = [
            (names[int(cls)], conf, (int(x1), int(y1), int(x2), int(y2)))
            for x1, y1, x2, y2, conf, cls in rows
        ]

    return out


def summarise(results, floor):
    """
    Count detections per class above a confidence.

    :param results: Output of detect.
    :param floor: Lowest confidence to count.
    :return: Tuple of (per class count, per class frame count, per class mean confidence).
    """
    counts = defaultdict(int)
    frames = defaultdict(set)
    confs = defaultdict(list)

    for name, dets in results.items():
        for label, conf, _ in dets:
            if conf >= floor:
                counts[label] += 1
                frames[label].add(name)
                confs[label].append(conf)

    means = {k: sum(v) / len(v) for k, v in confs.items()}
    return counts, {k: len(v) for k, v in frames.items()}, means


def table(title, results, total_frames):
    """
    Print the per class breakdown of one model at both confidences.

    :param title: Heading to print.
    :param results: Output of detect.
    :param total_frames: Number of frames scored.
    """
    bot_counts, bot_frames, bot_means = summarise(results, BOT_CONFIDENCE)
    low_counts, low_frames, _ = summarise(results, SWEEP_CONFIDENCE)

    print(f"\n{title}")
    print(f"  {'class':<10}{'boxes>=.5':>11}{'frames>=.5':>12}"
          f"{'mean conf':>11}{'boxes>=.2':>11}")

    for label in sorted(set(low_counts) | set(bot_counts)):
        mean = f"{bot_means.get(label, 0):.2f}" if label in bot_means else "-"
        print(f"  {label:<10}{bot_counts.get(label, 0):>11}"
              f"{str(bot_frames.get(label, 0)) + '/' + str(total_frames):>12}"
              f"{mean:>11}{low_counts.get(label, 0):>11}")


def annotate(img, dets, floor, colour):
    """
    Draw the detections of one model onto a copy of a frame.

    :param img: Frame to draw on.
    :param dets: List of (label, confidence, box) tuples.
    :param floor: Lowest confidence to draw.
    :param colour: BGR colour of the boxes.
    :return: Annotated copy.
    """
    img = img.copy()

    for label, conf, (x1, y1, x2, y2) in dets:
        if conf < floor:
            continue
        cv.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv.LINE_AA)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default=ROOT / "best.pt")
    parser.add_argument("--new", default=ROOT / "best_merged.pt")
    parser.add_argument("--frames", default=ROOT / "images" / "dataset")
    parser.add_argument("--out", default=ROOT / "images" / "model_compare")
    parser.add_argument("--resource", default="tree",
                        help="Class the verdict at the end is about.")
    args = parser.parse_args()

    frames = sorted(Path(args.frames).glob("*.png"))
    if not frames:
        sys.exit(f"No frames in {args.frames}")

    print(f"Scoring {len(frames)} frames from {args.frames}")

    old, new = load(args.old), load(args.new)
    print(f"  old {Path(args.old).name}: {list(names_of(old).values())}")
    print(f"  new {Path(args.new).name}: {list(names_of(new).values())}")

    old_res, new_res = detect(old, frames), detect(new, frames)

    table(f"OLD  {Path(args.old).name}", old_res, len(frames))
    table(f"NEW  {Path(args.new).name}", new_res, len(frames))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        img = cv.imread(str(frame))
        pair = cv.hconcat([
            annotate(img, old_res[frame.name], BOT_CONFIDENCE, (0, 0, 255)),
            annotate(img, new_res[frame.name], BOT_CONFIDENCE, (0, 255, 0)),
        ])
        cv.putText(pair, "OLD", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(pair, "NEW", (img.shape[1] + 10, 25), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.imwrite(str(out / frame.name), pair)

    res = args.resource
    old_hit = summarise(old_res, BOT_CONFIDENCE)[1].get(res, 0)
    new_hit = summarise(new_res, BOT_CONFIDENCE)[1].get(res, 0)

    print(f"\n{res} found on {old_hit}/{len(frames)} frames by the old weights, "
          f"{new_hit}/{len(frames)} by the new ones.")
    print(f"Side by side images written to {out}, old in red, new in green.")
    print("Look at them before switching anything: these frames have no labels, so a "
          "higher count here is only better if the extra boxes are really trees.")


if __name__ == "__main__":
    main()
