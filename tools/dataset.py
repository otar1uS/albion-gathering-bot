"""
Build the training set of the model, without taking a single screenshot by hand.

    download   public Albion datasets already labeled by others, from Roboflow Universe
    capture    screenshots of the game taken on their own while you play
    autolabel  labels for those screenshots, drawn by the current model
    merge      everything above in one dataset, every class renamed to tree, stone, ore,
               fiber, hide or monster, ready for tools/train.py

Downloading needs a free Roboflow account, its API key is read from the
ROBOFLOW_API_KEY environment variable and never written anywhere.
"""

import argparse
import hashlib
import io
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from time import sleep, time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import yaml

from albion_bot import paths
from albion_bot.vision import resources

API_URL = "https://api.roboflow.com"
EXPORT_FORMAT = "yolov11"

# Public Albion Online datasets on Roboflow Universe, workspace/project. Their classes
# are all named differently, "fiber3", "rough log", "iron ore", merge sorts them out.
DATASETS = (
    "albion-online/ao-resource-detection",
    "albiononline-6no0l/albion-bpgjq",
    "albiononline-c8fxi/albiongathering",
    "albion-database/albion-database",
    "albion-zmhvd/ore1",
    "claros-solutions/albion-ores",
    "test-n2brz/albion-online-gathering",
    # Monsters, so the bot can leave the nodes they guard alone.
    "albiononlinez/albion-enemys",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Share of the captured screenshots kept aside to measure the model.
VALIDATION_SHARE = 0.1


def api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")

    if not key:
        sys.exit("Set ROBOFLOW_API_KEY first, the free key of a Roboflow account, from "
                 "https://app.roboflow.com/settings/api")

    return key


def get(url, key, **named):
    """
    GET on the Roboflow API. The key goes in a header, out of every URL and error.
    """
    response = requests.get(url, params=named, headers={"Authorization": f"Bearer {key}"}, timeout=60)

    if response.status_code not in (200, 202):
        raise Exception(f"{url} answered {response.status_code}: {response.text[:300]}")

    return response


def latest_version(dataset, key):
    versions = get(f"{API_URL}/{dataset}", key).json().get("versions", [])

    if not versions:
        raise Exception(f"{dataset} has no version to download")

    return max(int(version["id"].rsplit("/", 1)[1]) for version in versions)


def download(datasets):
    key = api_key()
    paths.DATASET_RAW.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        try:
            version = latest_version(dataset, key)
            target = paths.DATASET_RAW / f"{dataset.replace('/', '__')}__v{version}"

            if target.exists():
                print(f"{dataset} v{version} already downloaded")
                continue

            print(f"{dataset} v{version}: asking for the export...")

            # Roboflow builds the export on the first request, 202 while it works on it.
            while True:
                response = get(f"{API_URL}/{dataset}/{version}/{EXPORT_FORMAT}", key, nocache="true")

                if response.status_code == 200 and "export" in response.json():
                    break

                sleep(2)

            link = response.json()["export"]["link"]
            archive = requests.get(link, timeout=600)
            archive.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(archive.content)) as content:
                content.extractall(target)

            print(f"{dataset} v{version}: {sum(1 for _ in target.rglob('*.txt'))} label files in {target}")

        except Exception as e:
            print(f"{dataset}: skipped, {e}")


def class_names(folder):
    """
    :return: Class names of a YOLO dataset, by id.
    """
    info = yaml.safe_load((folder / "data.yaml").read_text())
    names = info.get("names", [])

    return dict(names) if isinstance(names, dict) else dict(enumerate(names))


def box(values):
    """
    :param values: What follows the class id on a label line, a box or a polygon.
    :return: (x, y, w, h) normalized, None when the line cannot be read.
    """
    try:
        numbers = [float(value) for value in values]
    except ValueError:
        return None

    if len(numbers) == 4:
        return tuple(numbers)

    if len(numbers) >= 6 and len(numbers) % 2 == 0:
        xs, ys = numbers[0::2], numbers[1::2]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)

    return None


def convert(label_file, mapping):
    """
    Rewrite a label file with the classes of the merged dataset.

    :param mapping: Class id of the source to class id of the merged dataset, None to drop.
    :return: Lines of the new label file.
    """
    lines = []

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        target = mapping.get(int(float(parts[0])))
        coordinates = box(parts[1:])

        if target is None or coordinates is None:
            continue

        lines.append(f"{target} " + " ".join(f"{value:.6f}" for value in coordinates))

    return lines


def sources():
    """
    Every labeled folder to merge: (images folder, labels folder, split, mapping).
    """
    canonical = {name: index for index, name in enumerate(resources.CLASSES)}

    for folder in sorted(paths.DATASET_RAW.glob("*")) if paths.DATASET_RAW.exists() else []:
        if not (folder / "data.yaml").exists():
            continue

        names = class_names(folder)
        mapping = {}

        for class_id, name in names.items():
            profile = resources.resolve(str(name))
            mapping[int(class_id)] = canonical.get(str(profile))
            print(f"  {folder.name}: {name!r} -> {profile if mapping[int(class_id)] is not None else 'dropped'}")

        for split in ("train", "valid", "test"):
            if (folder / split / "images").exists():
                yield folder / split / "images", folder / split / "labels", split, mapping

    captured = paths.DATASET_CAPTURED
    identity = {index: index for index in range(len(resources.CLASSES))}

    if (captured / "labels").exists():
        yield captured / "images", captured / "labels", None, identity


def merge():
    merged = paths.DATASET_MERGED

    if merged.exists():
        shutil.rmtree(merged)

    seen = set()
    counts = {"train": 0, "val": 0}
    boxes = [0] * len(resources.CLASSES)
    random.seed(0)

    for images, labels, split, mapping in sources():
        for image in sorted(images.iterdir()):
            label = labels / f"{image.stem}.txt"

            if image.suffix.lower() not in IMAGE_SUFFIXES or not label.exists():
                continue

            # The same pictures are shared between several of the public datasets.
            digest = hashlib.sha1(image.read_bytes()).hexdigest()

            if digest in seen:
                continue

            seen.add(digest)
            lines = convert(label, mapping)

            # A picture with none of our resources still teaches what is not one.
            target = "val" if split in ("valid", "test") or (
                split is None and random.random() < VALIDATION_SHARE) else "train"

            (merged / target / "images").mkdir(parents=True, exist_ok=True)
            (merged / target / "labels").mkdir(parents=True, exist_ok=True)

            shutil.copy2(image, merged / target / "images" / f"{digest}{image.suffix.lower()}")
            (merged / target / "labels" / f"{digest}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

            counts[target] += 1
            for line in lines:
                boxes[int(line.split()[0])] += 1

    if counts["train"] == 0:
        sys.exit("Nothing to merge, run download or capture and autolabel first")

    (merged / "data.yaml").write_text(yaml.safe_dump({
        "path": str(merged),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(resources.CLASSES)),
    }, sort_keys=False))

    print(f"\n{counts['train']} training and {counts['val']} validation pictures in {merged}")

    for name, count in zip(resources.CLASSES, boxes, strict=True):
        print(f"  {name}: {count} boxes{'  <- too few, capture and label more of it' if count < 200 else ''}")


def capture(every, minutes, window_name):
    """
    Save a screenshot of the game every few seconds, skipping the ones looking like the
    last one saved, standing still teaches the model nothing.
    """
    import cv2 as cv
    import numpy as np

    from albion_bot.platform import DEFAULT_WINDOW_NAME, enable_dpi_awareness
    from albion_bot.platform.capture import GameWindow

    enable_dpi_awareness()
    window = GameWindow(window_name or DEFAULT_WINDOW_NAME)
    folder = paths.DATASET_CAPTURED / "images"
    folder.mkdir(parents=True, exist_ok=True)

    last = None
    saved = 0
    end = time() + minutes * 60

    print(f"Saving a screenshot every {every}s for {minutes} min in {folder}, ctrl+c to stop")

    try:
        while time() < end:
            frame = window.grab()
            small = cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (160, 90))

            if last is None or float(np.mean(cv.absdiff(small, last))) > 8:
                cv.imwrite(str(folder / f"{int(time() * 1000)}.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 92])
                last = small
                saved += 1

            sleep(every)
    except KeyboardInterrupt:
        pass

    print(f"{saved} screenshots saved")


def autolabel(confidence):
    """
    Label the captured screenshots with the current model. The labels are guesses: look
    through them, in Roboflow or labelImg for instance, before trusting them, then merge
    and train again. Each round makes the next guesses better.
    """
    from ultralytics import YOLO

    from albion_bot.vision.detector import model_path

    path = model_path()

    if path is None:
        sys.exit(f"No model in {paths.MODELS} yet, train one on the downloaded datasets first")

    model = YOLO(str(path), task="detect")
    canonical = {name: index for index, name in enumerate(resources.CLASSES)}
    mapping = {int(k): canonical.get(str(resources.resolve(str(v)))) for k, v in model.names.items()}

    images = paths.DATASET_CAPTURED / "images"
    labels = paths.DATASET_CAPTURED / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    done = 0

    for image in sorted(images.iterdir()) if images.exists() else []:
        label = labels / f"{image.stem}.txt"

        if image.suffix.lower() not in IMAGE_SUFFIXES or label.exists():
            continue

        result = model.predict(str(image), conf=confidence, verbose=False)[0]
        lines = []

        for class_id, xywhn in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist(), strict=True):
            target = mapping.get(int(class_id))

            if target is not None:
                lines.append(f"{target} " + " ".join(f"{value:.6f}" for value in xywhn))

        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        done += 1

    print(f"{done} screenshots labeled in {labels}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("download", help="download the public Albion datasets")
    command.add_argument("datasets", nargs="*", default=DATASETS, help="workspace/project, all known ones by default")

    commands.add_parser("merge", help="merge everything into datasets/albion")

    command = commands.add_parser("capture", help="save screenshots of the game while you play")
    command.add_argument("--every", type=float, default=2.0, help="seconds between two screenshots")
    command.add_argument("--minutes", type=float, default=30.0, help="how long to capture")
    command.add_argument("--window", default=None, help="title of the game window")

    command = commands.add_parser("autolabel", help="label the captured screenshots with the current model")
    command.add_argument("--confidence", type=float, default=0.4)

    arguments = parser.parse_args()

    if arguments.command == "download":
        download(arguments.datasets)
    elif arguments.command == "merge":
        merge()
    elif arguments.command == "capture":
        capture(arguments.every, arguments.minutes, arguments.window)
    elif arguments.command == "autolabel":
        autolabel(arguments.confidence)


if __name__ == "__main__":
    main()
