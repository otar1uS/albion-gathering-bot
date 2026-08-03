#!/usr/bin/env python
"""
Download the public Albion datasets and merge them into one the model can train on.

This is steps 3 and 4 of training/albion_training_merged.ipynb, lifted out of Colab so
the whole job runs on this PC. It writes datasets/albion_merged, which train_local.py
then trains on.

The first model was trained on one dataset of 657 pictures, all shot in the same bright
grassy place. It scores 0.74 on a tree there and 0.26 on a tree in Forgotten Woods, which
is why the bot goes blind the moment it walks into a dark forest. Seven datasets from
seven different people, their zones and their times of day, is the fix: a model that has
seen trees in twenty places recognises the twenty-first.

Needs a Roboflow API key from https://app.roboflow.com/settings/api. Either paste it when
asked, or put it in training/roboflow_key.txt to avoid being asked again. The datasets are
public, so one key downloads all of them.

    .venv-train\\Scripts\\python.exe training\\download_datasets.py
"""

import re
import shutil
import sys
from collections import Counter
from getpass import getpass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "datasets" / "raw"
MERGED = ROOT / "datasets" / "albion_merged"
KEY_FILE = Path(__file__).resolve().parent / "roboflow_key.txt"

# (workspace, project, short name used to prefix the files). Taken from the Universe
# urls: universe.roboflow.com/<workspace>/<project>
SOURCES = [
    ("albion-online", "ao-resource-detection", "aores"),
    ("albiononline-6no0l", "albion-bpgjq", "albion2"),
    ("joao-angelo", "albion-resources-6myx5", "joao"),
    ("ai-workspace-yolo-ghcsl", "albion-online---farm-bot", "farmbot"),
    ("albion-zmhvd", "ore1", "ore1"),
    ("albiononlinez", "albion-enemys", "enemys"),
    ("albiononline-c8fxi", "albiongathering", "original"),
]

# The same aliases as Application/Albion/resources.py, in the same order, plus monsters,
# which the bot does not gather but does walk around.
ALIASES = {
    # "rough" is deliberately absent: tier 2 wood is "Rough Logs" and tier 2 rock is
    # "Rough Stone", and tree being tried first claimed the rock. "logs" catches the wood.
    "tree": ("tree", "trees", "wood", "log", "logs", "birch", "chestnut",
             "pine", "cedar", "bloodoak", "ashenbark"),
    "stone": ("stone", "stones", "rock", "rocks", "limestone", "sandstone", "travertine",
              "granite", "slate", "basalt", "marble"),
    "ore": ("ore", "ores", "copper", "tin", "iron", "titanium", "runite", "meteorite",
            "adamantium"),
    "fiber": ("fiber", "fibre", "cotton", "flax", "hemp", "amberleaf", "sunflax", "ghost",
              "redleaf", "silk"),
    "hide": ("hide", "hides", "animal", "rabbit", "fox", "boar", "wolf", "bear", "deer"),
    "monster": ("monster", "monsters", "mob", "mobs", "enemy", "enemies", "creature"),
}

CANONICAL = list(ALIASES)


def resolve(label):
    """
    Find which of the six a dataset's class name means.

    Word matching identical to ResourceProfile.match, so whatever trains here is what the
    bot recognises later.

    :param label: Class name as the dataset spells it.
    :return: Canonical name, None when the label is not a resource.
    """
    label = label.lower().strip()
    words = re.split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", label)

    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in words or (len(alias) >= 4 and alias in label.replace(" ", "")):
                return canonical

    return None


def api_key():
    """
    Read the Roboflow key from training/roboflow_key.txt, asking for it when it is absent.

    :return: The key, without the whitespace a copy and paste tends to bring along.
    """
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()

        if key:
            return key

    key = getpass("Paste your Roboflow private API key, then press Enter: ").strip()

    if not key:
        raise SystemExit("No key given, nothing to download")

    KEY_FILE.write_text(key)
    print(f"Saved to {KEY_FILE}, it will not ask again")

    return key


def download():
    """
    Fetch the newest version of every dataset.

    One unavailable dataset is skipped and the rest still go ahead, so a single broken
    one does not waste the run.

    :return: List of (prefix, folder) for the datasets that arrived.
    """
    from roboflow import Roboflow

    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    roboflow = Roboflow(api_key=api_key())
    downloaded = []

    for workspace, name, prefix in SOURCES:
        target = DOWNLOADS / prefix

        # Downloading nine thousand pictures again on every run would be a waste, and the
        # datasets do not change from one day to the next.
        if (target / "data.yaml").exists():
            downloaded.append((prefix, target))
            print(f"HAVE  {prefix:9} {target}")
            continue

        try:
            project = roboflow.workspace(workspace).project(name)

            # The newest version, whatever number it happens to be.
            versions = sorted(project.versions(), key=lambda v: int(str(v.version).split("/")[-1]))
            dataset = versions[-1].download("yolov5", location=str(target))

            downloaded.append((prefix, Path(dataset.location)))
            print(f"OK    {prefix:9} {dataset.location}")
        except Exception as e:
            shutil.rmtree(target, ignore_errors=True)
            print(f"SKIP  {prefix:9} {type(e).__name__}: {str(e)[:110]}")

    print(f"\n{len(downloaded)} of {len(SOURCES)} datasets available")

    return downloaded


def merge(downloaded):
    """
    Rewrite every dataset onto the same six class names and pile them into one folder.

    A class matching nothing, a chest or a player name, has its boxes dropped. The picture
    is still kept: an image with no boxes teaches the model what is not a resource, which
    is what stops the bot clicking on scenery.

    :param downloaded: List of (prefix, folder) from download.
    :return: (pictures per split, boxes per class).
    """
    shutil.rmtree(MERGED, ignore_errors=True)

    for split in ("train", "valid", "test"):
        (MERGED / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED / split / "labels").mkdir(parents=True, exist_ok=True)

    instances = Counter()
    pictures = Counter()

    for prefix, location in downloaded:
        config = yaml.safe_load((Path(location) / "data.yaml").read_text())

        names = config["names"]
        names = list(names.values()) if isinstance(names, dict) else names

        mapping = {}
        print(f"\n{prefix}")

        for index, name in enumerate(names):
            canonical = resolve(name)
            mapping[index] = CANONICAL.index(canonical) if canonical else None
            print(f"   {name!r} -> {canonical or 'dropped'}")

        for split in ("train", "valid", "test"):
            images = Path(location) / split / "images"
            labels = Path(location) / split / "labels"

            if not images.exists():
                continue

            for picture in images.iterdir():
                if picture.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                kept = []
                label = labels / f"{picture.stem}.txt"

                if label.exists():
                    for line in label.read_text().splitlines():
                        parts = line.split()

                        if not parts:
                            continue

                        new = mapping.get(int(parts[0]))

                        if new is None:
                            continue

                        kept.append(" ".join([str(new)] + parts[1:]))
                        instances[CANONICAL[new]] += 1

                # The name is prefixed because two datasets happily use the same file names.
                stem = f"{prefix}_{picture.stem}"
                shutil.copy(picture, MERGED / split / "images" / f"{stem}{picture.suffix}")
                (MERGED / split / "labels" / f"{stem}.txt").write_text("\n".join(kept))
                pictures[split] += 1

    return pictures, instances


def write_config(pictures):
    """
    Write the data.yaml yolov5 reads, with absolute paths so it works from any folder.

    :param pictures: Counter of pictures per split, to know whether there is a test set.
    :return: Path of the config.
    """
    config = {
        "train": str(MERGED / "train" / "images"),
        "val": str(MERGED / "valid" / "images"),
        "nc": len(CANONICAL),
        "names": CANONICAL,
    }

    if pictures.get("test"):
        config["test"] = str(MERGED / "test" / "images")

    data_yaml = MERGED / "data.yaml"
    data_yaml.write_text(yaml.dump(config))

    return data_yaml


def main():
    downloaded = download()

    if not downloaded:
        raise SystemExit("Nothing downloaded, check the key and the connection")

    pictures, instances = merge(downloaded)

    data_yaml = write_config(pictures)

    print("\n" + "=" * 50)
    print("pictures :", dict(pictures), "total", sum(pictures.values()))
    print("boxes    :", dict(instances), "total", sum(instances.values()))
    print()

    for name in CANONICAL:
        count = instances.get(name, 0)
        note = "plenty" if count >= 500 else "thin, expect misses" if count >= 100 else "TOO FEW"
        print(f"  {name:8} {count:6} boxes   {note}")

    print(f"\nwrote {data_yaml}")
    print("now run:  .venv-train\\Scripts\\python.exe training\\train_local.py")


if __name__ == "__main__":
    sys.exit(main())
