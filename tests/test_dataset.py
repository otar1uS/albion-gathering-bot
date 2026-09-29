import yaml

from albion_bot import paths
from tools import dataset


def test_box_reads_boxes_and_polygons():
    assert dataset.box(["0.5", "0.5", "0.2", "0.1"]) == (0.5, 0.5, 0.2, 0.1)
    x, y, w, h = dataset.box(["0.1", "0.2", "0.3", "0.2", "0.3", "0.4", "0.1", "0.4"])
    assert (round(x, 6), round(y, 6), round(w, 6), round(h, 6)) == (0.2, 0.3, 0.2, 0.2)
    assert dataset.box(["a", "b", "c", "d"]) is None


def test_merge_renames_classes_and_drops_duplicates(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "someone__albion__v1"
    monkeypatch.setattr(paths, "DATASET_RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "DATASET_CAPTURED", tmp_path / "captured")
    monkeypatch.setattr(paths, "DATASET_MERGED", tmp_path / "albion")

    (raw / "train" / "images").mkdir(parents=True)
    (raw / "train" / "labels").mkdir(parents=True)
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": ["monster", "rough log", "iron ore", "albion"]}))

    for name, content in (("a", b"one"), ("b", b"two"), ("c", b"one")):
        (raw / "train" / "images" / f"{name}.jpg").write_bytes(content)
        (raw / "train" / "labels" / f"{name}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.1 0.1\n3 0.1 0.1 0.1 0.1\n")

    dataset.merge()

    merged = tmp_path / "albion"
    labels = sorted((merged / "train" / "labels").glob("*.txt"))
    assert len(labels) == 2  # c.jpg is the same picture as a.jpg

    lines = labels[0].read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["5", "0", "2"]  # monster, tree, ore, "albion" dropped

    data = yaml.safe_load((merged / "data.yaml").read_text())
    assert data["names"] == {0: "tree", 1: "stone", 2: "ore", 3: "fiber", 4: "hide", 5: "monster"}
