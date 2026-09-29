"""
The YOLO model finding the resources in a frame.
"""

from dataclasses import dataclass
from pathlib import Path

from albion_bot import paths
from albion_bot.vision import resources

# Size the frames are shrunk to for the model, the one it is trained at.
IMG_SIZE = 640


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    profile: resources.Profile

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def click_point(self):
        return (self.x1 + self.x2) / 2, self.y1 + (self.y2 - self.y1) * self.profile.click_height

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def box(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


def model_path():
    """
    :return: The model to load, the OpenVINO export first, None when there is none.
    """
    exported = paths.MODEL_OPENVINO.joinpath("metadata.yaml")

    # An export older than the weights holds a previous training.
    if exported.exists() and (not paths.MODEL.exists() or exported.stat().st_mtime >= paths.MODEL.stat().st_mtime):
        return paths.MODEL_OPENVINO

    if paths.MODEL.exists():
        return paths.MODEL

    return None


def torch_device():
    """
    :return: Best device torch can see: Intel Arc, then Nvidia, then the CPU.
    """
    import torch

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    if torch.cuda.is_available():
        return "0"

    return "cpu"


class Detector:

    def __init__(self, path=None, confidence=0.5, device=None):
        """
        :param path: Weights or OpenVINO folder, the ones of models/ by default.
        :param confidence: Lowest confidence of a detection to keep.
        :param device: Where to run the model, found on its own by default.
        """
        path = Path(path) if path is not None else model_path()

        if path is None or not path.exists():
            raise FileNotFoundError(f"no model in {paths.MODELS}, build one with tools/dataset.py "
                                    f"then tools/train.py, see the README")

        # Imported here, torch takes seconds to load and the interface does not need it.
        from ultralytics import YOLO

        openvino = path.is_dir()

        # Ultralytics falls back on the CPU by itself when OpenVINO sees no Intel GPU.
        self.device = device or ("intel:gpu" if openvino else torch_device())
        self.path = path
        self.confidence = confidence

        try:
            self.model = YOLO(str(path), task="detect")
        except Exception as e:
            raise RuntimeError(f"could not load {path}: {e}. Weights of the old yolov5 repository "
                               f"cannot be loaded, train new ones with tools/train.py") from e

        names = self.model.names
        self.names = {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        self.profiles = {class_id: resources.resolve(name) for class_id, name in self.names.items()}

    def knows(self, profile):
        return profile in self.profiles.values()

    def describe(self):
        """
        :return: One line per class of the model, telling what the bot makes of it.
        """
        return [f"class {k} {v!r} -> {self.profiles[k]}" for k, v in sorted(self.names.items())]

    def detect(self, frame):
        """
        :param frame: BGR frame of the game.
        :return: Detections of every class the bot knows, in frame pixels.
        """
        wanted = sorted(k for k, profile in self.profiles.items() if profile is not resources.UNKNOWN)
        result = self.model.predict(frame, imgsz=IMG_SIZE, conf=self.confidence, classes=wanted,
                                    device=self.device, verbose=False)[0]

        detections = []

        for xyxy, confidence, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                              result.boxes.cls.tolist(), strict=True):
            class_id = int(class_id)
            detections.append(Detection(*xyxy, float(confidence), self.names[class_id], self.profiles[class_id]))

        return detections
