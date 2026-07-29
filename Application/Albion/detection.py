import cv2 as cv
import torch
from Application import paths
from Application.Albion import resources, screen
from Application.Capture.Factory import CaptureFactory
from time import time
from math import sqrt
from pathlib import Path


class AlbionDetection:
    MODEL_NAME = paths.MODEL
    IMG_SIZE = screen.IMG_SIZE
    CONFIDENCE = 0.5

    def __init__(self,
                 model_name=MODEL_NAME,
                 debug=False,
                 confidence=CONFIDENCE,
                 window_name="Albion Online Client",
                 targets=None,
                 preview=None
                 ):
        """
        Initialize the AlbionDetection object.

        :param model_name: Path of the YOLOv5 weights.
        :param debug: Flag to enable debug mode.
        :param confidence: Lowest confidence of a detection to keep.
        :param window_name: Title of the game window.
        :param targets: Resource names to detect, None to detect all of them.
        :param preview: Flag to show the window drawing the detections, follows debug
                        when left to None. The interface logs without that window.
        """
        self.model_name = model_name
        self.debug = debug
        self.preview = debug if preview is None else preview
        self.model = self._load_model()
        self.classes = self._load_classes()
        self.targets = resources.resolve_targets(targets)
        self.target_ids = self._load_target_ids()
        self.window_capture = self._load_capture(window_name)
        self.confidence = confidence
        self.character_position_X = self.IMG_SIZE / 2
        self.character_position_Y = self.IMG_SIZE / 2 - 60

    def _process_image(self, img):
        """
        Preprocess the image, see Application.Albion.screen.

        :param img: Input image.
        :return: Processed image.
        """
        return screen.process(img)

    def _load_capture(self, window_name):
        """
        Find the Albion window.

        :param window_name: Title of the game window.
        :return: Capture bound to the window.
        """
        capture = CaptureFactory(window_name).capture

        if capture is None:
            raise Exception(f"Failed to capture the window {window_name}")

        return capture

    def _load_classes(self):
        """
        Load class information.

        :return: Dictionary containing class information.
        """
        names = self.model.names

        # Depending on the yolov5 version used to train the weights, names is
        # either a dict of id to label or a plain list of labels.
        if not isinstance(names, dict):
            names = dict(enumerate(names))

        classes = {}
        for k, v in names.items():
            classes[k] = {
                "label": v,
                "color": (0, 255, k * 10)
            }
        return classes

    def _load_target_ids(self):
        """
        Find the class ids of the wanted resources.

        :return: Set of class ids to keep during a prediction.
        """
        ids = set()

        for k, v in self.classes.items():
            profile = resources.resolve(v["label"])

            if profile in self.targets:
                ids.add(k)

            if self.debug:
                # Classes matching no resource, monsters for example, are never clicked.
                print(f"Class {k} {v['label']!r} -> {profile}{'' if k in ids else ' (ignored)'}")

        if len(ids) == 0:
            raise Exception(
                f"None of the targets {[str(t) for t in self.targets]} is known by the model, "
                f"available classes are {[v['label'] for v in self.classes.values()]}"
            )

        return ids

    def profile_of(self, class_id):
        """
        Find the gathering profile of a detected class.

        :param class_id: Class id given by the model.
        :return: Matching ResourceProfile.
        """
        return resources.resolve(self.classes[int(class_id)]["label"])

    def draw_boxes(self, img, coordinates):
        """
        Draw bounding boxes on the image.

        :param img: Input image.
        :param coordinates: Bounding box coordinates.
        """
        for coord in coordinates:
            x1, y1, x2, y2 = (int(value) for value in coord[:4])
            confidence, class_id = float(coord[4]), self.classes[int(coord[5])]
            cv.rectangle(img, (x1, y1), (x2, y2), class_id["color"], 2)
            label = f"{class_id['label']} {confidence:.2f}"
            cv.putText(img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, class_id["color"], 2)

        self.__cross_line(img)

        cv.drawMarker(img, (int(self.character_position_X), int(self.character_position_Y)), (255, 255, 255),
                      cv.MARKER_DIAMOND, 10, 2)
        self.__marker_closest(img, coordinates)
        # cv.imshow("Boxes", img)

    def __cross_line(self, img):
        cv.line(img, (0, int(self.character_position_Y)), (self.IMG_SIZE, int(self.character_position_Y)),
                (255, 255, 255), 2)
        cv.line(img, (int(self.character_position_X), 0), (int(self.character_position_X), self.IMG_SIZE),
                (255, 255, 255), 2)

    def __marker_closest(self, img, coordinates):
        closest = self.closest_point(coordinates)

        if closest is not None:
            cv.drawMarker(img, (int(closest[0]), int(closest[1])), (37, 150, 190), cv.MARKER_CROSS, 10, 2)
            cv.putText(img, "Closest", (int(closest[0]), int(closest[1]) + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)

    def centers(self, coordinates):
        """
        Compute the center of every box, from the closest to the character to the
        farthest.

        :param coordinates: Bounding box coordinates.
        :return: List of (center_x, center_y, class_id) in image coordinates.
        """
        centers = []

        for coord in coordinates:
            x1, y1, x2, y2 = (int(value) for value in coord[:4])

            center_x, center_y = ((x2 - x1) / 2) + x1, ((y2 - y1) / 2) + y1

            distance = sqrt(
                abs(self.character_position_X - center_x) ** 2 + abs(self.character_position_Y - center_y) ** 2)

            centers.append((distance, center_x, center_y, int(coord[5])))

        centers.sort(key=lambda center: center[0])

        return [(center_x, center_y, class_id) for _, center_x, center_y, class_id in centers]

    def closest_point(self, coordinates):
        """
        Find the box the closest to the character.

        :param coordinates: Bounding box coordinates.
        :return: (center_x, center_y, class_id) in image coordinates, None when
                 there is no box.
        """
        centers = self.centers(coordinates)

        if len(centers) == 0:
            return None

        return centers[0]

    def __convert_coordinates_to_screen_position(self, center_x, center_y):

        center_x = ((center_x * self.window_capture.window.width) / self.IMG_SIZE) + self.window_capture.window.left
        center_y = ((center_y * self.window_capture.window.height) / self.IMG_SIZE) + self.window_capture.window.top

        # pyautogui works with pixels, and a torch value would leak here otherwise.
        return int(center_x), int(center_y)

    def character_screen_position(self):
        """
        Position of the character on the screen.

        :return: (x, y) in screen coordinates.
        """
        return self.__convert_coordinates_to_screen_position(self.character_position_X, self.character_position_Y)

    def _load_model(self):
        """
        Load the YOLOv5 model.

        :return: Loaded YOLOv5 model.
        """
        if not paths.YOLOV5.joinpath("hubconf.py").exists():
            raise Exception(
                f"The yolov5 submodule is missing in {paths.YOLOV5}, "
                f"get it with: git submodule update --init --recursive"
            )

        if not Path(self.model_name).exists():
            raise Exception(
                f"The trained weights are missing in {self.model_name}, "
                f"train them with yolov5 and drop the best.pt file there"
            )

        try:
            model = torch.hub.load(str(paths.YOLOV5), 'custom', path=str(self.model_name), source="local",
                                   force_reload=True, verbose=True)

        except Exception as e:
            raise Exception(f"Failed to load the model: {e}")

        return model

    def predict(self, ignore=None):
        """
        Make predictions using the YOLOv5 model.

        :param ignore: Callable taking a screen position and returning True when the
                       resource has to be skipped, None to keep all of them.
        :return: Screen position, profile of the closest resource and the image,
                 position and profile are None when nothing has been detected.
        """
        loop_time = time()
        img = self._process_image(self.window_capture.screenshot())
        res = self.model(img)
        coordinates = [coord for coord in res.xyxy[0]
                       if float(coord[4]) > self.confidence and int(coord[5]) in self.target_ids]

        if self.preview:
            self.draw_boxes(img, coordinates)
            cv.putText(img, f'FPS {1 / (time() - loop_time):.1f}', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            cv.imshow("Founded", img)

        for center_x, center_y, ressource in self.centers(coordinates):
            center_x, center_y = self.__convert_coordinates_to_screen_position(center_x, center_y)

            if ignore is not None and ignore(center_x, center_y):
                continue

            return center_x, center_y, self.profile_of(ressource), img

        return None, None, None, img
