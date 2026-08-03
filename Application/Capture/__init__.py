from abc import abstractmethod
from mss import mss
from numpy import array

from Application.game import DEFAULT_WINDOW_NAME


class ScreenInformation:
    top = 0
    left = 0
    width = 0
    height = 0

    ALBION_HEADER_HEIGHT = 30

    def __init__(self, top, left, width, height, header=ALBION_HEADER_HEIGHT):
        """
        :param header: Height of the title bar to cut out of the capture, so the model
                       only sees the game. Window managers drawing no decoration around
                       the game, every Wayland compositor for instance, pass 0 here.
        """
        self.top = top + header
        self.left = left
        self.width = width
        self.height = height - header

    def center(self):
        return self.left + self.width / 2, self.top + self.height / 2

    def __str__(self):
        return f"Screen located at ({self.left}x, {self.top}y) with size of ({self.width}w, {self.height}h)"


class Capture:
    WINDOWS_NAME = DEFAULT_WINDOW_NAME

    def __init__(self, window_name=WINDOWS_NAME):
        self.windowName = window_name

        self.screen = self.__get_screen_information()

        self.grab_coordinates = {
            "top": 0,
            "left": 0,
            "width": 0,
            "height": 0
        }

    def __get_screen_information(self) -> ScreenInformation | None:
        # mss needs X11, and a Wayland session running no XWayland has none. Only the
        # window is grabbed anyway, so the size of the screen is not worth failing on.
        try:
            with mss() as sct:
                info = sct.monitors[1]
                return ScreenInformation(
                    top=info["top"],
                    left=info["left"],
                    width=info["width"],
                    height=info["height"]
                )
        except Exception:
            return None

    def focus(self) -> bool:
        """
        Bring the game in front of everything else.

        The capture grabs a rectangle of the screen, not the private picture of the
        game, so any window sitting over it is what the model gets shown. The bot has
        no reason to keep looking at a browser.

        :return: True when the game was brought in front, False when the platform has
                 no way to do it, which is not worth failing on.
        """
        return False

    @abstractmethod
    def get_window_information(self) -> ScreenInformation | None:
        pass

    @abstractmethod
    def __get_window_id(self) -> int:
        pass

    def screenshot(self) -> array:
        with mss() as sct:
            return array(sct.grab(self.grab_coordinates))

