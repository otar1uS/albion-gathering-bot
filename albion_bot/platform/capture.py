"""
The game window: where it is, what it shows, and how a pixel of a frame maps to the
screen.
"""

import subprocess
import threading
from time import monotonic

import cv2 as cv
import numpy as np

from albion_bot.platform import IS_LINUX, IS_MAC, IS_WINDOWS, OWN_WINDOW_TITLES, is_wayland


class GameNotFound(Exception):
    pass


def backend():
    if IS_WINDOWS:
        from albion_bot.platform import windows
        return windows
    if IS_MAC:
        from albion_bot.platform import macos
        return macos
    if IS_LINUX:
        from albion_bot.platform import linux
        return linux

    raise GameNotFound("only Windows, Linux and macOS are supported")


def _matches(window, name):
    name = name.lower()
    return name in window.title.lower() or name in getattr(window, "application", "").lower()


def find_native(name):
    """
    Look for the game by a part of its title.

    :return: (native window, rect), None when there is no such window.
    """
    platform = backend()
    candidates = []

    for window in platform.windows():
        if window.title in OWN_WINDOW_TITLES or not _matches(window, name):
            continue

        # The launcher is called Albion Online as well, gathering in it would be short.
        if "launcher" not in name.lower() and "launcher" in (window.title + getattr(window, "application", "")).lower():
            continue

        rect = platform.rect(window)

        if rect is not None:
            candidates.append((window, rect))

    if not candidates:
        return None

    # The biggest one, the game fills the screen while a chat or an overlay does not.
    return max(candidates, key=lambda candidate: candidate[1].width * candidate[1].height)


def open_windows():
    """
    :return: Titles of the open windows, to tell the user what can be picked.
    """
    try:
        return [str(window) for window in backend().windows()]
    except Exception as e:
        return [f"(could not list the windows: {e})"]


_local = threading.local()


def _mss():
    # An mss instance holds handles that belong to the thread that opened them.
    if getattr(_local, "sct", None) is None:
        import mss
        _local.sct = mss.mss()

    return _local.sct


class GameWindow:
    # Time in second a window position is trusted before being asked again, so moving or
    # resizing the game does not send the bot clicking next to it.
    GEOMETRY_LIFETIME = 1.0

    def __init__(self, name):
        self.name = name
        found = find_native(name)

        if found is None:
            raise GameNotFound(f"no window named {name!r}, start the game. Open windows: "
                               f"{', '.join(open_windows()) or 'none'}")

        self.native, self._rect = found
        self._rect_at = monotonic()
        self.wayland = is_wayland()

    def __str__(self):
        return f"{self.native} {self._rect}"

    @property
    def rect(self):
        if monotonic() - self._rect_at >= self.GEOMETRY_LIFETIME:
            self._rect_at = monotonic()

            if IS_WINDOWS:
                rect = backend().rect(self.native)
            else:
                # The window object of Linux and macOS is a snapshot, ask again.
                found = find_native(self.name)
                rect = None if found is None else found[1]

                if found is not None:
                    self.native = found[0]

            if rect is None:
                raise GameNotFound(f"the game window {self.native} is closed or minimized")

            self._rect = rect

        return self._rect

    def grab(self):
        """
        :return: BGR frame of what the game draws, without borders.
        """
        rect = self.rect

        if self.wayland:
            if not getattr(self.native, "visible", True):
                raise GameNotFound(f"the game {self.native} is on a workspace that is not shown, "
                                   f"leave it in front while the bot runs")

            geometry = f"{rect.left},{rect.top} {rect.width}x{rect.height}"
            result = subprocess.run(["grim", "-g", geometry, "-t", "ppm", "-"], capture_output=True, timeout=5)

            if result.returncode != 0:
                raise GameNotFound(f"grim could not grab {geometry}: "
                                   f"{result.stderr.decode('utf8', 'replace').strip()}")

            frame = cv.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv.IMREAD_COLOR)

            if frame is None:
                raise GameNotFound("grim gave back an image that cannot be read")

            return frame

        shot = _mss().grab({"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height})
        return cv.cvtColor(np.asarray(shot), cv.COLOR_BGRA2BGR)

    def focus(self):
        """
        :return: True when the game is in front.
        """
        return backend().focus(self.native)

    def to_screen(self, x, y, shape):
        """
        Turn a position in a frame into a position on the screen. The frame is bigger
        than the window on a scaled Wayland monitor, hence the ratio.

        :return: (x, y) in whole screen pixels.
        """
        rect = self.rect
        height, width = shape[:2]

        return (int(round(rect.left + x * rect.width / width)),
                int(round(rect.top + y * rect.height / height)))


def stop_key_pressed():
    try:
        return backend().stop_key_pressed()
    except Exception:
        return False
