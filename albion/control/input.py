"""
Driving the mouse and the keyboard.

One detail in here is worth more than the rest of the file: the cursor is moved, then
waited on, then clicked, as three separate steps. The previous bot called
`pyautogui.leftClick(x, y, interval=0.5)` and looked entirely correct, but pyautogui's
interval is the gap between repeated clicks and it sleeps *after* the button has already
gone down. So the cursor was placed on the tree and the button pressed on the very same
instant, and only then did anything wait.

Albion works out what is under the cursor before it reads the button, so every one of
those clicks was resolved against wherever the cursor had been previously, which was the
corner the bot parks in. The character never moved and never gathered anything, while the
bot cheerfully reported taking charges from a dozen trees. The move also takes time
rather than teleporting, for the same reason.

Wayland cannot be driven by pyautogui at all: it talks X11, and a Wayland compositor
never lets a client move the real cursor, so the clicks land in the void. There the
cursor is placed by the compositor and the buttons pressed by ydotool, which writes to
/dev/uinput underneath it and cannot be told from a real mouse.
"""

import os
import shutil
import subprocess
from platform import system
from time import sleep, time

from .. import logs

log = logs.get("control.input")

# Linux input event codes, from linux/input-event-codes.h.
KEY_CODES = {
    "alt": 56, "ctrl": 29, "shift": 42, "esc": 1, "space": 57, "tab": 15, "enter": 28,
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34, "h": 35, "i": 23,
    "j": 36, "k": 37, "l": 38, "m": 50, "n": 49, "o": 24, "p": 25, "q": 16, "r": 19,
    "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}

YDOTOOL_LEFT_CLICK = "0xC0"


class Stopped(Exception):
    """The user asked the bot to stop, by the corner or by the button."""


class Controller:
    """What the bot needs from a mouse and a keyboard."""

    def __init__(self, config):
        self.config = config
        self.stop_requested = False
        self.last_failsafe = 0.0

    # ------------------------------------------------------------------ overridden

    def _move(self, x, y):
        raise NotImplementedError

    def _click(self):
        raise NotImplementedError

    def position(self):
        raise NotImplementedError

    def press(self, key):
        raise NotImplementedError

    # ------------------------------------------------------------------ shared

    def move(self, x, y):
        """
        Take the cursor somewhere, over time rather than instantly.

        :param x: Screen position.
        :param y: Screen position.
        """
        self.check()
        self._move(int(x), int(y))

    def click(self, x, y):
        """
        Click something, having first let the game notice the cursor is on it.

        :param x: Screen position.
        :param y: Screen position.
        """
        self.move(x, y)
        self.wait(self.config.input.click_delay)
        self._click()

    def park(self, rect):
        """
        Put the cursor out of the way, so the tooltip stops covering the world.

        :param rect: The game window.
        """
        self.move(*rect.fraction(self.config.input.park_x, self.config.input.park_y))

    def wait(self, seconds):
        """
        Sleep, still watching for the panic corner.

        Gathering a node takes tens of seconds, and the mouse thrown in the corner has to
        stop the bot during that time too and not only between two nodes.

        :param seconds: How long to wait.
        """
        end = time() + seconds

        while True:
            remaining = end - time()

            if remaining <= 0:
                break

            sleep(min(remaining, 0.2))
            self.check()

    def check(self):
        """
        Stop the bot when the user threw the mouse in the corner, or pressed stop.

        Asking the compositor for the cursor costs a call, so it is not asked more than
        once a second.
        """
        if self.stop_requested:
            raise Stopped("stop was asked for")

        if time() - self.last_failsafe < 1.0:
            return

        self.last_failsafe = time()

        try:
            x, y = self.position()
        except Exception:
            return

        radius = self.config.input.failsafe_radius

        if x <= radius and y <= radius:
            # Named because the bot parks the cursor near that corner itself, and a panic
            # button going off on its own is otherwise a mystery.
            raise Stopped(f"the mouse was put in the corner of the screen, at ({x}, {y})")


class PyAutoGUIController(Controller):
    """Windows, MacOS and X11."""

    def __init__(self, config):
        super().__init__(config)

        import pyautogui

        # Its own corner check fires only on the calls it is given, and the bot spends
        # most of its time in waits it knows nothing about, so the check above does it.
        pyautogui.FAILSAFE = False

        self.pyautogui = pyautogui

    def _move(self, x, y):
        self.pyautogui.moveTo(x, y, duration=self.config.input.move_duration)

    def _click(self):
        self.pyautogui.click()

    def position(self):
        return self.pyautogui.position()

    def press(self, key):
        self.check()
        self.pyautogui.press(key)


class WaylandController(Controller):
    """Wayland, the compositor for the cursor and ydotool for the buttons."""

    def __init__(self, config):
        super().__init__(config)
        self.socket = self.__socket()

    @staticmethod
    def __socket():
        for candidate in (os.environ.get("YDOTOOL_SOCKET"),
                          "/run/user/%s/.ydotool_socket" % os.getuid(),
                          "/tmp/.ydotool_socket"):
            if candidate and os.path.exists(candidate):
                return candidate

        raise RuntimeError(
            "ydotool is not running. Start ydotoold, it is what presses the buttons on "
            "Wayland because a client is not allowed to.")

    def __ydotool(self, arguments):
        if shutil.which("ydotool") is None:
            raise RuntimeError("ydotool is not installed")

        subprocess.run(["ydotool", *arguments], check=True, capture_output=True,
                       env={**os.environ, "YDOTOOL_SOCKET": self.socket})

    def _move(self, x, y):
        # Placed by the compositor, which is the only one that knows where things are.
        subprocess.run(["hyprctl", "dispatch", "movecursor", str(x), str(y)],
                       check=False, capture_output=True)
        sleep(self.config.input.move_duration)

    def _click(self):
        self.__ydotool(["click", YDOTOOL_LEFT_CLICK])

    def position(self):
        result = subprocess.run(["hyprctl", "cursorpos"], check=True,
                                capture_output=True, text=True)
        x, y = result.stdout.strip().split(",")

        return int(x), int(y.strip())

    def press(self, key):
        self.check()
        code = KEY_CODES.get(key.lower())

        if code is None:
            raise RuntimeError(f"the key {key!r} is not known, add it to KEY_CODES")

        self.__ydotool(["key", f"{code}:1", f"{code}:0"])


def build(config):
    """
    The controller for this session.

    :param config: Config.
    :return: Controller.
    """
    wayland = (os.environ.get("XDG_SESSION_TYPE") == "wayland"
               or bool(os.environ.get("WAYLAND_DISPLAY")))

    if system() != "Windows" and wayland:
        log.info("Wayland session, driving the cursor through the compositor")
        return WaylandController(config)

    return PyAutoGUIController(config)
