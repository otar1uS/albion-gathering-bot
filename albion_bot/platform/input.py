"""
Mouse and keyboard.

Windows, macOS and X11 are driven by pyautogui. Wayland lets no program move the real
cursor, so Hyprland places it and ydotool, writing to /dev/uinput below the compositor,
presses the buttons and the keys.
"""

import os
import shutil
import subprocess
from time import sleep

from albion_bot.platform import IS_LINUX, is_wayland

# Distance in pixel from the top left corner of the screen under which the bot stops,
# the panic button of pyautogui, reproduced for Wayland.
FAILSAFE_RADIUS = 2

# Linux input event codes, linux/input-event-codes.h.
KEY_CODES = {
    "alt": 56, "ctrl": 29, "shift": 42, "esc": 1, "space": 57, "tab": 15, "enter": 28,
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34, "h": 35, "i": 23,
    "j": 36, "k": 37, "l": 38, "m": 50, "n": 49, "o": 24, "p": 25, "q": 16, "r": 19,
    "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}


class FailSafe(Exception):
    """The user threw the mouse into the top left corner to stop the bot."""


class Pointer:
    # Time in second between putting the cursor somewhere and clicking. The game ignores
    # a click landing on the same frame the cursor arrived.
    CLICK_DELAY = 0.12

    def move(self, x, y):
        raise NotImplementedError

    def position(self):
        raise NotImplementedError

    def click(self, x, y):
        raise NotImplementedError

    def press(self, key):
        raise NotImplementedError

    def hotkey(self, *keys):
        raise NotImplementedError

    def check_failsafe(self):
        x, y = self.position()

        if x <= FAILSAFE_RADIUS and y <= FAILSAFE_RADIUS:
            raise FailSafe("stopped by the mouse in the top left corner of the screen")


class PyAutoGUIPointer(Pointer):

    def __init__(self):
        import pyautogui

        self.pyautogui = pyautogui
        pyautogui.FAILSAFE = True
        # pyautogui waits 0.1 s after every call by default, which adds up while walking.
        pyautogui.PAUSE = 0.02

    def __guarded(self, call, *arguments, **named):
        try:
            return call(*arguments, **named)
        except self.pyautogui.FailSafeException:
            raise FailSafe("stopped by the mouse in a corner of the screen") from None

    def move(self, x, y):
        self.__guarded(self.pyautogui.moveTo, x, y)

    def position(self):
        return tuple(self.pyautogui.position())

    def click(self, x, y):
        self.__guarded(self.pyautogui.moveTo, x, y)
        sleep(self.CLICK_DELAY)
        # Down and up by hand, a click sent in one go is sometimes missed by the game.
        self.__guarded(self.pyautogui.mouseDown)
        sleep(0.03)
        self.__guarded(self.pyautogui.mouseUp)

    def press(self, key):
        self.__guarded(self.pyautogui.press, key)

    def hotkey(self, *keys):
        self.__guarded(self.pyautogui.hotkey, *keys)


class WaylandPointer(Pointer):
    """Hyprland for the cursor, ydotool for the buttons and the keys."""

    def __init__(self):
        self.socket = ydotool_socket()

    def __ydotool(self, arguments):
        environment = dict(os.environ)

        if self.socket is not None:
            environment["YDOTOOL_SOCKET"] = self.socket

        result = subprocess.run(["ydotool"] + arguments, capture_output=True, timeout=5, env=environment)

        if result.returncode != 0:
            raise RuntimeError(f"ydotool {' '.join(arguments)} failed: "
                               f"{result.stderr.decode('utf8', 'replace').strip()}")

    def move(self, x, y):
        result = subprocess.run(["hyprctl", "dispatch", "movecursor", str(int(x)), str(int(y))],
                                capture_output=True, timeout=5)

        if result.returncode != 0:
            raise RuntimeError(f"hyprctl could not move the cursor: "
                               f"{result.stderr.decode('utf8', 'replace').strip()}")

    def position(self):
        result = subprocess.run(["hyprctl", "cursorpos"], capture_output=True, timeout=5)

        try:
            x, y = result.stdout.decode("utf8", "replace").split(",")
            return int(x.strip()), int(y.strip())
        except ValueError:
            return 9999, 9999

    def click(self, x, y):
        self.check_failsafe()
        self.move(x, y)
        sleep(self.CLICK_DELAY)
        self.__ydotool(["click", "0xC0"])

    def __codes(self, keys):
        codes = []

        for key in keys:
            code = KEY_CODES.get(key.lower())

            if code is None:
                raise RuntimeError(f"the key {key!r} is not known, add it in KEY_CODES")

            codes.append(code)

        return codes

    def press(self, key):
        code = self.__codes([key])[0]
        self.__ydotool(["key", f"{code}:1", f"{code}:0"])

    def hotkey(self, *keys):
        codes = self.__codes(keys)
        self.__ydotool(["key"] + [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)])


def ydotool_socket():
    if os.environ.get("YDOTOOL_SOCKET"):
        return os.environ["YDOTOOL_SOCKET"]

    for candidate in (f"/run/user/{os.getuid()}/.ydotool_socket", "/run/.ydotool_socket", "/tmp/.ydotool_socket"):
        if os.path.exists(candidate):
            return candidate

    return None


def availability():
    """
    :return: (ok, detail) telling whether the mouse and the keyboard can be driven.
    """
    if not (IS_LINUX and is_wayland()):
        try:
            import pyautogui  # noqa: F401
        except Exception as e:
            return False, f"pyautogui cannot be loaded: {e}"

        return True, "pyautogui"

    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") is None:
        return False, "only Hyprland is supported on Wayland, log in an X11 session instead"

    if shutil.which("ydotool") is None:
        return False, "ydotool is missing: sudo pacman -S ydotool"

    socket = ydotool_socket()

    if socket is None:
        return False, "ydotoold is not running: systemctl --user enable --now ydotoold"

    if not os.access(socket, os.W_OK):
        return False, f"{socket} cannot be written to: sudo usermod -aG input $USER, then log in again"

    return True, f"hyprctl and ydotool, through {socket}"


def create() -> Pointer:
    ok, detail = availability()

    if not ok:
        raise RuntimeError(f"the mouse cannot be driven: {detail}")

    return WaylandPointer() if IS_LINUX and is_wayland() else PyAutoGUIPointer()
