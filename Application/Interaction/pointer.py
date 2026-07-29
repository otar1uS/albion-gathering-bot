"""
Mouse and keyboard, one backend per session.

Windows, MacOS and an X11 session are driven by pyautogui. A Wayland session cannot
be: pyautogui talks X11, and a Wayland compositor never lets a client move the real
cursor, so every click would land in the void, or in the XWayland layer the game does
not read.

Wayland is driven by two tools instead. The cursor is placed by the compositor itself,
which is the only one knowing where things are, and the buttons are pressed by ydotool,
which writes to /dev/uinput below the compositor and cannot be told apart from a real
mouse.
"""

import os
import shutil
import subprocess
from platform import system
from time import sleep

# Distance in pixel from the top left corner under which the bot gives up, the panic
# button of pyautogui reproduced for Wayland. It stays small because the bot parks the
# cursor near that corner by itself between two nodes.
FAILSAFE_RADIUS = 2

# Linux input event codes, from linux/input-event-codes.h, for the keys the bot presses.
KEY_CODES = {
    "alt": 56, "ctrl": 29, "shift": 42, "esc": 1, "space": 57, "tab": 15, "enter": 28,
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34, "h": 35, "i": 23,
    "j": 36, "k": 37, "l": 38, "m": 50, "n": 49, "o": 24, "p": 25, "q": 16, "r": 19,
    "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}

# Left button, pressed then released, as ydotool spells it.
YDOTOOL_LEFT_CLICK = "0xC0"


class FailSafe(Exception):
    """Raised when the user parked the mouse in the corner to stop the bot."""


def is_wayland():
    return os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


class Pointer:
    """What the bot needs from a mouse and a keyboard."""

    # Time in second between putting the cursor on a resource and clicking on it. The
    # game ignores a click landing on the same frame the cursor arrived.
    CLICK_DELAY = 0.5

    def move(self, x, y):
        raise NotImplementedError

    def position(self):
        raise NotImplementedError

    def left_click(self, x, y):
        raise NotImplementedError

    def press(self, key):
        raise NotImplementedError

    def hotkey(self, *keys):
        raise NotImplementedError

    def check_failsafe(self):
        """
        Stop the bot when the user threw the mouse in the top left corner.
        """
        x, y = self.position()

        if x <= FAILSAFE_RADIUS and y <= FAILSAFE_RADIUS:
            raise FailSafe("Stopped by the mouse in the corner of the screen")


class PyAutoGUIPointer(Pointer):
    """Windows, MacOS and X11."""

    def __init__(self):
        import pyautogui

        self.pyautogui = pyautogui

    def __guarded(self, call, *arguments, **named):
        """
        Run a pyautogui call, turning its own panic button into the one the rest of the
        bot knows about.
        """
        try:
            return call(*arguments, **named)
        except self.pyautogui.FailSafeException:
            raise FailSafe("Stopped by the mouse in the corner of the screen")

    def move(self, x, y):
        self.__guarded(self.pyautogui.moveTo, x, y)

    def position(self):
        return self.pyautogui.position()

    def left_click(self, x, y):
        self.__guarded(self.pyautogui.leftClick, x, y, interval=self.CLICK_DELAY)

    def press(self, key):
        self.__guarded(self.pyautogui.press, key)

    def hotkey(self, *keys):
        self.__guarded(self.pyautogui.hotkey, *keys)

    # check_failsafe is the one of Pointer. pyautogui watches the corner on its own, but
    # only on the calls it is given, and gathering a node is tens of seconds during which
    # it is never called. Reading the position instead stops the bot there too.


class WaylandPointer(Pointer):
    """
    Wayland, Hyprland for the cursor and ydotool for the buttons and the keys.
    """

    def __init__(self):
        self.socket = ydotool_socket()

    def __ydotool(self, arguments):
        environment = dict(os.environ)

        if self.socket is not None:
            environment["YDOTOOL_SOCKET"] = self.socket

        result = subprocess.run(["ydotool"] + arguments, capture_output=True,
                                timeout=5, env=environment)

        if result.returncode != 0:
            raise Exception(
                f"ydotool {' '.join(arguments)} failed: "
                f"{result.stderr.decode('utf8', 'replace').strip()}"
            )

    def move(self, x, y):
        # The compositor is asked instead of ydotool, because it is the only one
        # knowing the layout of the monitors and their scaling. ydotool would have to
        # be told, and would be wrong the day a monitor is plugged in.
        result = subprocess.run(["hyprctl", "dispatch", "movecursor", str(int(x)), str(int(y))],
                                capture_output=True, timeout=5)

        if result.returncode != 0:
            raise Exception(f"hyprctl could not move the cursor: "
                            f"{result.stderr.decode('utf8', 'replace').strip()}")

    def position(self):
        result = subprocess.run(["hyprctl", "cursorpos"], capture_output=True, timeout=5)

        if result.returncode != 0:
            return 0, 0

        try:
            x, y = result.stdout.decode("utf8", "replace").split(",")
            return int(x.strip()), int(y.strip())
        except ValueError:
            return 0, 0

    def left_click(self, x, y):
        self.check_failsafe()
        self.move(x, y)
        sleep(self.CLICK_DELAY)
        self.__ydotool(["click", YDOTOOL_LEFT_CLICK])

    def press(self, key):
        code = KEY_CODES.get(key.lower())

        if code is None:
            raise Exception(f"the key {key!r} is not known, add it in KEY_CODES")

        self.__ydotool(["key", f"{code}:1", f"{code}:0"])

    def hotkey(self, *keys):
        codes = []

        for key in keys:
            code = KEY_CODES.get(key.lower())

            if code is None:
                raise Exception(f"the key {key!r} is not known, add it in KEY_CODES")

            codes.append(code)

        # Pressed in order and released the other way around, the way a hand does it.
        pressed = [f"{code}:1" for code in codes]
        released = [f"{code}:0" for code in reversed(codes)]

        self.__ydotool(["key"] + pressed + released)


def ydotool_socket():
    """
    Look for the socket of the ydotoold daemon, ydotool talks to it and only looks at
    one place by itself.

    :return: Path of the socket, None when the default one is fine.
    """
    if os.environ.get("YDOTOOL_SOCKET"):
        return os.environ["YDOTOOL_SOCKET"]

    candidates = [
        f"/run/user/{os.getuid()}/.ydotool_socket",
        "/run/.ydotool_socket",
        "/tmp/.ydotool_socket",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def availability():
    """
    Tell whether the mouse and the keyboard can be driven at all, and what is missing
    when they cannot.

    :return: (ok, detail).
    """
    if system() != "Linux" or not is_wayland():
        try:
            import pyautogui  # noqa: F401
        except Exception as e:
            return False, f"pyautogui cannot be loaded: {e}"

        return True, "pyautogui"

    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") is None:
        return False, ("only Hyprland is supported on Wayland for now, because the cursor is "
                       "moved through it, log in an X11 session instead")

    if shutil.which("ydotool") is None:
        return False, "ydotool is missing, install it with: sudo pacman -S ydotool"

    socket = ydotool_socket()

    if socket is None:
        return False, ("the ydotoold daemon is not running, start it with: "
                       "systemctl --user enable --now ydotoold")

    if not os.access(socket, os.W_OK):
        return False, (f"{socket} cannot be written to, add yourself to the input group with: "
                       f"sudo usermod -aG input $USER, then log out and back in")

    return True, f"hyprctl and ydotool, through {socket}"


def create():
    """
    Build the way to drive the mouse and the keyboard for this session.

    :return: Pointer.
    """
    ok, detail = availability()

    if not ok:
        raise Exception(f"the mouse cannot be driven: {detail}")

    if system() == "Linux" and is_wayland():
        return WaylandPointer()

    return PyAutoGUIPointer()
