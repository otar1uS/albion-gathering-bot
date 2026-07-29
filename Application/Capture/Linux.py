"""
Window capture on Linux.

Two things change compared to Windows and MacOS.

The window is looked for through the compositor instead of a system API. Hyprland
answers on its own socket, and an X11 session is asked through Xlib, which also
covers the games running under XWayland when no compositor is recognized.

The screenshot is taken with grim on Wayland. mss talks X11 only, and on a wlroots
compositor the X11 root window holds none of what is on screen, so it would grab a
black image. On an X11 session mss is kept, it is faster than spawning a process.
"""

import json
import os
import shutil
import subprocess
from abc import ABC
from time import time

import cv2 as cv
import numpy as np

from Application import game
from . import Capture, ScreenInformation

# Time in second a window position is trusted before being asked again, so moving or
# resizing the game does not send the bot clicking next to it forever.
GEOMETRY_LIFETIME = 2.0

# Compositors draw no title bar around the game, unlike Windows.
HEADER_HEIGHT = 0


class Window:
    """One window of the session, in the logical coordinates of the compositor."""

    def __init__(self, title, application, left, top, width, height, pid=None, visible=True):
        self.title = title
        self.application = application
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.pid = pid
        # False when the window sits on a workspace that is not shown. Wayland hands
        # over what a monitor displays and nothing else, so such a window cannot be
        # grabbed at all, see LinuxCapture.screenshot.
        self.visible = visible

    def matches(self, name):
        name = name.lower()

        return name in self.title.lower() or name in self.application.lower()

    def __str__(self):
        return f"{self.title!r} ({self.application})"


def is_wayland():
    """
    :return: True when the session is Wayland, X11 otherwise.
    """
    return os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def __run(command):
    """
    Run a command and give back its output, without letting a missing tool or a
    failure escape.

    :param command: Command and its arguments.
    :return: Standard output as bytes, None when the command failed.
    """
    if shutil.which(command[0]) is None:
        return None

    try:
        result = subprocess.run(command, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout


def __hyprland_shown_workspaces():
    """
    Workspaces currently displayed, one per monitor.

    :return: Set of workspace ids.
    """
    output = __run(["hyprctl", "monitors", "-j"])

    if output is None:
        return set()

    try:
        monitors = json.loads(output)
    except ValueError:
        return set()

    return {monitor.get("activeWorkspace", {}).get("id") for monitor in monitors}


def __hyprland_windows():
    """
    Ask Hyprland for its windows, XWayland ones included.

    :return: List of Window, empty when Hyprland is not the compositor.
    """
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return []

    output = __run(["hyprctl", "clients", "-j"])

    if output is None:
        return []

    try:
        clients = json.loads(output)
    except ValueError:
        return []

    shown = __hyprland_shown_workspaces()
    windows = []
    for client in clients:
        left, top = client.get("at", [0, 0])
        width, height = client.get("size", [0, 0])

        # A window with no surface, minimized or on no workspace, cannot be grabbed.
        if width <= 0 or height <= 0:
            continue

        windows.append(Window(
            title=client.get("title") or "",
            application=client.get("class") or "",
            left=left,
            top=top,
            width=width,
            height=height,
            pid=client.get("pid"),
            visible=client.get("workspace", {}).get("id") in shown,
        ))

    return windows


def __x11_windows():
    """
    Walk the X11 windows, which covers a plain X session as well as the games running
    under XWayland when the compositor is not recognized.

    :return: List of Window, empty when there is no X display.
    """
    try:
        from Xlib import display as xdisplay
        from Xlib.error import XError
    except ImportError:
        return []

    try:
        display = xdisplay.Display()
    except Exception:
        return []

    windows = []

    try:
        root = display.screen().root
        # Every window of the session, and not only the children of the root, because
        # window managers reparent the windows they decorate.
        pending = [root]

        while pending:
            window = pending.pop()

            try:
                children = window.query_tree().children
                pending.extend(children)

                name = window.get_wm_name() or ""
                classes = window.get_wm_class() or ()
                geometry = window.get_geometry()

                if not name and not classes:
                    continue

                if geometry.width <= 1 or geometry.height <= 1:
                    continue

                # The geometry is relative to the parent, translate_coords gives back
                # the position on the screen.
                position = root.translate_coords(window, 0, 0)

                pid = window.get_full_property(display.intern_atom("_NET_WM_PID"), 0)

                windows.append(Window(
                    title=name if isinstance(name, str) else name.decode("utf8", "replace"),
                    application=" ".join(classes),
                    left=position.x,
                    top=position.y,
                    width=geometry.width,
                    height=geometry.height,
                    pid=pid.value[0] if pid is not None and len(pid.value) > 0 else None,
                ))
            except (XError, UnicodeError):
                continue
    finally:
        display.close()

    return windows


def windows():
    """
    Every window of the session, whatever the compositor.

    The windows of the bot itself are left out, by the process they belong to and by
    their title. The title is looked at as well because the interface and the bot are
    not always the same process, running main.py while the interface is open would find
    the interface otherwise.

    :return: List of Window.
    """
    found = __hyprland_windows()

    if len(found) == 0:
        found = __x11_windows()

    own_pid = os.getpid()

    return [window for window in found
            if window.pid != own_pid and window.title not in game.OWN_WINDOW_TITLES]


def find(window_name):
    """
    Look for a window by name, matching a part of its title or of its class, so
    "albion" is enough to find the game.

    :param window_name: Name to look for.
    :return: Matching Window, None when there is none.
    """
    candidates = [window for window in windows() if window.matches(window_name)]

    # The launcher is called Albion Online as well, and gathering in it would be a
    # short career. It is only left in when it is what was explicitly asked for.
    if "launcher" not in window_name.lower():
        candidates = [window for window in candidates
                      if "launcher" not in window.title.lower()
                      and "launcher" not in window.application.lower()]

    if len(candidates) == 0:
        return None

    # The biggest one, the game fills the screen while the smaller windows around it,
    # a chat or a shop overlay, do not.
    return max(candidates, key=lambda window: window.width * window.height)


class LinuxCapture(Capture, ABC):

    def __init__(self, window_name=Capture.WINDOWS_NAME):
        super().__init__(window_name)

        self.wayland = is_wayland()

        if self.wayland and shutil.which("grim") is None:
            raise Exception(
                "grim is missing and a Wayland session cannot be captured without it, "
                "install it with: sudo pacman -S grim"
            )

        self.window = self.get_window_information()
        self.__grabbed_at = time()

        self.grab_coordinates = {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width,
            "height": self.window.height
        }

    def get_window_information(self):
        window = find(self.windowName)

        if window is None:
            raise Exception(
                f"could not find window named {self.windowName}, open windows are "
                f"{[str(candidate) for candidate in windows()]}"
            )

        if self.wayland and not window.visible:
            # Wayland hands over what a monitor shows, so a window sitting on another
            # workspace cannot be grabbed. Grabbing its rectangle anyway would give back
            # whatever is displayed over it, and the model would hunt for trees in it.
            raise Exception(
                f"the window {window} is on a workspace that is not displayed, "
                f"switch to it and leave it in front while the bot runs"
            )

        return ScreenInformation(
            top=window.top,
            left=window.left,
            width=window.width,
            height=window.height,
            header=HEADER_HEIGHT,
        )

    def __refresh(self):
        """
        Follow the window when it is moved or resized, without asking the compositor on
        every single frame.
        """
        if time() - self.__grabbed_at < GEOMETRY_LIFETIME:
            return

        self.__grabbed_at = time()

        try:
            self.window = self.get_window_information()
        except Exception:
            # The window went away, keep aiming at the last place it was seen and let
            # the detection be the one to fail.
            return

        self.grab_coordinates = {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width,
            "height": self.window.height
        }

    def screenshot(self):
        self.__refresh()

        if not self.wayland:
            return super().screenshot()

        geometry = (f"{self.grab_coordinates['left']},{self.grab_coordinates['top']} "
                    f"{self.grab_coordinates['width']}x{self.grab_coordinates['height']}")

        result = subprocess.run(["grim", "-g", geometry, "-t", "png", "-"],
                                capture_output=True, timeout=5)

        if result.returncode != 0:
            raise Exception(f"grim failed to capture {geometry}: {result.stderr.decode('utf8', 'replace').strip()}")

        img = cv.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv.IMREAD_COLOR)

        if img is None:
            raise Exception(f"grim gave back an image that cannot be read, for {geometry}")

        # mss hands over BGRA, and the rest of the code is written around that, so the
        # alpha channel is put back to keep a single kind of image everywhere.
        return cv.cvtColor(img, cv.COLOR_BGR2BGRA)
