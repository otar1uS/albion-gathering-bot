"""
Linux. Hyprland is asked for its windows on Wayland, Xlib on X11, which also covers the
games running under XWayland when the compositor is not Hyprland.

Wayland lets no program read the screen or move the cursor, so the screen is grabbed
with grim and the cursor moved by the compositor, see albion_bot.platform.input.
"""

import json
import os
import shutil
import subprocess

from albion_bot.geometry import Rect


class NativeWindow:
    def __init__(self, title, application, rect, pid=None, visible=True):
        self.title = title
        self.application = application
        self.rect = rect
        self.pid = pid
        # False when the window sits on a workspace that is not shown. Wayland hands over
        # what a monitor displays and nothing else, so such a window cannot be grabbed.
        self.visible = visible

    def __str__(self):
        return f"{self.title!r} ({self.application})"


def run(command):
    """
    :return: Standard output as bytes, None when the command is missing or failed.
    """
    if shutil.which(command[0]) is None:
        return None

    try:
        result = subprocess.run(command, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout if result.returncode == 0 else None


def _hyprland_windows():
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return []

    try:
        clients = json.loads(run(["hyprctl", "clients", "-j"]) or b"[]")
        monitors = json.loads(run(["hyprctl", "monitors", "-j"]) or b"[]")
    except ValueError:
        return []

    shown = {monitor.get("activeWorkspace", {}).get("id") for monitor in monitors}
    found = []

    for client in clients:
        left, top = client.get("at", [0, 0])
        width, height = client.get("size", [0, 0])

        if width <= 0 or height <= 0:
            continue

        found.append(NativeWindow(client.get("title") or "", client.get("class") or "",
                                  Rect(left, top, width, height), client.get("pid"),
                                  client.get("workspace", {}).get("id") in shown))

    return found


def _x11_windows():
    try:
        from Xlib import display as xdisplay
        from Xlib.error import XError
    except ImportError:
        return []

    try:
        display = xdisplay.Display()
    except Exception:
        return []

    found = []

    try:
        root = display.screen().root
        pid_atom = display.intern_atom("_NET_WM_PID")
        # Every window and not only the children of the root, window managers reparent
        # the windows they decorate.
        pending = [root]

        while pending:
            window = pending.pop()

            try:
                pending.extend(window.query_tree().children)
                name = window.get_wm_name() or ""
                classes = window.get_wm_class() or ()
                geometry = window.get_geometry()

                if (not name and not classes) or geometry.width <= 1 or geometry.height <= 1:
                    continue

                position = root.translate_coords(window, 0, 0)
                pid = window.get_full_property(pid_atom, 0)

                found.append(NativeWindow(
                    name if isinstance(name, str) else name.decode("utf8", "replace"),
                    " ".join(classes),
                    Rect(position.x, position.y, geometry.width, geometry.height),
                    pid.value[0] if pid is not None and len(pid.value) > 0 else None,
                ))
            except (XError, UnicodeError):
                continue
    finally:
        display.close()

    return found


def windows():
    found = _hyprland_windows() or _x11_windows()
    own = os.getpid()
    return [window for window in found if window.pid != own]


def rect(window):
    return window.rect


def focus(window):
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") and window.pid:
        return run(["hyprctl", "dispatch", "focuswindow", f"pid:{window.pid}"]) is not None

    return True


def stop_key_pressed():
    # Reading the keyboard of the whole session needs root on Linux, the corner of the
    # screen and the Stop button are used instead.
    return False
