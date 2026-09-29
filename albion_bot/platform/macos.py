"""
macOS, through Quartz. Not tested for a while, Windows and Linux are the ones in use.
"""

import os

import Quartz

from albion_bot.geometry import Rect


class NativeWindow:
    def __init__(self, handle, title, rect, pid):
        self.handle = handle
        self.title = title
        self.rect = rect
        self.pid = pid

    def __str__(self):
        return repr(self.title)


def windows():
    found = []
    own = os.getpid()
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements

    for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID):
        title = info.get("kCGWindowName") or info.get("kCGWindowOwnerName") or ""
        bounds = info.get("kCGWindowBounds") or {}
        pid = info.get("kCGWindowOwnerPID")

        if not title or pid == own or bounds.get("Width", 0) <= 1:
            continue

        found.append(NativeWindow(info["kCGWindowNumber"], title,
                                  Rect(int(bounds["X"]), int(bounds["Y"]), int(bounds["Width"]),
                                       int(bounds["Height"])), pid))

    return found


def rect(window):
    return window.rect


def focus(window):
    return True


def stop_key_pressed():
    return False
