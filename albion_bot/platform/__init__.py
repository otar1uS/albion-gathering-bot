"""
What differs between Windows, Linux and macOS: finding the game window, grabbing it, and
driving the mouse and the keyboard.
"""

import os
import sys

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

# Only a part of the title has to match. The Windows client is "Albion Online Client",
# the Linux one is not, and the launcher sharing the name is filtered out on its own.
DEFAULT_WINDOW_NAME = "Albion Online Client" if IS_WINDOWS else "Albion"

# Windows of the bot itself, never the game even though their title holds "Albion".
UI_WINDOW_TITLE = "Albion gathering bot"
OWN_WINDOW_TITLES = (UI_WINDOW_TITLE, "Albion gathering bot - preview", "Albion gathering bot - pick")


def is_wayland():
    return IS_LINUX and (os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY")))


def enable_dpi_awareness():
    """
    Make Windows hand over real pixels. Without it, on a screen scaled to 125% or 150%,
    the window position, the screenshot and the mouse each use a different scale and every
    click lands next to its target. Has to run before any window is created.
    """
    if not IS_WINDOWS:
        return

    import ctypes

    try:
        # Per monitor v2, Windows 10 1703 and later.
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
