"""
Windows, through the Win32 API directly, nothing to install.
"""

import ctypes
import os
from ctypes import wintypes

from albion_bot.geometry import Rect

user32 = ctypes.WinDLL("user32", use_last_error=True)

_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

SW_RESTORE = 9
VK_F12 = 0x7B


class NativeWindow:
    def __init__(self, handle, title, pid):
        self.handle = handle
        self.title = title
        self.pid = pid

    def __str__(self):
        return repr(self.title)


def _title(handle):
    length = user32.GetWindowTextLengthW(handle)

    if length == 0:
        return ""

    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, length + 1)
    return buffer.value


def windows():
    """
    :return: Every visible window with a title, the ones of this process left out.
    """
    found = []
    own = os.getpid()

    def callback(handle, _):
        if user32.IsWindowVisible(handle):
            title = _title(handle)

            if title:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))

                if pid.value != own:
                    found.append(NativeWindow(handle, title, pid.value))

        return True

    user32.EnumWindows(_EnumWindowsProc(callback), 0)
    return found


def rect(window):
    """
    :return: Client area of the window on the screen, what the game draws in without the
             title bar and the borders, None when the window is gone or minimized.
    """
    if not user32.IsWindow(window.handle) or user32.IsIconic(window.handle):
        return None

    client = wintypes.RECT()

    if not user32.GetClientRect(window.handle, ctypes.byref(client)):
        return None

    origin = wintypes.POINT(0, 0)

    if not user32.ClientToScreen(window.handle, ctypes.byref(origin)):
        return None

    width, height = client.right - client.left, client.bottom - client.top

    if width <= 0 or height <= 0:
        return None

    return Rect(origin.x, origin.y, width, height)


def is_minimized(window):
    return bool(user32.IsIconic(window.handle))


def focus(window):
    """
    Bring the game in front, the clicks go to whatever window is under the cursor.

    :return: True when the game is in front.
    """
    if user32.IsIconic(window.handle):
        user32.ShowWindow(window.handle, SW_RESTORE)

    user32.SetForegroundWindow(window.handle)
    return user32.GetForegroundWindow() == window.handle


def stop_key_pressed():
    """
    :return: True while F12 is held, wherever the focus is.
    """
    return bool(user32.GetAsyncKeyState(VK_F12) & 0x8000)
