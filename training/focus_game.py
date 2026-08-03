"""Brings Albion in front, so a capture holds the game and not what covers it."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import win32com.client
import win32con
import win32gui

from Application import game


def focus(title=game.DEFAULT_WINDOW_NAME):
    hwnd = win32gui.FindWindow(None, title)

    if hwnd == 0:
        raise Exception(f"no window named {title}")

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # Windows refuses SetForegroundWindow to a process that is not already in front.
    # Sending a key through WScript.Shell first makes the call allowed, it is the
    # trick every window manager helper on Windows ends up using.
    win32com.client.Dispatch("WScript.Shell").SendKeys("%")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.5)

    return hwnd


if __name__ == "__main__":
    hwnd = focus()
    front = win32gui.GetForegroundWindow()
    print(f"albion hwnd {hwnd}, foreground now {front} -> {'OK' if front == hwnd else 'FAILED'}")
    print("rect", win32gui.GetWindowRect(hwnd))
