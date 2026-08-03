"""
Check whether the game can be photographed without being the window in front.

The bot reads the screen where the window happens to be, so anything drawn over the game
is what the model is handed, and it has already scored trees against a terminal full of
source code. Windows can also ask a window for its own picture, which would make the
whole question go away and let the machine be used while the bot runs. Whether that works
depends on how the game draws itself: a window rendering through the GPU usually answers
with a black rectangle, so this asks all three ways and reports what came back rather
than assuming.

Run it with something covering the game on purpose, that is the case that matters.

    .venv\\Scripts\\python.exe training\\probe_capture.py
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import win32con
import win32gui
import win32ui

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.Albion.detection import AlbionDetection
from Application.game import DEFAULT_WINDOW_NAME

# Undocumented but long standing flag asking PrintWindow for the composited content,
# which is the only one that works for a window drawing through the GPU.
PW_RENDERFULLCONTENT = 0x00000002


def window_shot(hwnd, width, height, printwindow):
    """
    Photograph a window through its own device context.

    :param hwnd: Handle of the window.
    :param width: Width to grab.
    :param height: Height to grab.
    :param printwindow: Use PrintWindow rather than BitBlt.
    :return: BGR image, or None when the call failed.
    """
    window_dc = win32gui.GetWindowDC(hwnd)
    source = win32ui.CreateDCFromHandle(window_dc)
    memory = source.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()

    try:
        bitmap.CreateCompatibleBitmap(source, width, height)
        memory.SelectObject(bitmap)

        if printwindow:
            import ctypes
            ok = ctypes.windll.user32.PrintWindow(hwnd, memory.GetSafeHdc(),
                                                  PW_RENDERFULLCONTENT)
            if not ok:
                return None
        else:
            memory.BitBlt((0, 0), (width, height), source, (0, 0), win32con.SRCCOPY)

        raw = bitmap.GetBitmapBits(True)
        img = np.frombuffer(raw, dtype="uint8").reshape((height, width, 4))

        return np.ascontiguousarray(img[..., :3])
    finally:
        memory.DeleteDC()
        source.DeleteDC()
        win32gui.ReleaseDC(hwnd, window_dc)
        win32gui.DeleteObject(bitmap.GetHandle())


def report(tag, img, out):
    """
    Say whether a picture holds anything, and keep it.

    :param tag: Name of the method that produced it.
    :param img: Picture, or None.
    :param out: Directory to write to.
    """
    if img is None:
        print(f"  {tag:<28} failed outright")
        return

    black = float((img.max(axis=2) < 8).mean())
    print(f"  {tag:<28} mean {img.mean():6.1f}   black pixels {black:5.1%}"
          f"   {'looks empty' if black > 0.9 else 'has content'}")

    cv.imwrite(str(out / f"{tag.replace(' ', '_')}.png"), img)


def main():
    model = AlbionDetection(debug=False, preview=False)
    window = model.window_capture.window
    hwnd = win32gui.FindWindow(None, DEFAULT_WINDOW_NAME)

    out = Path("images/capture_probe")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Window {window}, handle {hwnd}")
    print(f"Foreground right now: {win32gui.GetForegroundWindow() == hwnd}\n")

    report("mss screen region", model.window_capture.screenshot()[..., :3], out)
    report("BitBlt window dc", window_shot(hwnd, window.width, window.height, False), out)
    report("PrintWindow fullcontent", window_shot(hwnd, window.width, window.height, True), out)

    print(f"\nPictures in {out}. Whichever has content while the game is covered is the "
          f"one the capture should use.")


if __name__ == "__main__":
    main()
