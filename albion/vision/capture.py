"""
Photographing the game.

The important decision here is that the window is asked for its own picture rather than
the screen being read where the window happens to be. Reading the screen means reading
whatever is on top of the game, and during development the model was handed a terminal, a
notification and the bot's own preview window in turn, and dutifully looked for trees in
each. Measured with the game deliberately covered by a console: 2.4% of the frame came
back black asking the window, against 41.5% reading the screen.

There is still a reason to raise the game: clicks land on whatever is under the cursor on
the real screen. So the bot keeps the window in front to act, but no longer depends on it
to see, which is the difference between a stray notification costing a frame and costing
the whole run.
"""

import ctypes
from platform import system

import numpy as np

from .. import logs
from ..geometry import Rect

log = logs.get("vision.capture")

# Undocumented flag asking PrintWindow for what the window has drawn rather than for the
# part of it the desktop is showing. It is the only value that returns anything but a
# black rectangle for a window rendering through the GPU, which Albion does.
PW_RENDERFULLCONTENT = 0x00000002


class WindowNotFound(Exception):
    """The game is not running, or is not called what the settings say."""


class Capture:
    """What the rest of the bot needs from a screenshot."""

    def __init__(self, config):
        self.config = config
        self.rect = None

    def refresh(self):
        """
        Read where the window is again.

        Every click is worked out from this, so a window moved or resized mid run would
        otherwise send every click to where the game used to be. Asked for on every pass
        rather than once at startup, which costs one call for a rectangle.

        :return: The window rectangle.
        """
        raise NotImplementedError

    def focus(self):
        """
        Bring the game in front, so clicks reach it.

        :return: True when it is in front.
        """
        return False

    def grab(self):
        """
        Photograph the game.

        :return: BGRA image of the game, without the window decoration.
        """
        raise NotImplementedError


class WindowsCapture(Capture):
    """Windows, through the window's own device context."""

    def __init__(self, config):
        super().__init__(config)

        import win32con
        import win32gui
        import win32ui

        self.win32con, self.win32gui, self.win32ui = win32con, win32gui, win32ui

        self.hwnd = self.__find()
        self.refresh()

    def __find(self):
        """
        Look for the game window by part of its title.

        :return: Window handle.
        """
        wanted = self.config.window.title.lower()
        found = []

        def visit(hwnd, _):
            if not self.win32gui.IsWindowVisible(hwnd):
                return

            title = self.win32gui.GetWindowText(hwnd)

            # The bot's own windows carry the game's name in their title and would
            # otherwise be photographed instead of it.
            if wanted in title.lower() and "gathering bot" not in title.lower():
                found.append(hwnd)

        self.win32gui.EnumWindows(visit, None)

        if not found:
            raise WindowNotFound(
                f"No window with {self.config.window.title!r} in its title. Start Albion "
                f"Online first, or correct the title in the settings.")

        return found[0]

    def refresh(self):
        left, top, right, bottom = self.win32gui.GetWindowRect(self.hwnd)
        decoration = self.config.window.decoration_height

        self.rect = Rect(left=left, top=top + decoration,
                         width=right - left, height=bottom - top - decoration)

        return self.rect

    def focus(self):
        gui, con = self.win32gui, self.win32con

        try:
            if gui.IsIconic(self.hwnd):
                gui.ShowWindow(self.hwnd, con.SW_RESTORE)

            front = gui.GetForegroundWindow()

            if front == self.hwnd:
                return True

            # Windows only lets the process already in front hand the foreground over.
            # Attaching to the thread owning it makes this process count as that one for
            # the length of the call, which is how a program raises a window it does not
            # own.
            import win32api
            import win32process

            ours = win32api.GetCurrentThreadId()
            theirs = win32process.GetWindowThreadProcessId(front)[0]
            attached = False

            try:
                attached = bool(win32process.AttachThreadInput(theirs, ours, True))
                gui.BringWindowToTop(self.hwnd)
                gui.SetForegroundWindow(self.hwnd)
            finally:
                if attached:
                    win32process.AttachThreadInput(theirs, ours, False)

            return gui.GetForegroundWindow() == self.hwnd
        except Exception as error:
            # Refused happens against a window running as administrator and on a locked
            # screen. A bot that stopped because it could not raise a window it can
            # already see would be worse than one that carries on.
            log.debug("could not raise the game: %s", error)
            return False

    def grab(self):
        if self.config.window.capture_window_content:
            picture = self.__print_window()

            if picture is not None:
                return picture

            log.warning("the window would not photograph itself, reading the screen "
                        "instead: anything covering the game will be seen as the game")

        return self.__screen()

    def __print_window(self):
        """
        Ask the window for its own picture.

        :return: BGRA image of the game, None when the call failed.
        """
        left, top, right, bottom = self.win32gui.GetWindowRect(self.hwnd)
        width, height = right - left, bottom - top

        if width <= 0 or height <= 0:
            return None

        window_dc = source = memory = bitmap = None

        try:
            window_dc = self.win32gui.GetWindowDC(self.hwnd)
            source = self.win32ui.CreateDCFromHandle(window_dc)
            memory = source.CreateCompatibleDC()
            bitmap = self.win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(source, width, height)
            memory.SelectObject(bitmap)

            if not ctypes.windll.user32.PrintWindow(self.hwnd, memory.GetSafeHdc(),
                                                    PW_RENDERFULLCONTENT):
                return None

            raw = bitmap.GetBitmapBits(True)
            picture = np.frombuffer(raw, dtype="uint8").reshape((height, width, 4))

            # Cut from the rectangle just measured rather than from one remembered at
            # startup: the window can be moved while the bot runs, and asking for rows
            # that are no longer there hands back an empty picture and stops everything.
            game = picture[self.config.window.decoration_height:, :]

            return np.ascontiguousarray(game) if game.size else None
        except Exception as error:
            log.debug("PrintWindow failed: %s", error)
            return None
        finally:
            if memory is not None:
                memory.DeleteDC()
            if source is not None:
                source.DeleteDC()
            if window_dc is not None:
                self.win32gui.ReleaseDC(self.hwnd, window_dc)
            if bitmap is not None:
                self.win32gui.DeleteObject(bitmap.GetHandle())

    def __screen(self):
        """
        Read the rectangle of the screen the window sits in.

        :return: BGRA image.
        """
        from mss import mss

        with mss() as sct:
            return np.array(sct.grab({"top": self.rect.top, "left": self.rect.left,
                                      "width": self.rect.width,
                                      "height": self.rect.height}))


class ScreenCapture(Capture):
    """
    Anything that is not Windows, by reading the screen.

    Neither MacOS nor an X11 session offers the equivalent of PrintWindow for a window
    drawing through the GPU, so the game has to be the window in front here.
    """

    def __init__(self, config):
        super().__init__(config)
        self.refresh()

    def refresh(self):
        from mss import mss

        with mss() as sct:
            monitor = sct.monitors[1]

        self.rect = Rect(left=monitor["left"], top=monitor["top"],
                         width=monitor["width"], height=monitor["height"])

        return self.rect

    def grab(self):
        from mss import mss

        with mss() as sct:
            return np.array(sct.grab({"top": self.rect.top, "left": self.rect.left,
                                      "width": self.rect.width,
                                      "height": self.rect.height}))


def build(config):
    """
    The capture for this machine.

    :param config: Config.
    :return: Capture.
    """
    if system() == "Windows":
        return WindowsCapture(config)

    log.warning("not on Windows, so the screen is read rather than the window: keep the "
                "game in front and uncovered")

    return ScreenCapture(config)
