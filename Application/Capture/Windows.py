import ctypes
from abc import ABC

import win32api
import win32con
import win32gui
import win32process
import win32ui
from numpy import ascontiguousarray, frombuffer

from . import Capture, ScreenInformation

# Asks PrintWindow for what the window has actually drawn rather than for the part of it
# the desktop happens to be showing. Undocumented, and the only value that returns
# anything but a black rectangle for a window rendering through the GPU, which Albion is.
PW_RENDERFULLCONTENT = 0x00000002


class WindowsCapture(Capture, ABC):

    def __init__(self, window_name=Capture.WINDOWS_NAME):
        super().__init__(window_name)

        self.hwnd = self.__get_window_id()
        self.window = self.get_window_information()

        self.grab_coordinates = {
            "top": self.window.top,
            "left": self.window.left,
            "width": self.window.width,
            "height": self.window.height
        }

    def screenshot(self):
        """
        Photograph the game through its own device context rather than off the screen.

        Grabbing the screen means grabbing whatever is on top of the game, and the model
        has already been handed a terminal, a notification and the bot's own preview
        window that way, each of which it dutifully looked for trees in. Asking the
        window for its picture gives the game and only the game, covered or not, which
        also means the machine can be used while the bot runs. Measured against the same
        moment with the game behind a console: 2.4% of the frame black this way, 41.5%
        off the screen.

        The alpha channel is kept and the title bar is cut off, so what comes out is the
        same shape and the same region as the mss path it replaces.

        :return: BGRA image of the game, without the window decoration.
        """
        picture = self.__print_window()

        return super().screenshot() if picture is None else picture

    def __print_window(self):
        """
        Ask the window for its own picture.

        :return: BGRA image of the whole window, None when the call failed and the
                 screen has to be grabbed instead.
        """
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width, height = right - left, bottom - top

        if width <= 0 or height <= 0:
            return None

        window_dc = source = memory = bitmap = None

        try:
            window_dc = win32gui.GetWindowDC(self.hwnd)
            source = win32ui.CreateDCFromHandle(window_dc)
            memory = source.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(source, width, height)
            memory.SelectObject(bitmap)

            if not ctypes.windll.user32.PrintWindow(self.hwnd, memory.GetSafeHdc(),
                                                    PW_RENDERFULLCONTENT):
                return None

            raw = bitmap.GetBitmapBits(True)
            picture = frombuffer(raw, dtype="uint8").reshape((height, width, 4))

            # The rectangle takes in the window decoration and the game is what is under
            # it. Cut from the rectangle that was just measured rather than from the one
            # remembered at startup: the window can be moved or resized while the bot
            # runs, and subtracting a stale top from a fresh one asks for rows that are
            # not there, which hands back an empty picture and stops the bot.
            game = picture[ScreenInformation.ALBION_HEADER_HEIGHT:, :]

            return ascontiguousarray(game) if game.size else None
        except Exception:
            # Falling back to the screen is worse but still works, and a capture that
            # raises would stop the bot outright.
            return None
        finally:
            if memory is not None:
                memory.DeleteDC()
            if source is not None:
                source.DeleteDC()
            if window_dc is not None:
                win32gui.ReleaseDC(self.hwnd, window_dc)
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())

    def refresh(self):
        """
        Read where the window is again.

        Every position the bot clicks is worked out from this, so a window that gets
        moved or resized while the bot runs sends every click to where the game used to
        be. It is asked for on every pass rather than once at startup, which costs one
        call for a rectangle.
        """
        window = self.get_window_information()

        if window is None:
            return

        self.window = window
        self.grab_coordinates = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
        }

    def focus(self):
        self.refresh()

        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

            front = win32gui.GetForegroundWindow()

            if front != self.hwnd:
                # Windows only lets the process already in front hand the foreground
                # over. Attaching to the thread that owns it makes this process count
                # as that one for the length of the call, which is how a program raises
                # a window it does not own.
                ours = win32api.GetCurrentThreadId()
                theirs = win32process.GetWindowThreadProcessId(front)[0]
                attached = False

                try:
                    attached = bool(win32process.AttachThreadInput(theirs, ours, True))
                    win32gui.BringWindowToTop(self.hwnd)
                    win32gui.SetForegroundWindow(self.hwnd)
                finally:
                    if attached:
                        win32process.AttachThreadInput(theirs, ours, False)

            if win32gui.GetForegroundWindow() == self.hwnd:
                return True

            # Refused anyway, which happens on a locked screen or against a window
            # running as administrator. Lifting it above the others without giving it
            # the keyboard is enough here, the capture only needs it uncovered.
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

            return win32gui.GetForegroundWindow() == self.hwnd
        except Exception:
            # A bot that stops because it could not raise a window it can already see
            # would be worse than one that carries on and looks at whatever is there.
            return False

    def get_window_information(self):
        rect = win32gui.GetWindowRect(self.hwnd)
        return ScreenInformation(
            top=rect[1],
            left=rect[0],
            width=rect[2] - rect[0],
            height=rect[3] - rect[1]
        )

    def __get_window_id(self):
        hwnd = win32gui.FindWindow(None, self.windowName)

        if hwnd == 0:
            raise Exception('could not find window named %s' % self.windowName)

        return hwnd

