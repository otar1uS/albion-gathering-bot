from platform import system

from Application.game import DEFAULT_WINDOW_NAME


class CaptureFactory:
    capture = None

    def __init__(self, window_name=DEFAULT_WINDOW_NAME):

        self.windowName = window_name

        # Both captures are imported here and not at the top of the file, because
        # each one needs a library only available on its own platform.
        if system() == "Windows":
            from .Windows import WindowsCapture
            self.capture = WindowsCapture(window_name=window_name)
        elif system() == "Darwin":
            from .MacOS import MacOSCapture
            self.capture = MacOSCapture(window_name=window_name)
        elif system() == "Linux":
            from .Linux import LinuxCapture
            self.capture = LinuxCapture(window_name=window_name)
        else:
            raise Exception(f"{system()} is not supported, only Windows, MacOS and Linux are")
