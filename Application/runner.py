import queue
import threading
from contextlib import redirect_stdout
from time import sleep

from Application import paths
from Application.Albion import resources
from Application.game import DEFAULT_WINDOW_NAME


def game_window(window_name=DEFAULT_WINDOW_NAME):
    """
    Look for the game window.

    :param window_name: Title of the game window.
    :return: ScreenInformation of the window, None when the game is not running.
    """
    try:
        from Application.Capture.Factory import CaptureFactory

        return CaptureFactory(window_name).capture.window
    except Exception:
        return None


def checks(window_name=DEFAULT_WINDOW_NAME):
    """
    Check everything the bot needs before it can start.

    :param window_name: Title of the game window.
    :return: List of (name, ok, detail).
    """
    from Application.Interaction import pointer

    window = game_window(window_name)
    mouse_ok, mouse_detail = pointer.availability()

    return [
        ("Albion Online", window is not None,
         str(window) if window is not None else f"no window named {window_name}, start the game"),
        ("Mouse and keyboard", mouse_ok, mouse_detail),
        ("Model", paths.MODEL.exists(),
         str(paths.MODEL) if paths.MODEL.exists() else f"put the trained weights in {paths.MODEL}"),
        ("Yolov5", paths.YOLOV5.joinpath("hubconf.py").exists(),
         str(paths.YOLOV5) if paths.YOLOV5.joinpath("hubconf.py").exists()
         else "run: git submodule update --init --recursive"),
        ("Gathering bar", paths.RESOURCE_BAR.exists(),
         str(paths.RESOURCE_BAR) if paths.RESOURCE_BAR.exists()
         else "press Calibrate while the character gathers a resource"),
    ]


class Writer:
    """Sends what the bot prints to the interface."""

    def __init__(self, messages):
        self.messages = messages

    def write(self, text):
        text = text.strip()

        if text:
            self.messages.put(("log", text))

    def flush(self):
        pass


class BotRunner:
    """
    Runs the bot in its own thread, so the interface stays alive, and reports what
    happens through a queue of (kind, text) messages.
    """

    def __init__(self):
        self.messages = queue.Queue()
        self.thread = None
        self.interaction = None
        self.stopping = False

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def log(self, text):
        self.messages.put(("log", text))

    def start(self, targets, confidence=0.5, window_name=DEFAULT_WINDOW_NAME, preview=False):
        """
        Start gathering.

        :param targets: Resource names to gather.
        :param confidence: Lowest confidence of a detection to keep.
        :param window_name: Title of the game window.
        :param preview: Flag to show the window drawing the detections.
        :return: True when the bot has been started.
        """
        if self.is_running():
            return False

        if len(targets) == 0:
            self.log("Pick at least one resource to gather")
            self.messages.put(("stopped", ""))
            return False

        self.stopping = False
        self.interaction = None
        self.thread = threading.Thread(target=self.__gather, daemon=True,
                                       args=(list(targets), confidence, window_name, preview))
        self.thread.start()

        return True

    def stop(self):
        """
        Ask the bot to stop, it ends once the resource being gathered is over.
        """
        if not self.is_running():
            return

        self.stopping = True

        if self.interaction is not None:
            self.interaction.stop_requested = True

        self.log("Stopping...")

    def calibrate(self, window_name=DEFAULT_WINDOW_NAME, delay=10):
        """
        Save the picture of the gathering bar, the character has to be gathering when
        the delay is over.

        :param window_name: Title of the game window.
        :param delay: Time in second before the screenshot is taken.
        :return: True when the calibration has been started.
        """
        if self.is_running():
            return False

        self.stopping = False
        self.thread = threading.Thread(target=self.__calibrate, daemon=True, args=(window_name, delay))
        self.thread.start()

        return True

    def __gather(self, targets, confidence, window_name, preview):
        try:
            if game_window(window_name) is None:
                raise Exception(f"No window named {window_name}, start Albion Online first")

            self.log("Loading the model, it takes a few seconds...")

            # Imported here and not at the top of the file, because loading torch is
            # slow and the interface has to show up right away.
            from Application.Albion.detection import AlbionDetection
            from Application.Interaction.interaction import Interaction

            model = AlbionDetection(debug=True, preview=preview, confidence=confidence,
                                    window_name=window_name, targets=targets)
            interaction = Interaction(model)

            self.interaction = interaction

            # The stop button may have been pressed while the model was loading.
            if self.stopping:
                interaction.stop_requested = True

            self.log(f"Gathering {', '.join(targets)}, move the mouse to the top left corner to stop")

            with redirect_stdout(Writer(self.messages)):
                interaction.loop()

        except Exception as e:
            self.messages.put(("error", str(e)))
        finally:
            self.interaction = None
            self.messages.put(("stopped", ""))

    def __calibrate(self, window_name, delay):
        try:
            from Application.Albion import bar
            from Application.Capture.Factory import CaptureFactory

            if game_window(window_name) is None:
                raise Exception(f"No window named {window_name}, start Albion Online first")

            capture = CaptureFactory(window_name).capture

            self.log("Start gathering a resource by hand, now")

            for remaining in range(delay, 0, -1):
                if self.stopping:
                    self.log("Calibration cancelled")
                    return

                self.log(f"Screenshot in {remaining}s...")
                sleep(1)

            template = bar.save(capture)

            self.log(f"Saved a {template.shape[1]}x{template.shape[0]} picture of the bar "
                     f"in {paths.RESOURCE_BAR}")

            if not bar.usable(template):
                self.log("That picture is flat, the character was most likely not gathering when it "
                         "was taken. Start gathering first, then calibrate again")
                return

            # The character is still gathering, so the bar is still there and matching it
            # right away says whether the calibration is worth anything.
            found = bar.confidence(capture, template)

            self.log(f"The bar is recognized at {found:.2f}, it has to stay over {bar.CONFIDENCE} "
                     f"while gathering and drop under it once the node is empty")

            if found < bar.CONFIDENCE:
                self.log("That is too low, calibrate again while the bar is really on screen")

        except Exception as e:
            self.messages.put(("error", str(e)))
        finally:
            self.messages.put(("stopped", ""))


def resource_names():
    """
    Name of every resource that can be gathered.

    :return: List of names.
    """
    return [str(profile) for profile in resources.PROFILES]
