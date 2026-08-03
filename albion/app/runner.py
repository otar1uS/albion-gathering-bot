"""
Running the bot behind an interface.

The bot spends most of its life inside waits, so it cannot share a thread with a window
that has to stay responsive. It gets its own, and everything it has to say comes back
through a queue the interface drains on a timer. Logging is pointed at that queue too, so
the window shows the same record that goes to the file rather than a second, poorer one.
"""

import queue
import threading

from .. import logs
from ..config import Config
from ..game import resources

log = logs.get("app.runner")


def preflight(config):
    """
    Everything that has to be true before the bot can start.

    Checked up front and reported together, because finding out about a missing model
    thirty seconds into loading torch is a worse experience than being told immediately.

    :param config: Config.
    :return: List of (name, ok, detail).
    """
    checks = []

    try:
        from ..vision import capture as capture_module

        window = capture_module.build(config).rect
        checks.append(("Albion Online", True, str(window)))
    except Exception as error:
        checks.append(("Albion Online", False, str(error)))

    checks.append(("Model", config.weights_path.exists(),
                   str(config.weights_path) if config.weights_path.exists()
                   else f"no weights at {config.weights_path}"))

    yolov5 = config.yolov5_path.joinpath("hubconf.py").exists()
    checks.append(("Yolov5", yolov5, str(config.yolov5_path) if yolov5
                   else "run: git submodule update --init --recursive"))

    try:
        from ..control import input as input_module

        checks.append(("Mouse and keyboard", True,
                       type(input_module.build(config)).__name__))
    except Exception as error:
        checks.append(("Mouse and keyboard", False, str(error)))

    return checks


class BotRunner:
    """Starts and stops the bot from the interface."""

    def __init__(self):
        self.messages = queue.Queue()
        self.thread = None
        self.bot = None
        self.stopping = False

        logs.setup(queue=self.messages)

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, config, targets):
        """
        Begin gathering.

        :param config: Config.
        :param targets: Resource names to gather.
        :return: True when the bot was started.
        """
        if self.is_running():
            return False

        if not targets:
            log.error("pick at least one resource to gather")
            self.messages.put(("stopped", ""))
            return False

        self.stopping = False
        self.bot = None
        self.thread = threading.Thread(target=self.__run, daemon=True,
                                       args=(config, list(targets)))
        self.thread.start()

        return True

    def stop(self):
        """Ask the bot to finish the node it is on and stop."""
        if not self.is_running():
            return

        self.stopping = True

        if self.bot is not None:
            self.bot.stop()

        log.info("stopping...")

    def __run(self, config, targets):
        try:
            log.info("loading the model, this takes a few seconds")

            # Imported here rather than at the top, because pulling in torch takes long
            # enough that the window would sit blank if it happened at startup.
            from . import session

            self.bot = session.build(config, targets)

            if self.stopping:
                self.bot.stop()

            log.info("gathering %s. Throw the mouse into the top left corner of the "
                     "screen to stop at any time.", ", ".join(targets))

            self.bot.run()
        except Exception as error:
            log.exception("the bot stopped: %s", error)
            self.messages.put(("error", str(error)))
        finally:
            self.bot = None
            self.messages.put(("stopped", ""))


def resource_names():
    """
    Every resource that can be gathered.

    :return: List of names.
    """
    return resources.names()


def load_config(path=None):
    """
    Read the settings.

    :param path: File to read, the default when left to None.
    :return: Config.
    """
    return Config.load(path)
