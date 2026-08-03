"""
Logging, set up once and shared.

The old bot printed straight to stdout and the interface captured that by replacing
stdout wholesale, which meant nothing could be filtered, nothing carried a timestamp, and
a run left no trace once the window was closed. A long unattended run is exactly the case
where the record afterwards is the only evidence of what went wrong, so it is written to
a file as well as shown.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "albion_bot.log"

_configured = False


class QueueHandler(logging.Handler):
    """
    Hands log records to the interface.

    The bot runs in its own thread so that the window stays alive, and a thread cannot
    draw on it, so the records go through a queue the interface drains on its own timer.
    """

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        try:
            self.queue.put(("log", self.format(record)))
        except Exception:
            # A log handler that raises would take the bot down with it, and losing a
            # line off the interface matters far less than that. The file still has it.
            pass


def setup(level="INFO", queue=None):
    """
    Prepare logging, once per process.

    :param level: Lowest level to keep, as its name.
    :param queue: Queue for the interface to read, None when running from a console.
    :return: The root logger of the bot.
    """
    global _configured

    root = logging.getLogger("albion")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if not _configured:
        plain = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
                                  datefmt="%H:%M:%S")

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(plain)
        root.addHandler(console)

        # Kept across runs on purpose. The interesting failures happen overnight and the
        # question in the morning is what the bot was doing an hour before it stopped.
        to_file = logging.FileHandler(LOG_FILE, encoding="utf-8")
        to_file.setFormatter(plain)
        root.addHandler(to_file)

        _configured = True

    if queue is not None:
        for handler in list(root.handlers):
            if isinstance(handler, QueueHandler):
                root.removeHandler(handler)

        to_ui = QueueHandler(queue)
        to_ui.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(to_ui)

    return root


def get(name):
    """
    A logger for one part of the bot.

    :param name: Short name, "vision.detector" for instance.
    :return: Logger under the bot's root.
    """
    return logging.getLogger(f"albion.{name}")
