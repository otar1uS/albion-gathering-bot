#!/usr/bin/env python
"""
Live display of a training run, to watch in a terminal window while it goes.

Training writes to training_run.log, but that file is not pleasant to tail by hand: the
progress bar redraws itself by printing carriage returns rather than new lines, so the
usual tools show it as one enormous line that never ends. This splits the file the way a
terminal would and reprints only the newest state, which is what a progress bar is meant
to look like.

Above that live line it prints one row per finished epoch, read from results.csv, which
yolov5 appends to as soon as each epoch is scored. mAP50 is the number that matters:
over 0.8 is a model that finds things reliably.

Open a terminal and run:

    C:\\Users\\otopk\\albion-gathering-bot\\.venv-train\\Scripts\\python.exe C:\\Users\\otopk\\albion-gathering-bot\\training\\watch_training.py

Closing it does nothing to the training, it only reads.
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "training_run.log"
RUN = ROOT / "yolov5" / "runs" / "train" / "albion_merged"

# Colours and cursor moves the progress bar leaves in the file. They mean nothing once
# the line is being reprinted from scratch, and some of them would clear the screen.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Enough of the tail to be sure of catching a whole line, without reading a log that has
# grown to megabytes over a night of training.
TAIL_BYTES = 65536


def newest_line():
    """
    The line the progress bar would be showing right now.

    :return: Newest non-empty line of the log, empty string when there is nothing yet.
    """
    if not LOG.exists():
        return ""

    with LOG.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(handle.tell() - TAIL_BYTES, 0))
        blob = handle.read().decode("utf-8", "replace")

    # A carriage return means "draw over what is there", so it separates states of the
    # bar exactly as a newline separates lines.
    for line in reversed(blob.replace("\r", "\n").split("\n")):
        line = ANSI.sub("", line).strip()

        if line:
            return line

    return ""


def epochs():
    """
    Every epoch scored so far.

    :return: List of (epoch, precision, recall, mAP50, mAP50-95) as strings.
    """
    results = RUN / "results.csv"

    if not results.exists():
        return []

    try:
        rows = list(csv.DictReader(results.read_text().splitlines()))
    except Exception:
        # Caught mid-write by the training process, it will be readable a second later.
        return []

    def column(row, wanted):
        for name, value in row.items():
            if name and wanted in name.strip():
                return value.strip()

        return "?"

    return [
        (
            column(row, "epoch"),
            column(row, "precision"),
            column(row, "recall"),
            column(row, "mAP_0.5") if any("mAP_0.5" in (n or "") and "0.95" not in (n or "") for n in row) else "?",
            column(row, "mAP_0.5:0.95"),
        )
        for row in rows
    ]


def main():
    # The progress bar is drawn out of box characters, and a Windows console still
    # defaults to an encoding from before those existed, which otherwise ends the watch
    # with an encoding error the moment the first bar is printed.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"watching {LOG}")
    print("ctrl-c to close this window, the training keeps going\n")
    print(f"{'epoch':>7}  {'precision':>10}  {'recall':>10}  {'mAP50':>10}  {'mAP50-95':>10}")

    shown = 0
    previous = ""

    while True:
        scored = epochs()

        # Reprinting only the new rows keeps the finished epochs scrolling up the window
        # as a history, rather than redrawing the whole table every second.
        for row in scored[shown:]:
            epoch, precision, recall, map50, map95 = row
            print(f"\r{' ' * 100}\r{epoch:>7}  {precision:>10}  {recall:>10}  {map50:>10}  {map95:>10}")

        shown = len(scored)

        line = newest_line()

        if line != previous:
            # Trimmed to a sane width and padded, so a shorter line does not leave the
            # tail of the longer one behind it on screen.
            sys.stdout.write("\r" + line[:110].ljust(112))
            sys.stdout.flush()
            previous = line

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nclosed, training continues")
