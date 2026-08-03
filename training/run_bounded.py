"""
Runs the bot for a fixed time and prints everything it does.

Used to try the bot without leaving it running: the loop is started in a thread and
asked to stop once the time is up, the same way the Stop button of the interface does.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time import sleep, time

from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction
from training.focus_game import focus

SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TARGETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["tree"]
CONFIDENCE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.60

focus()

print(f"targets {TARGETS}, confidence {CONFIDENCE}, running {SECONDS}s")

model = AlbionDetection(debug=True, preview=False, confidence=CONFIDENCE, targets=TARGETS)
interaction = Interaction(model)

print(f"character on screen at {model.character_screen_position()}")
print(f"window {model.window_capture.window}")
print("-" * 60)

thread = threading.Thread(target=interaction.loop, daemon=True)
started = time()
thread.start()

while thread.is_alive() and time() - started < SECONDS:
    sleep(0.5)

interaction.stop_requested = True
thread.join(timeout=20)

print("-" * 60)
print(f"stopped after {time() - started:.0f}s")
