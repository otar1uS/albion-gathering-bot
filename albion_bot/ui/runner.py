"""
Runs whatever touches the game in a thread of its own, so the interface stays alive, and
reports through a queue of (kind, payload) messages, tkinter can only be touched from the
thread that created it.

    log        text for the log
    error      text of what went wrong
    stats      dict of albion_bot.bot.engine.Stats
    preview    BGR frame of what the model sees
    snapshot   (purpose, BGR frame) for a picker of the interface
    matrix     minimap matrix measured by calibrate_walking
    routes     name of a route just recorded
    done       name of the task that ended
"""

import queue
import threading
import traceback
from dataclasses import replace
from time import monotonic, sleep

from albion_bot.platform import capture


class Runner:

    def __init__(self):
        self.messages = queue.Queue()
        self.thread = None
        self.stopping = False

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def log(self, text):
        self.messages.put(("log", text))

    def should_stop(self):
        return self.stopping or capture.stop_key_pressed()

    def stop(self):
        if self.busy():
            self.stopping = True
            self.log("Stopping...")

    def _launch(self, name, task, *arguments):
        if self.busy():
            return False

        self.stopping = False

        def body():
            try:
                task(*arguments)
            except Exception as e:
                self.messages.put(("error", str(e) or e.__class__.__name__))
                traceback.print_exc()
            finally:
                self.messages.put(("done", name))

        self.thread = threading.Thread(target=body, name=name, daemon=True)
        self.thread.start()
        return True

    def _countdown(self, seconds, text):
        for remaining in range(seconds, 0, -1):
            if self.should_stop():
                raise RuntimeError("cancelled")

            self.log(f"{text} in {remaining}s...")
            sleep(1)

    # ---------------------------------------------------------------- tasks

    def start(self, settings, preview=False):
        return self._launch("gather", self._gather, replace(settings), preview)

    def snapshot(self, settings, purpose, delay=0, text="Screenshot"):
        """
        Grab the game for a picker, after a countdown when the game has to be set up first.
        """
        return self._launch("snapshot", self._snapshot, settings.window_name, purpose, delay, text)

    def record(self, settings, name):
        return self._launch("record", self._record, settings.window_name, list(settings.minimap), name)

    def test_bar(self, settings, seconds=30):
        return self._launch("test bar", self._test_bar, settings.window_name, settings.bar_threshold, seconds)

    def calibrate_walking(self, settings):
        return self._launch("calibrate walking", self._calibrate_walking, replace(settings))

    # ---------------------------------------------------------------- bodies

    def _gather(self, settings, preview):
        from albion_bot.bot.engine import Bot
        from albion_bot.platform import input
        from albion_bot.vision.gather_bar import GatherBar

        window = capture.GameWindow(settings.window_name)
        bar = GatherBar.load(settings.bar_threshold)

        if bar is None:
            raise RuntimeError("the gathering bar is not calibrated, see the Setup tab")

        navigator = None

        if settings.route:
            from albion_bot.navigation.route import Navigator, Route

            navigator = Navigator(Route.load(settings.route), settings.minimap, settings.minimap_matrix,
                                  settings.route_mode)

        pointer = input.create()
        self.log("Loading the model...")

        from albion_bot.vision.detector import Detector

        detector = Detector(confidence=settings.confidence)
        self.log(f"Model {detector.path.name} on {detector.device}")

        for line in detector.describe():
            self.log(f"  {line}")

        if self.should_stop():
            return

        bot = Bot(window, pointer, detector, bar, settings, navigator=navigator, log=self.log,
                  should_stop=self.should_stop,
                  on_stats=lambda stats: self.messages.put(("stats", stats)),
                  on_preview=(lambda frame: self.messages.put(("preview", frame))) if preview else None)
        bot.run()

    def _snapshot(self, window_name, purpose, delay, text):
        window = capture.GameWindow(window_name)
        window.focus()

        if delay:
            self._countdown(delay, text)
        else:
            # Leave the interface the time to get out of the way.
            sleep(0.6)

        self.messages.put(("snapshot", (purpose, window.grab())))

    def _record(self, window_name, region, name):
        from albion_bot.navigation.route import Recorder, Route

        Route.folder(name)
        window = capture.GameWindow(window_name)
        window.focus()
        recorder = Recorder(region)

        self.log(f"Recording {name}: walk the route in the game, press Stop at its end. Keep the same "
                 f"minimap zoom when following it")

        while not self.should_stop():
            message = recorder.feed(window.grab())

            if message:
                self.log(message)

            sleep(0.4)

        recorder.finish(name)
        self.log(f"Route {name} saved with {len(recorder.patches)} waypoints")
        self.messages.put(("routes", name))

    def _test_bar(self, window_name, threshold, seconds):
        from albion_bot.vision.gather_bar import GatherBar

        bar = GatherBar.load(threshold)

        if bar is None:
            raise RuntimeError("calibrate the gathering bar first")

        window = capture.GameWindow(window_name)
        window.focus()
        self.log(f"Showing the bar score for {seconds}s: gather something, it has to go over {threshold} "
                 f"while gathering and stay under it otherwise")
        end = monotonic() + seconds

        while monotonic() < end and not self.should_stop():
            score = bar.score(window.grab())
            self.log(f"  {score:.2f}  {'GATHERING' if score >= threshold else 'idle'}")
            sleep(0.5)

    def _calibrate_walking(self, settings):
        from albion_bot import geometry
        from albion_bot.navigation.route import calibrate
        from albion_bot.platform import input

        window = capture.GameWindow(settings.window_name)
        pointer = input.create()
        window.focus()
        self._countdown(3, "Stand in an open place, the character walks right then down")

        def walk(direction):
            frame = window.grab()
            x, y = geometry.point(frame.shape, settings.character)
            radius = frame.shape[0] * 0.2
            pointer.click(*window.to_screen(x + direction[0] * radius, y + direction[1] * radius, frame.shape))

        matrix = calibrate(window.grab, walk, settings.minimap, lambda: sleep(1.5))
        self.log(f"Walking calibrated: {[[round(v, 3) for v in row] for row in matrix]}")
        self.messages.put(("matrix", matrix))


def checks(settings):
    """
    Everything the bot needs before it can start.

    :return: List of (name, ok, detail).
    """
    from importlib.util import find_spec

    from albion_bot import paths
    from albion_bot.navigation import route
    from albion_bot.platform import input
    from albion_bot.vision import detector

    found = capture.find_native(settings.window_name) if settings.window_name else None
    mouse_ok, mouse_detail = input.availability()
    model = detector.model_path()
    routes = route.names()

    return [
        ("Albion Online", found is not None,
         f"{found[0]} {found[1]}" if found else f"no window named {settings.window_name!r}, start the game"),
        ("Mouse and keyboard", mouse_ok, mouse_detail),
        ("Ultralytics", find_spec("ultralytics") is not None,
         "installed" if find_spec("ultralytics") else "pip install -r requirements.txt"),
        ("Model", model is not None,
         str(model.relative_to(paths.ROOT)) if model else "run tools/dataset.py then tools/train.py, see the README"),
        ("Gathering bar", paths.GATHER_BAR.exists() and paths.GATHER_BAR_INFO.exists(),
         "calibrated" if paths.GATHER_BAR.exists() else "Setup tab, step 1"),
        ("Routes", True, f"{len(routes)} recorded" if routes else "none yet, optional, Routes tab"),
    ]
