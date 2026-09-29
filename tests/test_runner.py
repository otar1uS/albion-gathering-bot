import queue
from pathlib import Path
from time import monotonic, sleep

from albion_bot.config import Settings
from albion_bot.platform import capture, input
from albion_bot.ui import runner as runner_module
from albion_bot.ui.runner import Runner
from albion_bot.vision import detector as detector_module
from albion_bot.vision.gather_bar import GatherBar
from tests import fake_game
from tests.fake_game import FakeGame, Node


class FakeDetector:
    def __init__(self, game):
        self.game = game
        self.path = Path("best.pt")
        self.device = "cpu"

    def describe(self):
        return ["class 0 'tree' -> tree"]

    def detect(self, frame):
        return self.game.detect(frame)

    def knows(self, profile):
        return True


def wait_done(runner, timeout=20):
    messages = []
    end = monotonic() + timeout

    while monotonic() < end:
        try:
            kind, payload = runner.messages.get(timeout=0.1)
        except queue.Empty:
            continue

        messages.append((kind, payload))

        if kind == "done":
            return messages

    raise AssertionError(f"never done: {messages}")


def patch_game(monkeypatch, game):
    monkeypatch.setattr(capture, "GameWindow", lambda name: game)
    monkeypatch.setattr(input, "create", lambda: game)
    monkeypatch.setattr(detector_module, "Detector", lambda confidence: FakeDetector(game))


def test_gather_reports_and_stops(monkeypatch, world):
    game = FakeGame(world, (2000, 1500), [Node(2100, 1480, 50, ident=1)])
    patch_game(monkeypatch, game)
    game.bar_on = True
    GatherBar.calibrate(game.grab(), fake_game.BAR_BOX).save()
    game.bar_on = False

    runner = Runner()
    assert runner.start(Settings(targets=["tree"], character=list(fake_game.CHARACTER)))
    assert not runner.start(Settings()), "a second task must wait for the first"
    sleep(1.5)
    runner.stop()
    messages = wait_done(runner)

    kinds = [kind for kind, _ in messages]
    logs = [payload for kind, payload in messages if kind == "log"]
    assert "error" not in kinds, messages
    assert any("on cpu" in line for line in logs)
    assert "stats" in kinds
    assert any("Stopped: asked to" in line for line in logs)


def test_gather_without_calibration_says_so(monkeypatch, world):
    patch_game(monkeypatch, FakeGame(world, (2000, 1500), []))
    runner = Runner()
    runner.start(Settings())
    errors = [payload for kind, payload in wait_done(runner) if kind == "error"]
    assert errors and "not calibrated" in errors[0]


def test_snapshot(monkeypatch, world):
    patch_game(monkeypatch, FakeGame(world, (2000, 1500), []))
    runner = Runner()
    runner.snapshot(Settings(), "minimap")
    snapshots = [payload for kind, payload in wait_done(runner) if kind == "snapshot"]
    assert snapshots[0][0] == "minimap" and snapshots[0][1].shape == (fake_game.H, fake_game.W, 3)


def test_checks_list_everything():
    names = [name for name, _, _ in runner_module.checks(Settings(window_name="no such window 123"))]
    assert names == ["Albion Online", "Mouse and keyboard", "Ultralytics", "Model", "Gathering bar", "Routes"]
