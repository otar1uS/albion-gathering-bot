import numpy as np
import pytest

from albion_bot.bot.engine import Bot
from albion_bot.config import Settings
from albion_bot.navigation import route
from albion_bot.vision.gather_bar import GatherBar
from tests import fake_game
from tests.fake_game import FakeGame, Node

SQUARE = [(2000 + i * 8, 1500) for i in range(60)] + [(2480, 1500 + i * 8) for i in range(50)]


def record(game, name="square", path=SQUARE):
    recorder = route.Recorder(fake_game.MINIMAP)

    for x, y in path:
        game.x, game.y = x, y
        recorder.feed(game.grab())

    return recorder.finish(name)


def walk(game, navigator, steps=600):
    """Follow the route the way the bot does, return the waypoints in the order reached."""
    visited = []

    for _ in range(steps):
        step = navigator.step(game.grab())
        assert step.status != "lost"

        if step.status == "arrived":
            visited.append(step.waypoint)
        elif step.status == "walk":
            cx, cy = fake_game.W * fake_game.CHARACTER[0], fake_game.H * fake_game.CHARACTER[1]
            game.click(cx + step.direction[0] * 100, cy + step.direction[1] * 100)

        game.sleep(0.3)

    return visited


def test_record_save_load(world):
    game = FakeGame(world, (2000, 1500), [])
    recorded = record(game)

    assert route.names() == ["square"]
    assert len(route.Route.load("square")) == len(recorded) >= 6

    route.delete("square")
    assert route.names() == []


def test_names_are_checked():
    for name in ("../escape", "", " padded", "a/b"):
        with pytest.raises(ValueError):
            route.Route.folder(name)


def test_back_and_forth(world):
    game = FakeGame(world, (2000, 1500), [])
    recorded = record(game)
    game.x, game.y = 2000, 1500
    navigator = route.Navigator(route.Route.load("square"), fake_game.MINIMAP, [[1, 0], [0, 1]],
                                "back and forth", clock=game.clock)

    visited = walk(game, navigator)
    last = len(recorded) - 1
    end = visited.index(last)

    assert visited[:end + 1] == list(range(visited[0], last + 1))
    assert visited[end:end + last + 1] == list(range(last, -1, -1))


def test_calibrate_finds_how_the_minimap_turns(world):
    game = FakeGame(world, (2000, 1500), [])
    cx, cy = fake_game.W * fake_game.CHARACTER[0], fake_game.H * fake_game.CHARACTER[1]

    matrix = np.array(route.calibrate(game.grab, lambda d: game.click(cx + d[0] * 100, cy + d[1] * 100),
                                      fake_game.MINIMAP, lambda: game.sleep(1.5)))
    matrix /= np.linalg.norm(matrix, axis=0)

    assert matrix == pytest.approx(np.eye(2), abs=0.1)


def test_bot_walks_the_route_and_gathers_on_the_way(world):
    game = FakeGame(world, (2000, 1500), [])
    record(game)

    # Nodes along the route, out of sight from the start.
    nodes = [Node(2470, 1480, 2, ident=1), Node(2500, 1880, 2, ident=2)]
    game = FakeGame(world, (2000, 1500), nodes)
    game.bar_on = True
    bar = GatherBar.calibrate(game.grab(), fake_game.BAR_BOX)
    game.bar_on = False
    assert not any(n for n in game.detect(game.grab()))

    settings = Settings(targets=["tree"], character=list(fake_game.CHARACTER), route="square",
                        minimap=list(fake_game.MINIMAP), route_mode="back and forth")
    navigator = route.Navigator(route.Route.load("square"), settings.minimap, settings.minimap_matrix,
                                settings.route_mode, clock=game.clock)
    logs = []
    bot = Bot(game, game, game, bar, settings, navigator=navigator, log=logs.append,
              should_stop=lambda: game.time > 200, clock=game.clock, sleep=game.sleep)
    bot.run()

    assert all(not n.alive for n in nodes), logs
    assert game.bad_clicks == 0, logs
