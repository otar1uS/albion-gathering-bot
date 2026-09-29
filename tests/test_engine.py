import numpy as np
import pytest

from albion_bot.bot.engine import Bot
from albion_bot.config import Settings
from albion_bot.vision import resources
from albion_bot.vision.gather_bar import GatherBar
from tests import fake_game
from tests.fake_game import FakeGame, Node


def calibrated_bar(game):
    game.bar_on = True
    bar = GatherBar.calibrate(game.grab(), fake_game.BAR_BOX)
    game.bar_on = False
    return bar


def make_bot(game, until, logs, **settings):
    settings = Settings(targets=["tree"], character=list(fake_game.CHARACTER), **settings)
    return Bot(game, game, game, calibrated_bar(game), settings, log=logs.append,
               should_stop=lambda: game.time > until, clock=game.clock, sleep=game.sleep)


def test_gathers_every_node_once_and_never_goes_back_to_a_stump(world):
    nodes = [Node(2150, 1440, 3, ident=1), Node(1800, 1580, 4, ident=2), Node(2060, 1650, 2, ident=3)]
    game = FakeGame(world, (2000, 1500), nodes)
    logs = []
    bot = make_bot(game, 150, logs)

    bot.run()

    assert all(not node.alive for node in nodes), logs
    assert bot.stats.gathered == 3, logs
    # The model sees the stumps as trees, only the memory keeps the bot off them.
    assert game.bad_clicks == 0, logs
    assert bot.stats.failed == 0, logs


def test_waits_through_the_gap_between_two_charges(world):
    node = Node(2100, 1500, 5, ident=4)
    game = FakeGame(world, (2000, 1500), [node])
    logs = []
    bot = make_bot(game, 60, logs)

    bot.run()

    assert not node.alive
    assert len([c for c in game.clicks if abs(c[0] - node.x) < 25]) == 1, "clicked the node again mid gathering"


def test_skips_nodes_guarded_by_a_monster(world):
    tree = Node(2150, 1450, 2, ident=5)
    monster = Node(2160, 1470, 99, profile=resources.MONSTER, ident=6)
    game = FakeGame(world, (2000, 1500), [tree, monster])
    logs = []

    make_bot(game, 20, logs).run()
    assert tree.alive and not game.clicks

    game = FakeGame(world, (2000, 1500), [tree, monster])
    make_bot(game, 20, logs, avoid_monsters=False).run()
    assert not tree.alive


def test_stops_when_nodes_end_right_away(world):
    # One charge each: every node is over at once, like with full bags.
    nodes = [Node(2000 + dx, 1500 + dy, 1, ident=i) for i, (dx, dy) in
             enumerate([(120, -40), (-150, 60), (80, 140), (-60, -120)])]
    game = FakeGame(world, (2000, 1500), nodes)
    logs = []
    bot = make_bot(game, 300, logs)
    bot.QUICK_GATHER = 1.6

    bot.run()

    assert any("bags" in line for line in logs), logs
    assert bot.stats.gathered == bot.QUICK_LIMIT


def test_time_limit(world):
    game = FakeGame(world, (2000, 1500), [])
    logs = []
    make_bot(game, 10_000, logs, max_minutes=0.5).run()

    assert 30 <= game.time <= 32
    assert any("minutes are over" in line for line in logs)


@pytest.mark.parametrize("seed", range(10))
def test_random_layouts(world, seed):
    """Every node in sight gathered, no stump ever clicked, whatever the layout."""
    random = np.random.default_rng(seed)
    nodes = []

    while len(nodes) < int(random.integers(2, 6)) or not nodes:
        dx, dy = int(random.integers(-420, 420)), int(random.integers(-180, 280))

        if abs(dx) < 40 and abs(dy) < 40 or any(abs(2000 + dx - n.x) < 60 and abs(1500 + dy - n.y) < 60
                                                for n in nodes):
            continue

        nodes.append(Node(2000 + dx, 1500 + dy, int(random.integers(2, 6)), ident=len(nodes) + 1))

    game = FakeGame(world, (2000, 1500), nodes)
    logs = []
    make_bot(game, 40 + 25 * len(nodes), logs).run()

    left, top = game.origin()
    still_there = [n for n in nodes if n.alive and 20 <= n.x - left <= fake_game.W - 20
                   and 40 <= n.y - top <= fake_game.H - 10]

    assert not still_there, logs
    assert game.bad_clicks == 0, logs
