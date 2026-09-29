import pytest

from albion_bot.vision import resources


@pytest.mark.parametrize("label, expected", [
    ("rough log", "tree"), ("rought log", "tree"), ("wood4", "tree"), ("T5 birch", "tree"),
    ("rough stone", "stone"), ("rough", "stone"), ("travertine", "stone"), ("rock", "stone"), ("lime", "stone"),
    ("iron ore", "ore"), ("CopperOre", "ore"), ("titan", "ore"), ("ore2", "ore"),
    ("fiber3", "fiber"), ("cotton", "fiber"),
    ("rough hide", "hide"), ("wolf", "hide"),
    ("monster", "monster"), ("Enemys", "monster"), ("bandit", "monster"),
    ("albion-gathering", "unknown"), ("player", "unknown"),
])
def test_resolve(label, expected):
    assert resources.resolve(label).name == expected


def test_classes():
    assert resources.CLASSES == ("tree", "stone", "ore", "fiber", "hide", "monster")
    assert resources.MONSTER not in resources.GATHERABLE


def test_targets_must_be_gatherable():
    assert resources.resolve_targets(["tree", "ore"]) == (resources.TREE, resources.ORE)

    with pytest.raises(ValueError):
        resources.resolve_targets(["monster"])

    with pytest.raises(ValueError):
        resources.resolve_targets(["gold"])
