"""
The kinds of things the model finds, and how the bot deals with each of them.
"""

from dataclasses import dataclass
from re import split


@dataclass(frozen=True, eq=False)
class Profile:
    name: str
    # Words naming the kind of thing, "log" or "ore". They win over the aliases: "rough"
    # alone is the tier 1 rock, "rough log" is a tree.
    kinds: tuple
    # Names of the materials, in every tier and every dataset.
    aliases: tuple
    gatherable: bool = True
    # Maximum time in second spent walking to a node, then gathering it.
    moving_timeout: float = 20.0
    gathering_timeout: float = 60.0
    # Where to click, as a fraction of the height of the box from its top. Nodes are
    # drawn standing up, their foot is where the game expects the click.
    click_height: float = 0.65

    def match(self, label, names):
        label = label.lower().strip()
        words = split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", label)
        compact = label.replace(" ", "").replace("_", "").replace("-", "")

        return any(name in words or (len(name) >= 4 and name in compact) for name in names)

    def __str__(self):
        return self.name


TREE = Profile("tree", kinds=("tree", "trees", "wood", "log", "logs"),
               aliases=("birch", "chestnut", "pine", "cedar", "bloodoak", "ashenbark", "whitewood"),
               moving_timeout=20, gathering_timeout=75, click_height=0.8)

STONE = Profile("stone", kinds=("stone", "stones", "rock", "rocks"),
                aliases=("rough", "rought", "lime", "limestone", "sand", "sandstone", "travertine", "granite",
                         "slate", "basalt", "marble"))

ORE = Profile("ore", kinds=("ore", "ores"),
              aliases=("copper", "tin", "iron", "titan", "titanium", "runite", "meteorite", "adamantium"))

FIBER = Profile("fiber", kinds=("fiber", "fibre", "fibers"),
                aliases=("cotton", "flax", "hemp", "skyflower", "amberleaf", "sunflax", "ghost", "redleaf", "silk"),
                click_height=0.6)

HIDE = Profile("hide", kinds=("hide", "hides", "animal", "animals"),
               aliases=("rabbit", "fox", "boar", "wolf", "bear", "deer"),
               moving_timeout=30, gathering_timeout=60, click_height=0.5)

MONSTER = Profile("monster", kinds=("monster", "monsters", "mob", "mobs", "enemy", "enemys", "enemies"),
                  aliases=("bandit", "undead", "keeper", "heretic", "morgana", "skeleton"),
                  gatherable=False)

UNKNOWN = Profile("unknown", kinds=(), aliases=(), gatherable=False)

PROFILES = (TREE, STONE, ORE, FIBER, HIDE, MONSTER)
GATHERABLE = tuple(profile for profile in PROFILES if profile.gatherable)

# Classes of the model trained by tools/train.py, in the order of their ids.
CLASSES = tuple(profile.name for profile in PROFILES)


def resolve(label):
    """
    :param label: Class name given by a model or a dataset.
    :return: Matching Profile, UNKNOWN when nothing matches.
    """
    for names in ("kinds", "aliases"):
        for profile in PROFILES:
            if profile.match(label, getattr(profile, names)):
                return profile

    return UNKNOWN


def resolve_targets(names):
    """
    :param names: Names of the resources to gather, None for all of them.
    :return: Tuple of gatherable profiles.
    """
    if names is None:
        return GATHERABLE

    targets = []

    for name in names:
        profile = resolve(name)

        if not profile.gatherable:
            raise ValueError(f"{name!r} cannot be gathered, pick among {[p.name for p in GATHERABLE]}")

        targets.append(profile)

    return tuple(targets)
