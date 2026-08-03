"""
What the bot knows about each kind of resource.

The alias matching is kept from the previous bot because it earned its place: weights
trained on one tier call a tree "rought log" and a rock "travertine", so a label can only
be turned into behaviour by knowing every name the thing goes by. New weights can be
dropped in without touching anything else as long as their labels are covered here.

What is new is that a resource carries its own thresholds and priority, rather than the
whole bot running on one number. The classes are not learned equally well, and pretending
they are is what forced the old bot to choose between missing every tree and clicking
every shadow.
"""

from dataclasses import dataclass, field
from re import split


@dataclass
class ResourceProfile:
    """How one kind of resource is recognised and worked."""

    name: str
    aliases: tuple

    # Seconds spent on a node, and walking to one, before giving up.
    gathering_timeout: float = 30.0
    moving_timeout: float = 15.0

    # Seconds waited once a node is emptied.
    delay_between_nodes: float = 1.0

    # Lowest confidence to accept for this resource alone, None to use the general one.
    confidence: float = None

    # Which resource to go for when several are in reach, higher first. Only consulted
    # when more than one kind is being gathered at once.
    priority: int = 0

    # Whether this is something to gather at all. Monsters are recognised so they can be
    # walked around, never clicked.
    gatherable: bool = True

    # How far down the box to aim, 0 being its top edge. The middle of a tall tree's box
    # is canopy, and the part of the node the game lets a cursor land on is the trunk, so
    # trees are aimed low. Squat things like rocks are fine at their middle.
    aim_depth: float = 0.5

    def match(self, label):
        """
        Check whether a label from the model refers to this resource.

        Labels are not named the same in every dataset, so an alias matches either a
        whole word of the label ("T4 tree") or, when it is long enough to be unambiguous,
        a part of it ("CopperOre").

        :param label: Label given by the model.
        :return: True when the label refers to this resource.
        """
        label = label.lower().strip()
        words = split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", label)

        for alias in self.aliases:
            if alias in words or (len(alias) >= 4 and alias in label.replace(" ", "")):
                return True

        return False

    def threshold(self, default):
        """
        Lowest confidence to accept for this resource.

        :param default: The general threshold.
        :return: Confidence under which a box of this kind is dropped.
        """
        return default if self.confidence is None else self.confidence

    def __str__(self):
        return self.name


# Trees are the thinnest class the weights know: 560 boxes against the 11650 of the ore,
# a difference of twenty one times. The model therefore recognises a tree correctly and
# is not sure of it, and holding it to the same bar as an ore vein finds nothing at all.
# Measured over the 42 frames of the farming zone in images/dataset, with augmented
# inference: a tree is found in 24 of them at 0.3 and in 14 at 0.5. Put this back up once
# trees from the zone have been labelled and trained on.
#
# "rough" and "rought" are deliberately absent from the aliases. Tier 2 wood is "Rough
# Logs" and tier 2 rock is "Rough Stone", and since profiles are tried in order, a
# "rough" here claimed the rock as a tree before stone was ever asked. "logs" catches the
# wood on its own.
TREE = ResourceProfile(
    name="tree",
    aliases=("tree", "trees", "wood", "log", "logs", "birch", "chestnut",
             "pine", "cedar", "bloodoak", "ashenbark"),
    gathering_timeout=45,
    moving_timeout=20,
    delay_between_nodes=1.5,
    confidence=0.30,
    priority=10,
    aim_depth=0.68,
)

STONE = ResourceProfile(
    name="stone",
    aliases=("stone", "stones", "rock", "rocks", "limestone", "sandstone", "travertine",
             "granite", "slate", "basalt", "marble"),
    gathering_timeout=30,
    moving_timeout=15,
    priority=5,
)

ORE = ResourceProfile(
    name="ore",
    aliases=("ore", "ores", "copper", "tin", "iron", "titanium", "runite", "meteorite",
             "adamantium"),
    gathering_timeout=30,
    moving_timeout=15,
    priority=8,
)

FIBER = ResourceProfile(
    name="fiber",
    aliases=("fiber", "fibre", "cotton", "flax", "hemp", "amberleaf", "sunflax", "ghost",
             "redleaf", "silk"),
    gathering_timeout=30,
    moving_timeout=15,
    priority=5,
)

HIDE = ResourceProfile(
    name="hide",
    aliases=("hide", "hides", "animal", "rabbit", "fox", "boar", "wolf", "bear", "deer"),
    gathering_timeout=40,
    moving_timeout=25,
    priority=3,
)

# Recognised so the roaming can walk around it. Never gathered, never clicked.
MONSTER = ResourceProfile(
    name="monster",
    aliases=("monster", "mob", "enemy", "creature"),
    gatherable=False,
)

# Anything the model names that none of the above claims. Treated as a hazard rather than
# as a resource, because clicking something the bot cannot identify is how a gathering
# run turns into a fight.
UNKNOWN = ResourceProfile(name="unknown", aliases=(), gatherable=False)

PROFILES = (TREE, STONE, ORE, FIBER, HIDE, MONSTER)

GATHERABLE = tuple(profile for profile in PROFILES if profile.gatherable)


def resolve(label):
    """
    Find the profile a label belongs to.

    :param label: Label given by the model, or a canonical resource name.
    :return: Matching ResourceProfile, UNKNOWN when nothing claims it.
    """
    for profile in PROFILES:
        if profile.match(label):
            return profile

    return UNKNOWN


def resolve_targets(names):
    """
    Turn the resources asked for into profiles.

    :param names: Iterable of canonical names, None or empty for everything gatherable.
    :return: Tuple of ResourceProfile.
    """
    if not names:
        return GATHERABLE

    targets = []

    for name in names:
        profile = resolve(name)

        if profile is UNKNOWN or not profile.gatherable:
            raise ValueError(
                f"{name!r} is not something that can be gathered, pick from "
                f"{[str(p) for p in GATHERABLE]}")

        if profile not in targets:
            targets.append(profile)

    return tuple(targets)


def names():
    """
    Every resource that can be asked for.

    :return: List of canonical names.
    """
    return [str(profile) for profile in GATHERABLE]
