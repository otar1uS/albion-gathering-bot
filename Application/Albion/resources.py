from re import split


class ResourceProfile:
    """
    Gathering behaviour of one kind of resource node.
    """

    def __init__(self, name, aliases, gathering_timeout, moving_timeout, delay_between_nodes,
                 confidence=None):
        """
        :param name: Canonical name of the resource.
        :param aliases: Labels of the model matching this resource.
        :param gathering_timeout: Maximum time in second spent on a node before giving up.
        :param moving_timeout: Maximum time in second spent walking to a node before giving up.
        :param delay_between_nodes: Time in second waited once a node is depleted.
        :param confidence: Lowest confidence to accept for this resource alone, None to
                           use the one the detection was built with.
        """
        self.name = name
        self.aliases = aliases
        self.gathering_timeout = gathering_timeout
        self.moving_timeout = moving_timeout
        self.delay_between_nodes = delay_between_nodes
        self.confidence = confidence

    def match(self, label):
        """
        Check if a model label belongs to this resource.

        Labels aren't named the same in every dataset, so an alias matches either a
        whole word of the label ("T4 tree") or, when it is long enough to be
        unambiguous, a part of it ("CopperOre").

        :param label: Label given by the model.
        :return: True when the label refers to this resource.
        """
        label = label.lower().strip()
        words = split(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", label)

        for alias in self.aliases:
            if alias in words or (len(alias) >= 4 and alias in label.replace(" ", "")):
                return True

        return False

    def __str__(self):
        return self.name


# Aliases hold the name of the resource in every tier, because a model trained on
# tier 4 knows a tree as "rought log" and a rock as "travertine".
#
# A tree holds more charges than a rock and the character stops further away from it,
# so both the gathering and the walking phases need a longer timeout. Waiting too
# long costs nothing, the wait stops as soon as the node is depleted.
# "rough" and "rought" are deliberately absent. The tier 2 wood is "Rough Logs" and the
# tier 2 rock is "Rough Stone", and since the profiles are tried in order, a "rough" here
# claimed the rock as a tree before STONE was ever asked. "logs" catches the wood on its
# own, so nothing is lost by leaving the ambiguous word out.
# Trees are the thinnest class the weights were trained on, 683 boxes against the 1587
# of the ore, so the model recognises them correctly but is not sure of them: measured
# over the 42 Forgotten Vigils frames in images/dataset, it finds a tree on 2 of them
# above 0.5 and on 16 above 0.25, and the extra boxes are real trees, the birch being
# gathered included. Nothing else needs lowering, stone, ore and hide all score above
# 0.95 and a lower bar would only cost them false positives.
# Raise this back once trees from this zone have been labelled and trained on.
TREE = ResourceProfile(
    name="tree",
    aliases=("tree", "trees", "wood", "log", "logs", "birch", "chestnut",
             "pine", "cedar", "bloodoak", "ashenbark"),
    gathering_timeout=45,
    moving_timeout=20,
    delay_between_nodes=1.5,
    confidence=0.30,
)

STONE = ResourceProfile(
    name="stone",
    aliases=("stone", "stones", "rock", "rocks", "limestone", "sandstone", "travertine",
             "granite", "slate", "basalt", "marble"),
    gathering_timeout=30,
    moving_timeout=15,
    delay_between_nodes=1,
)

ORE = ResourceProfile(
    name="ore",
    aliases=("ore", "ores", "copper", "tin", "iron", "titanium", "runite", "meteorite",
             "adamantium"),
    gathering_timeout=30,
    moving_timeout=15,
    delay_between_nodes=1,
)

FIBER = ResourceProfile(
    name="fiber",
    aliases=("fiber", "fibre", "cotton", "flax", "hemp", "amberleaf", "sunflax", "ghost",
             "redleaf", "silk"),
    gathering_timeout=30,
    moving_timeout=15,
    delay_between_nodes=1,
)

HIDE = ResourceProfile(
    name="hide",
    aliases=("hide", "hides", "animal", "rabbit", "fox", "boar", "wolf", "bear", "deer"),
    gathering_timeout=40,
    moving_timeout=25,
    delay_between_nodes=1,
)

PROFILES = (TREE, STONE, ORE, FIBER, HIDE)

# Used when the model label doesn't match any known resource.
DEFAULT = ResourceProfile(
    name="unknown",
    aliases=(),
    gathering_timeout=15,
    moving_timeout=15,
    delay_between_nodes=1,
)


def resolve(label):
    """
    Find the profile of a model label.

    :param label: Label given by the model.
    :return: Matching ResourceProfile, DEFAULT when the label is unknown.
    """
    for profile in PROFILES:
        if profile.match(label):
            return profile

    return DEFAULT


def resolve_targets(names):
    """
    Find the profiles asked by the user.

    :param names: Iterable of canonical resource names, None for all of them.
    :return: Tuple of ResourceProfile.
    """
    if names is None:
        return PROFILES

    targets = []

    for name in names:
        profile = resolve(name)

        if profile is DEFAULT:
            raise Exception(f"Unknown resource {name}, pick one of {[str(p) for p in PROFILES]}")

        targets.append(profile)

    return tuple(targets)
