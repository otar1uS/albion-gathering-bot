"""
Turning the name the game writes over a node into a tier and an enchantment.

Albion never prints a tier number on a resource. It prints a name, and the name is the
tier: a tree holding tier 5 wood is called Pine Logs and nothing else. So a tier filter is
a vocabulary lookup, and the vocabulary is small enough to write down.

The tables are data on purpose. If one of these names is wrong for the current patch, or
a resource is added, this is the only file to correct, and the bot logs every name it
reads so a wrong one shows up as "unknown resource name" rather than as mysteriously bad
filtering.

Enchantment is not in the name. Albion colours the writing instead, the same colours it
uses for item rarity everywhere else, so it is read off the pixels rather than the text.
"""

from dataclasses import dataclass
from difflib import get_close_matches

from .. import logs

log = logs.get("game.tiers")

# Tier to name, per resource. Wood, stone and hide start at tier 1; ore and fiber have no
# tier 1 at all, which is why those tables start at 2.
NAMES = {
    "tree": {
        1: "rough logs", 2: "birch logs", 3: "chestnut logs", 4: "pine logs",
        5: "cedar logs", 6: "bloodoak logs", 7: "ashenbark logs", 8: "whitewood logs",
    },
    "stone": {
        1: "rough stone", 2: "limestone", 3: "sandstone", 4: "travertine",
        5: "granite", 6: "slate", 7: "basalt", 8: "marble",
    },
    "ore": {
        2: "copper ore", 3: "tin ore", 4: "iron ore", 5: "titanium ore",
        6: "runite ore", 7: "meteorite ore", 8: "adamantium ore",
    },
    "fiber": {
        2: "cotton", 3: "flax", 4: "hemp", 5: "skyflower",
        6: "redleaf cotton", 7: "sunflax", 8: "ghost hemp",
    },
    "hide": {
        1: "scrap hide", 2: "rugged hide", 3: "thin hide", 4: "medium hide",
        5: "heavy hide", 6: "robust hide", 7: "thick hide", 8: "resilient hide",
    },
}

# Every name the bot knows, to the resource and tier it means.
LOOKUP = {name: (resource, tier)
          for resource, tiers in NAMES.items()
          for tier, name in tiers.items()}

# Rarity colours Albion writes an enchanted name in, as BGR. Plain resources are written
# in white, and each enchantment level takes the next colour up. These are approximate on
# purpose: the writing is drawn over whatever is behind it and is anti aliased, so the
# match is to the nearest of these rather than to an exact value.
ENCHANTMENT_COLOURS = {
    0: (235, 235, 235),
    1: (110, 210, 120),
    2: (235, 180, 110),
    3: (225, 130, 200),
    4: (120, 215, 245),
}

# How far a read name may be from a known one and still count as it. OCR turns an l into
# a 1 and an rn into an m often enough that demanding an exact match throws away good
# readings, and the vocabulary is small and unalike enough that a loose match is safe.
CLOSENESS = 0.72


@dataclass
class Named:
    """What the game said is under the cursor."""

    # What was read, before any tidying, so a bad match can be understood.
    raw: str

    # Canonical resource name, None when nothing in the vocabulary is close.
    resource: str = None

    tier: int = None
    enchantment: int = 0

    @property
    def known(self):
        return self.resource is not None

    def __str__(self):
        if not self.known:
            return f"{self.raw!r} (not a resource this bot knows)"

        enchant = f".{self.enchantment}" if self.enchantment else ""

        return f"T{self.tier}{enchant} {self.raw}"


def normalise(text):
    """
    Reduce a read name to something comparable.

    The split before a capital matters: the engine returns "BirchLogs" as often as "Birch
    Logs", because the tooltip is tight and the space is a few pixels. Without the split
    that only matches by luck through the fuzzy pass.

    :param text: What the OCR returned.
    :return: Lowercase, single spaced, letters and spaces only.
    """
    spaced = []

    for index, character in enumerate(text):
        if (character.isupper() and index > 0 and text[index - 1].islower()):
            spaced.append(" ")

        spaced.append(character if character.isalpha() else " ")

    return " ".join("".join(spaced).lower().split())


def identify(text, enchantment=0):
    """
    Work out which resource and tier a read name refers to.

    :param text: What the OCR returned.
    :param enchantment: Enchantment level read from the colour of the writing.
    :return: Named.
    """
    cleaned = normalise(text)
    found = Named(raw=text.strip(), enchantment=enchantment)

    if not cleaned:
        return found

    if cleaned in LOOKUP:
        found.resource, found.tier = LOOKUP[cleaned]
        return found

    close = get_close_matches(cleaned, LOOKUP, n=1, cutoff=CLOSENESS)

    if close:
        found.resource, found.tier = LOOKUP[close[0]]
        log.debug("read %r, taking it as %r", text, close[0])
    else:
        log.debug("read %r, which is not a resource name this bot knows", text)

    return found


def restrictive(config, resource=None):
    """
    Whether the user has actually asked for anything to be skipped.

    Everything below only matters when a filter has been set. Left at the full range, a
    node whose name could not be read should simply be gathered, and no warning is worth
    printing about it.

    :param config: Config.
    :param resource: One resource, or None for any of them.
    :return: True when some tier or enchantment is being deliberately excluded.
    """
    if config.gathering.only_enchanted or not config.gathering.gather_enchanted:
        return True

    rules = ([config.tiers[resource]] if resource in config.tiers
             else config.tiers.values())

    return any(rule.minimum > 1 or rule.maximum < 8 for rule in rules)


def wanted(found, config):
    """
    Whether a named node passes the tier and enchantment rules.

    A node whose name could not be read is kept when no filter is set, and skipped when
    one is. Both halves of that matter. Reading fails for dull reasons, the writing being
    drawn over a pale patch of ground most of them, so refusing everything unreadable by
    default would let one bad reading turn the whole run off. But somebody who asked for
    tier 5 and up did not ask to be given tier 2 whenever the reader had a bad moment,
    and quietly ignoring the filter is the worse mistake of the two. The log says which
    happened either way.

    :param found: Named.
    :param config: Config.
    :return: (True to gather it, why not when False).
    """
    if not found.known:
        if restrictive(config):
            return False, "its name could not be read, and a tier filter is set"

        return True, ""

    rule = config.tiers.get(found.resource)

    if rule is None:
        return True, ""

    if found.tier is not None and not rule.minimum <= found.tier <= rule.maximum:
        return False, (f"T{found.tier} is outside the T{rule.minimum} to T{rule.maximum} "
                       f"asked for")

    if config.gathering.only_enchanted and not found.enchantment:
        return False, "not enchanted, and only enchanted nodes were asked for"

    if not config.gathering.gather_enchanted and found.enchantment:
        return False, f"enchanted (.{found.enchantment}), which was turned off"

    return True, ""
