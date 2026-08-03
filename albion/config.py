"""
Every number the bot can be tuned by, in one place and saved as JSON.

The previous bot kept its tuning as class constants spread over a dozen files, which
meant that changing a timeout was a code edit and that nobody could tell which values had
been arrived at by measurement and which were guesses. Everything here carries a comment
saying where it came from, and the ones that were measured against the running game say
so, because those are the ones not to "clean up" later.
"""

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WindowConfig:
    """Which window to drive and how to photograph it."""

    # Only part of the title has to match. The Windows client calls itself "Albion Online
    # Client"; the Linux one through Flatpak does not, so a short name covers both.
    title: str = "Albion Online Client"

    # Height in pixels of the title bar to cut off the top of the capture, so the model
    # is shown the game rather than the window decoration. Window managers that draw no
    # decoration, every Wayland compositor for instance, want 0.
    decoration_height: int = 30

    # Ask the window for its own picture rather than reading the screen where it sits.
    # Measured with the game deliberately covered by a console: 2.4% of the frame black
    # this way against 41.5% reading the screen. Turning it off falls back to the screen,
    # which works only while nothing overlaps the game.
    capture_window_content: bool = True

    # Raise the game before acting. Clicks go to whatever is under the cursor on screen,
    # so this is needed even when the capture no longer depends on it.
    keep_foreground: bool = True


@dataclass
class VisionConfig:
    """The model and how much of what it says to believe."""

    weights: str = "best_merged.pt"
    yolov5: str = "yolov5"

    # The square the window is squeezed into for the model. Do not raise it: the weights
    # were trained at 640 and inference at 960 was measured far worse, 5 frames of 42
    # holding a tree against 16.
    image_size: int = 640

    # Run the model over each frame flipped and rescaled as well, and merge the boxes.
    # Costs 157ms a frame against 69ms, which is still three times what the bot asks for,
    # and it is what makes trees usable: measured over the 42 frames of the farming zone
    # in images/dataset, frames holding a tree the bot would act on went from 2 to 14 at
    # 0.5 confidence, and the median best score in a frame from 0.16 to 0.36.
    augment: bool = True

    # yolov5 throws away anything under this during its own suppression, before the bot
    # is given the boxes. Its default of 0.25 quietly became the real threshold of the
    # old bot: a resource asking for less than that never saw the boxes to decide on, and
    # a node could not be followed at a lower bar than it was picked at. Kept low so the
    # thresholds below are the only ones that matter.
    model_floor: float = 0.10

    # Believed by default. Anything the model is less sure of asks for its own value in
    # the resource table.
    confidence: float = 0.50

    # Following a node already found is held to this fraction of the normal threshold.
    # The score of one tree wanders by ten points between frames while nothing on screen
    # moves, so following it at the same bar lost it every few frames.
    tracking_factor: float = 0.5


@dataclass
class InputConfig:
    """Mouse and keyboard timing."""

    # Seconds the cursor takes to travel, rather than being put there between two frames.
    # The game works out what is under the cursor from the movement it receives, so a
    # pointer that jumps the width of the screen leaves it having never been over the
    # tree at all.
    move_duration: float = 0.35

    # Seconds between the cursor arriving and the button going down. Without it the game
    # reads the click against whatever it last knew was under the cursor, which is the
    # corner the bot parks in, and the character never moves. This one line was the
    # difference between the old bot working and doing nothing whatsoever.
    click_delay: float = 0.5

    # Where the cursor waits between actions, as a fraction of the window, so the tooltip
    # the game draws under it stops covering what the model is looking at. Kept clear of
    # the top left corner, which is the panic button.
    park_x: float = 0.03
    park_y: float = 0.10

    # Distance in pixels from the top left corner at which the bot gives up, so throwing
    # the mouse there stops it.
    failsafe_radius: int = 2

    mount_key: str = "a"


@dataclass
class TierRule:
    """Which tiers of one resource are worth stopping for."""

    minimum: int = 1
    maximum: int = 8


@dataclass
class LabelConfig:
    """Reading the name the game writes over a node."""

    # Off until asked for. It costs about 600ms a node, which is nothing against the walk
    # to it, but it is only useful to somebody who actually wants to skip tiers.
    enabled: bool = False

    # How much of the window around the cursor to read, in native pixels. The name is
    # drawn above the node and is wider than the trunk it belongs to.
    across: int = 260
    above: int = 150
    below: int = 60

    # How sure the engine has to be of a line before it is believed at all.
    confidence: float = 0.55

    # How much nearer an enchantment colour than to white a reading has to be before it
    # counts as enchanted. White sits close to every rarity colour, so without a margin
    # every plain resource reads as faintly green.
    enchantment_margin: float = 25.0


@dataclass
class GatheringConfig:
    """Working a node."""

    # Take enchanted nodes as well as plain ones, and optionally nothing but enchanted.
    # Both need the label reading turned on, since enchantment is read off the colour of
    # the name the game writes.
    gather_enchanted: bool = True
    only_enchanted: bool = False

    # A node holds several charges and one click takes one of them, so it is clicked
    # again until it gives nothing back. A tree measured in the live game gave 12.
    charge_attempts: int = 16

    # Seconds between two charges, the character needs a moment to swing again and a
    # click through the animation is ignored.
    charge_delay: float = 0.8

    # Seconds a charge after the first waits for the character to start working. It is
    # already standing on the node by then, so it either starts swinging quickly or there
    # is nothing left, and waiting the full walking timeout to learn that is most of the
    # time the bot would spend.
    arrival_timeout: float = 6.0

    # How many scans in a row may fail to find a node the character is standing at before
    # it is given up on. One miss is the model blinking; walking off on the first one is
    # what left half chopped trees standing.
    lost_attempts: int = 3

    # Seconds between two looks while following a node.
    poll: float = 0.5

    # How far a node may have moved across the screen between two looks and still be the
    # same node, as a fraction of the window width. The camera follows the character
    # walking, so a node slides a long way on the approach.
    track_radius: float = 0.18

    # Seconds a gathered node is left alone afterwards, and how close counts as the same
    # node, in screen pixels.
    depleted_cooldown: float = 120.0
    depleted_radius: int = 60


@dataclass
class VerifyConfig:
    """Asking the game whether a detection is really a resource."""

    # The forests hold far more trees than gatherable ones and they are the same tree to
    # look at, so the model finds both: of six it picked out of the farming zone, one was
    # a node and five were scenery. The game knows the difference and says so, by writing
    # the name of a real node over it in near white and writing nothing over scenery.
    enabled: bool = True

    # Pixels around the hovered point to look in. The name is drawn above the node and is
    # wider than the trunk it belongs to.
    radius: int = 90

    # Grey level over which a pixel counts as part of that name.
    bright: int = 200

    # How many more bright pixels than before the hover mean the game named something.
    # Measured: the one real node put 42 more on screen, the five scenery ones put none.
    jump: int = 15

    # Seconds to leave the cursor there before looking, the name fades in.
    settle: float = 0.8

    # Seconds a tree the game refuses to name is left alone. It will still be scenery in
    # a minute, so this is far longer than an emptied node.
    scenery_cooldown: float = 300.0

    # A detection the model is at least this sure of is worth one click even when the
    # hover found no name, because the hover has ways of being wrong (the label drawn
    # outside the searched box, the cursor resting on a gap in the leaves) and the click
    # is the truth: a real node starts a gathering animation and scenery does not. Being
    # wrong costs one walk; the other mistake, trusting a bad verifier everywhere, cost a
    # whole run that gathered nothing.
    trust_confidence: float = 0.60


@dataclass
class NavigationConfig:
    """Walking about."""

    # The roam is an expanding square rather than a random walk, so ground is covered
    # outwards from where it started instead of crossing the same emptied clearing.
    radius: float = 0.22

    # The camera looks at the world from an angle, so a pixel up the screen is more world
    # than a pixel across it and the vertical part of a step is squashed to match.
    vertical_squash: float = 0.6

    wait: float = 3.0

    # The part of the window that can be clicked to walk, as fractions. The interface is
    # left out: clicking the portrait, the action bar or the minimap opens a panel over
    # the game instead of moving anybody.
    area_left: float = 0.08
    area_top: float = 0.16
    area_right: float = 0.74
    area_bottom: float = 0.60

    # How close to something the model knows but cannot gather, a monster above all, the
    # bot is willing to walk, in screen pixels.
    hazard_distance: int = 130


@dataclass
class AntiStuckConfig:
    """Noticing that nothing is happening any more."""

    # How much the frame has to change, as a mean grey level, for the character to count
    # as having moved. Below this the screen is the same picture and the character is
    # standing still.
    motion_threshold: float = 1.5

    # Seconds of no movement while the bot believes it is walking somewhere before it
    # calls itself stuck.
    stuck_after: float = 6.0

    # How many scans in a row may find nothing before the bot stops sweeping outwards and
    # tries something else.
    empty_scans_before_recovery: int = 8

    # How many times the same recovery may be tried before the bot gives up and says so
    # rather than repeating itself until morning.
    max_attempts: int = 5

    # A frame changing by less than this against the one before counts as the game not
    # rendering at all, rather than as a character standing still. Standing still in a
    # forest measures around 0.6 to 1.0, because the grass and the light move; the login
    # screen measures exactly 0.00.
    frozen_threshold: float = 0.05

    # How many frames in a row have to be identical before the bot decides it is not in
    # the game: a login screen, a loading screen, or a disconnect.
    frozen_samples: int = 4

    # How long to keep checking for the game to come back before giving up, in minutes.
    # Waiting is right rather than stopping, because a loading screen between zones looks
    # exactly the same as a disconnect and only one of them needs a human.
    wait_for_game_minutes: float = 10.0


@dataclass
class MountConfig:
    """Riding between nodes."""

    enabled: bool = True

    # Seconds to wait for the mounting animation, which is interrupted by any movement.
    mount_time: float = 4.0

    # Ride to anything further away than this, in screen pixels; walk to anything nearer,
    # since mounting and dismounting costs more than the walk saves.
    ride_beyond: int = 420

    # Seconds after a failed attempt before trying to mount again.
    retry_after: float = 5.0


@dataclass
class Config:
    """Everything, loaded from and saved to JSON."""

    window: WindowConfig = field(default_factory=WindowConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    input: InputConfig = field(default_factory=InputConfig)
    gathering: GatheringConfig = field(default_factory=GatheringConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    anti_stuck: AntiStuckConfig = field(default_factory=AntiStuckConfig)
    mount: MountConfig = field(default_factory=MountConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)

    # Which tiers of each resource to bother with, by canonical resource name.
    tiers: dict = field(default_factory=lambda: {
        name: TierRule() for name in ("tree", "stone", "ore", "fiber", "hide")})

    # Resources to gather, by canonical name. Empty means every one the model knows.
    targets: list = field(default_factory=lambda: ["tree"])

    # Name of a recorded route to walk instead of the expanding search, empty for none.
    # Record one with: python -m albion.record --name <name>
    route: str = ""

    log_level: str = "INFO"

    @property
    def weights_path(self):
        return ROOT / self.vision.weights

    @property
    def yolov5_path(self):
        return ROOT / self.vision.yolov5

    @classmethod
    def load(cls, path=None):
        """
        Read the settings, falling back to the defaults for anything missing.

        A file written by an older version is missing whatever has been added since, and
        a bot that refuses to start over that would be worse than one that fills the gaps
        in, so unknown keys are ignored and absent ones keep their default.

        :param path: File to read, config.json beside the project when left to None.
        :return: Config.
        """
        path = Path(path) if path else ROOT / "config.json"

        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, values):
        """
        Build from plain dictionaries, one level of nesting deep.

        :param values: Dictionary of settings.
        :return: Config.
        """
        known = {f.name: f for f in fields(cls)}
        arguments = {}

        for name, value in values.items():
            if name not in known:
                continue

            default = known[name].default_factory() if callable(
                getattr(known[name], "default_factory", None)) else None

            if is_dataclass(default) and isinstance(value, dict):
                allowed = {f.name for f in fields(default)}
                arguments[name] = type(default)(
                    **{k: v for k, v in value.items() if k in allowed})
            elif isinstance(default, dict) and isinstance(value, dict) and default:
                # A mapping whose values are themselves settings, the per resource tier
                # rules being the one of those. Saved as plain dictionaries, so they have
                # to be built back into their own type or every reader of them breaks on
                # an attribute a dictionary does not have.
                inner = type(next(iter(default.values())))
                allowed = {f.name for f in fields(inner)} if is_dataclass(inner) else None

                arguments[name] = {
                    key: inner(**{k: v for k, v in entry.items() if k in allowed})
                    if allowed is not None and isinstance(entry, dict) else entry
                    for key, entry in value.items()
                }
            else:
                arguments[name] = value

        return cls(**arguments)

    def save(self, path=None):
        """
        Write the settings out.

        :param path: File to write, config.json beside the project when left to None.
        :return: The path written to.
        """
        path = Path(path) if path else ROOT / "config.json"

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

        return path
