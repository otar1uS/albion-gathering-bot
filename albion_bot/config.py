"""
Settings of the bot, kept in data/settings.json between two runs.
"""

import json
from dataclasses import asdict, dataclass, field, fields

from albion_bot import geometry, paths
from albion_bot.platform import DEFAULT_WINDOW_NAME

ROUTE_MODES = ("loop", "back and forth")


@dataclass
class Settings:
    window_name: str = DEFAULT_WINDOW_NAME
    targets: list = field(default_factory=lambda: ["tree"])
    confidence: float = 0.5
    # Route to follow, empty to gather what is in sight without moving.
    route: str = ""
    route_mode: str = ROUTE_MODES[0]
    # Minutes after which the bot stops by itself, 0 for never.
    max_minutes: float = 0.0
    # Press alt+h around each node. The gathering bar has to be calibrated the same way.
    hide_hud: bool = False
    # Skip the nodes a monster stands next to.
    avoid_monsters: bool = True
    # Key pressed to mount before walking the route again, empty to walk.
    mount_key: str = ""
    # Live view of what the model sees, in its own window. Keep it off the game, on a
    # second screen, or the bot sees it too.
    preview: bool = False
    # Minimize the interface while the bot reads the game, or it may cover part of it.
    minimize_ui: bool = True
    # Where the character stands, fractions of the window. The camera follows it, so it
    # never moves, a bit above the middle because of the isometric view.
    character: list = field(default_factory=lambda: [0.5, 0.41])
    # Minimap, fractions of the window, set with the picker of the interface.
    minimap: list = field(default_factory=lambda: [0.845, 0.745, 0.14, 0.235])
    # Turns a way on the minimap into a way on the screen, see navigation.route.calibrate.
    minimap_matrix: list = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    # How much the gathering bar has to look like its calibration to be on screen.
    bar_threshold: float = 0.7

    def validate(self):
        """
        Replace whatever a hand edited file got wrong by the default, instead of failing
        in the middle of a run.

        :return: List of the fields that were reset.
        """
        default = Settings()
        reset = []

        def fix(name, ok):
            if not ok:
                setattr(self, name, getattr(default, name))
                reset.append(name)

        fix("window_name", isinstance(self.window_name, str) and self.window_name.strip() != "")
        fix("targets", isinstance(self.targets, list) and all(isinstance(t, str) for t in self.targets))
        fix("confidence", isinstance(self.confidence, (int, float)) and 0.01 <= self.confidence <= 0.99)
        fix("route", isinstance(self.route, str))
        fix("route_mode", self.route_mode in ROUTE_MODES)
        fix("max_minutes", isinstance(self.max_minutes, (int, float)) and self.max_minutes >= 0)
        fix("hide_hud", isinstance(self.hide_hud, bool))
        fix("avoid_monsters", isinstance(self.avoid_monsters, bool))
        fix("mount_key", isinstance(self.mount_key, str) and len(self.mount_key) <= 1)
        fix("preview", isinstance(self.preview, bool))
        fix("minimize_ui", isinstance(self.minimize_ui, bool))
        fix("character", isinstance(self.character, list) and len(self.character) == 2
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in self.character))
        fix("minimap", geometry.valid_region(self.minimap))
        fix("minimap_matrix", _matrix_ok(self.minimap_matrix))
        fix("bar_threshold", isinstance(self.bar_threshold, (int, float)) and 0.3 <= self.bar_threshold <= 0.99)

        return reset


def _matrix_ok(value):
    try:
        return len(value) == 2 and all(len(row) == 2 and all(isinstance(v, (int, float)) for v in row)
                                       for row in value)
    except TypeError:
        return False


def load(path=None) -> Settings:
    path = path or paths.SETTINGS
    settings = Settings()

    try:
        stored = json.loads(path.read_text())
    except FileNotFoundError:
        return settings
    except (OSError, ValueError) as e:
        print(f"Could not read {path}, using the defaults: {e}")
        return settings

    if isinstance(stored, dict):
        known = {f.name for f in fields(Settings)}
        for key, value in stored.items():
            if key in known:
                setattr(settings, key, value)

    reset = settings.validate()

    if reset:
        print(f"Reset to their defaults in {path}: {', '.join(reset)}")

    return settings


def save(settings: Settings, path=None):
    path = path or paths.SETTINGS
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=2))
    # Written aside then moved, so a crash never leaves half a file behind.
    temporary.replace(path)
