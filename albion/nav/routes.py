"""
Farming routes: recorded once by playing, then repeated.

A route is the user's own clicks, written down while they walk the lap they want farmed.
Recording watches the real mouse, keeps every left click that lands inside the game
window, and stores it as a fraction of the window rather than a pixel, so the same route
survives the window being moved or resized, and even a different resolution.

Playing one back is not blind replay. Blind replay drifts: one rock in the way and every
later click is somewhere it should not be. Instead the route only supplies the walking —
each waypoint is clicked, the character is given time to get there, and then the bot
scans and gathers whatever is around before moving on. The gathering, the verification
and the anti stuck all stay in charge; the route replaces only the aimless part of
roaming. When the lap ends it starts again, which is what a farming route is.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

from .. import logs

log = logs.get("nav.routes")

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "routes"


@dataclass
class Waypoint:
    """One click of the recorded lap."""

    # As fractions of the game window, so the route survives moves and resizes.
    fx: float
    fy: float

    # Seconds the recorder saw pass before the next click. Capped on playback: the
    # pauses where the user stopped to gather are the bot's job now.
    pause: float


class Route:
    """A recorded lap."""

    def __init__(self, name, waypoints=None):
        self.name = name
        self.waypoints = list(waypoints or [])

    @property
    def path(self):
        return FOLDER / f"{self.name}.json"

    def save(self):
        """
        Write the route out.

        :return: The file written.
        """
        FOLDER.mkdir(exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"name": self.name,
                       "waypoints": [asdict(w) for w in self.waypoints]},
                      handle, indent=2)

        log.info("saved %d waypoints to %s", len(self.waypoints), self.path)

        return self.path

    @classmethod
    def load(cls, name):
        """
        Read a route back.

        :param name: Name it was saved under.
        :return: Route.
        """
        path = FOLDER / f"{name}.json"

        if not path.exists():
            available = ", ".join(p.stem for p in FOLDER.glob("*.json")) or "none yet"
            raise FileNotFoundError(
                f"no route called {name!r}. Recorded routes: {available}. Make one with: "
                f"python -m albion.record --name {name}")

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        return cls(data["name"], [Waypoint(**w) for w in data["waypoints"]])

    @staticmethod
    def names():
        """
        Every route on disk.

        :return: List of names.
        """
        return sorted(p.stem for p in FOLDER.glob("*.json"))


class RoutePlayer:
    """Feeds the bot the next leg of the lap whenever it has nothing better to do."""

    # Longest the player will honour a recorded pause. The user stopping four minutes to
    # gather a big tree is not part of the walking.
    PAUSE_CAP = 4.0

    # Shortest pause between two waypoint clicks, so a route recorded with rapid
    # clicking still gives the character time to actually move.
    PAUSE_FLOOR = 1.5

    def __init__(self, route, capture, navigator, controller):
        self.route = route
        self.capture = capture
        self.navigator = navigator
        self.controller = controller
        self.at = 0
        self.laps = 0

    def step(self):
        """
        Walk the next leg of the route.

        :return: True when a waypoint was clicked.
        """
        if not self.route.waypoints:
            return False

        if self.at >= len(self.route.waypoints):
            self.at = 0
            self.laps += 1
            log.info("lap %d of the route done, starting it again", self.laps)

        waypoint = self.route.waypoints[self.at]
        self.at += 1

        x, y = self.capture.rect.fraction(waypoint.fx, waypoint.fy)

        if self.navigator.on_interface(x, y):
            # The window is a different shape than when it was recorded and this point
            # now lands on the interface. Skipping it loses one leg, clicking it opens a
            # panel over the game.
            log.warning("waypoint %d lands on the interface at this window size, "
                        "skipping it", self.at)
            return False

        log.info("route waypoint %d of %d", self.at, len(self.route.waypoints))
        self.controller.click(x, y)
        self.controller.wait(min(max(waypoint.pause, self.PAUSE_FLOOR), self.PAUSE_CAP))

        return True


def record(config, name, stop_key="f9"):
    """
    Watch the user walk a lap and write it down.

    :param config: Config.
    :param name: Name to save under.
    :param stop_key: Key that ends the recording.
    :return: The recorded Route.
    """
    from pynput import keyboard, mouse

    from ..vision import capture as capture_module

    capture = capture_module.build(config)
    capture.focus()

    log.info("recording %r. Play normally: walk the lap you want farmed by clicking in "
             "the game. Press %s when the lap is done.", name, stop_key.upper())

    waypoints = []
    last = {"when": time()}
    done = {"stop": False}

    def on_click(x, y, button, pressed):
        if not pressed or button is not mouse.Button.left:
            return

        rect = capture.refresh()

        if not rect.contains(x, y):
            return

        now = time()
        waypoints.append(Waypoint(
            fx=round((x - rect.left) / rect.width, 4),
            fy=round((y - rect.top) / rect.height, 4),
            pause=round(now - last["when"], 2),
        ))
        last["when"] = now

        log.info("waypoint %d at (%.2f, %.2f) of the window", len(waypoints),
                 waypoints[-1].fx, waypoints[-1].fy)

    def on_key(key):
        wanted = getattr(keyboard.Key, stop_key, None)

        if key == wanted:
            done["stop"] = True
            return False

    with mouse.Listener(on_click=on_click), keyboard.Listener(on_press=on_key) as keys:
        keys.join()

    if not waypoints:
        log.warning("nothing was recorded, no clicks landed in the game window")

    route = Route(name, waypoints)
    route.save()

    return route
