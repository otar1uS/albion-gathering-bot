"""
Routes: recorded by walking them once, then followed by the bot.

Recording saves a piece of the minimap every few steps, a waypoint. Following a route
means finding the next waypoint on the minimap, walking towards it, and moving on to the
next one once it is under the arrow. Only the screen is read.
"""

import json
import re
import shutil
from dataclasses import dataclass
from math import hypot

import cv2 as cv
import numpy as np

from albion_bot import paths
from albion_bot.config import ROUTE_MODES
from albion_bot.vision import minimap

# Time in second a waypoint can be walked towards without getting closer, before the bot
# considers it blocked and moves on to the next one.
STUCK_TIMEOUT = 8.0

NAME = re.compile(r"[\w\- ]{1,64}")


class Route:
    """A named list of waypoints, stored in data/routes/<name>/."""

    def __init__(self, name, patches=None):
        self.name = name
        self.patches = patches or []

    @staticmethod
    def folder(name):
        if not NAME.fullmatch(name) or name.strip() != name:
            raise ValueError(f"{name!r} cannot be a route name, use letters, digits, spaces, - and _")

        return paths.ROUTES / name

    def save(self):
        folder = self.folder(self.name)

        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(parents=True)

        for index, patch in enumerate(self.patches):
            cv.imwrite(str(folder / f"{index:04d}.png"), patch)

        (folder / "route.json").write_text(json.dumps({"waypoints": len(self.patches), "patch": minimap.PATCH,
                                                       "width": minimap.WIDTH}))

    @classmethod
    def load(cls, name):
        folder = cls.folder(name)

        try:
            info = json.loads((folder / "route.json").read_text())
        except (OSError, ValueError) as e:
            raise FileNotFoundError(f"the route {name} cannot be read: {e}") from e

        if info.get("width") != minimap.WIDTH or info.get("patch") != minimap.PATCH:
            raise ValueError(f"the route {name} was recorded by another version of the bot, record it again")

        patches = []

        for index in range(int(info.get("waypoints", 0))):
            patch = cv.imread(str(folder / f"{index:04d}.png"), cv.IMREAD_GRAYSCALE)

            if patch is None:
                raise FileNotFoundError(f"waypoint {index + 1} of the route {name} is missing, record it again")

            patches.append(patch)

        if len(patches) < 2:
            raise ValueError(f"the route {name} has less than 2 waypoints, record a longer one")

        return cls(name, patches)

    def __len__(self):
        return len(self.patches)


def names():
    if not paths.ROUTES.exists():
        return []

    return sorted(folder.name for folder in paths.ROUTES.iterdir() if (folder / "route.json").exists())


def delete(name):
    folder = Route.folder(name)

    if folder.exists():
        shutil.rmtree(folder)


class Recorder:
    """Builds a route from the frames seen while the player walks it."""

    def __init__(self, region):
        self.region = region
        self.patches = []
        self.lost = False

    def feed(self, frame):
        """
        :return: What happened worth telling, None when nothing.
        """
        current = minimap.read(frame, self.region)

        if not self.patches:
            self.patches.append(minimap.patch(current))
            return "waypoint 1 saved, walk the route now"

        dx, dy, score = minimap.locate(self.patches[-1], current)

        if score < minimap.LOST_SCORE:
            # Walked too far between two looks, or teleported: start again from here.
            self.patches.append(minimap.patch(current))
            message = None if self.lost else (f"the last waypoint went out of sight ({score:.2f}), "
                                              f"walk slower or check the minimap region")
            self.lost = True
            return message

        self.lost = False

        if hypot(dx, dy) >= minimap.step_distance(current):
            self.patches.append(minimap.patch(current))
            return f"waypoint {len(self.patches)} saved"

        return None

    def finish(self, name):
        if len(self.patches) < 2:
            raise ValueError("the route is too short, walk further before stopping the recording")

        route = Route(name, self.patches)
        route.save()
        return route


@dataclass
class Step:
    # "walk" towards direction, "arrived" at a waypoint, "lost" when no waypoint is in sight.
    status: str
    # Unit vector on screen, frame axes, while walking.
    direction: tuple = None
    waypoint: int = None


class Navigator:
    """Tells which way to walk to follow a route, one frame at a time."""

    def __init__(self, route, region, matrix, mode=ROUTE_MODES[0], clock=None):
        if mode not in ROUTE_MODES:
            raise ValueError(f"unknown route mode {mode}, pick one of {ROUTE_MODES}")

        from time import monotonic

        self.route = route
        self.region = region
        self.matrix = np.array(matrix, dtype=float)
        self.mode = mode
        self.clock = clock or monotonic
        self.index = None
        self.direction = 1
        self.best_distance = None
        self.best_at = 0.0

    def localize(self, current):
        """
        Find the waypoint the closest to the character, when starting or once lost.

        :return: True when one is in sight.
        """
        best = None

        for index, patch in enumerate(self.route.patches):
            dx, dy, score = minimap.locate(patch, current)

            if score >= minimap.LOST_SCORE and (best is None or hypot(dx, dy) < best[1]):
                best = (index, hypot(dx, dy))

        if best is None:
            return False

        self.index = best[0]
        self.best_distance = None
        return True

    def advance(self):
        last = len(self.route) - 1

        if self.mode == "loop":
            self.index = (self.index + 1) % len(self.route)
        else:
            if not 0 <= self.index + self.direction <= last:
                self.direction = -self.direction
            self.index += self.direction

        self.best_distance = None

    def step(self, frame):
        current = minimap.read(frame, self.region)

        if self.index is None and not self.localize(current):
            return Step("lost")

        dx, dy, score = minimap.locate(self.route.patches[self.index], current)

        if score < minimap.LOST_SCORE:
            # Pushed away by a fight or a long detour, look for any waypoint.
            if not self.localize(current):
                self.index = None
                return Step("lost")

            dx, dy, score = minimap.locate(self.route.patches[self.index], current)

        distance = hypot(dx, dy)
        now = self.clock()

        if distance <= minimap.arrive_distance(current):
            reached = self.index
            self.advance()
            return Step("arrived", waypoint=reached)

        if self.best_distance is None or distance < self.best_distance - 1:
            self.best_distance, self.best_at = distance, now
        elif now - self.best_at > STUCK_TIMEOUT:
            skipped = self.index
            self.advance()
            return Step("arrived", waypoint=skipped)

        screen_x, screen_y = self.matrix @ np.array([dx, dy])
        length = hypot(screen_x, screen_y)

        if length < 1e-6:
            return Step("arrived", waypoint=self.index)

        return Step("walk", direction=(screen_x / length, screen_y / length), waypoint=self.index)


def calibrate(grab, walk, region, wait):
    """
    Measure how a walk on the screen moves the character on the minimap, by walking right
    then down. Needed when the character walks the wrong way while following a route.

    :param grab: Callable giving a frame.
    :param walk: Callable taking a unit screen direction and clicking that way.
    :param region: Minimap region.
    :param wait: Callable waiting for the character to walk.
    :return: 2x2 matrix turning a minimap way into a screen way, as nested lists.
    """
    screen_moves = [(1.0, 0.0), (0.0, 1.0)]
    minimap_moves = []

    for direction in screen_moves:
        before = minimap.patch(minimap.read(grab(), region))
        walk(direction)
        wait()
        dx, dy, score = minimap.locate(before, minimap.read(grab(), region))

        if score < minimap.LOST_SCORE:
            raise RuntimeError("the minimap could not be followed, check its region")

        # The place the character stood is found at (dx, dy), it walked the other way.
        minimap_moves.append((-dx, -dy))

    moves = np.array(minimap_moves, dtype=float).T

    if abs(np.linalg.det(moves)) < 1.0:
        raise RuntimeError("the minimap did not move while the character walked, check its region "
                           "and that nothing blocked the way")

    return (np.array(screen_moves, dtype=float).T @ np.linalg.inv(moves)).tolist()
