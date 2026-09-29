"""
A small Albion for the tests: a world bigger than the screen, a camera following the
character, nodes with charges, a gathering bar blinking between two charges, a minimap
scrolling under an arrow, and time that only passes when the bot sleeps.
"""

from math import hypot

import cv2 as cv

from albion_bot.geometry import Rect
from albion_bot.vision import resources
from albion_bot.vision.detector import Detection

W, H = 960, 540
CHARACTER = (0.5, 0.41)
BAR = (430, 290, 530, 304)
BAR_BOX = (425 / W, 286 / H, 110 / W, 22 / H)
MINIMAP = (0.80, 0.70, 0.18, 0.28)
# Minimap pixels per world pixel.
MINIMAP_SCALE = 0.25


class Node:
    def __init__(self, x, y, charges, profile=resources.TREE, ident=0):
        self.x, self.y = x, y
        self.charges = charges
        self.profile = profile
        self.ident = ident

    @property
    def alive(self):
        return self.charges > 0


class FakeGame:
    SPEED = 150.0  # world pixels per second
    REACH = 30.0  # the character stops that far from a node
    CHARGE = 1.0  # seconds per charge, the bar is on
    BETWEEN = 0.4  # seconds between two charges, the bar is off

    def __init__(self, world, start, nodes, stumps_detected=True):
        self.world = world
        self.x, self.y = start
        self.nodes = nodes
        self.stumps_detected = stumps_detected
        self.time = 0.0
        self.destination = None
        self.target = None
        self.gathering = None
        self.phase = 0.0
        self.bar_on = False
        self.clicks = []
        self.bad_clicks = 0
        self.rect = Rect(0, 0, W, H)
        # Rendered minimap of the whole world, the arrow drawn on top of each view.
        small = cv.resize(world, None, fx=MINIMAP_SCALE, fy=MINIMAP_SCALE, interpolation=cv.INTER_AREA)
        self.minimap_world = cv.applyColorMap(cv.cvtColor(small, cv.COLOR_BGR2GRAY), cv.COLORMAP_OCEAN)

    # ---------------------------------------------------------------- window

    def origin(self):
        return int(round(self.x - W * CHARACTER[0])), int(round(self.y - H * CHARACTER[1]))

    def grab(self):
        left, top = self.origin()
        frame = self.world[top:top + H, left:left + W].copy()

        for node in self.nodes:
            sx, sy = int(node.x - left), int(node.y - top)

            if node.alive:
                color = ((node.ident * 70) % 255, 200, (node.ident * 130) % 255)
                cv.circle(frame, (sx, sy - 15), 18, color, -1)
                cv.line(frame, (sx - 10, sy - 25), (sx + 10, sy - 5), (0, 0, 0), 3)
            else:
                cv.rectangle(frame, (sx - 6, sy - 10), (sx + 6, sy), (90, 90, 90), -1)

        cv.circle(frame, (int(W * CHARACTER[0]), int(H * CHARACTER[1])), 14, (0, 0, 220), -1)
        cv.rectangle(frame, (0, 0), (W, 24), (30, 30, 30), -1)

        if self.bar_on:
            x1, y1, x2, y2 = BAR
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv.rectangle(frame, (x1 + 2, y1 + 2), (x1 + 66, y2 - 2), (0, 180, 0), -1)

        self.draw_minimap(frame)
        return frame

    def draw_minimap(self, frame):
        mx, my = int(W * MINIMAP[0]), int(H * MINIMAP[1])
        mw, mh = int(W * MINIMAP[2]), int(H * MINIMAP[3])
        cx, cy = int(self.x * MINIMAP_SCALE), int(self.y * MINIMAP_SCALE)
        view = self.minimap_world[cy - mh // 2:cy - mh // 2 + mh, cx - mw // 2:cx - mw // 2 + mw].copy()
        cv.circle(view, (mw // 2, mh // 2), 4, (255, 255, 255), -1)
        frame[my:my + mh, mx:mx + mw] = view

    def focus(self):
        return True

    def to_screen(self, x, y, shape):
        return int(round(x)), int(round(y))

    # ---------------------------------------------------------------- model

    def detect(self, frame):
        left, top = self.origin()
        found = []

        for node in self.nodes:
            if not node.alive and not self.stumps_detected:
                continue

            sx, sy = node.x - left, node.y - top

            if 20 <= sx <= W - 20 and 40 <= sy <= H - 10:
                found.append(Detection(sx - 20, sy - 40, sx + 20, sy + 10, 0.9, node.profile.name, node.profile))

        return found

    def knows(self, profile):
        return True

    # ---------------------------------------------------------------- mouse

    def click(self, sx, sy):
        left, top = self.origin()
        wx, wy = sx + left, sy + top
        self.clicks.append((wx, wy))
        self.gathering = None
        self.bar_on = False

        node = next((n for n in self.nodes if hypot(n.x - wx, n.y - wy) < 25), None)

        if node is not None:
            if not node.alive:
                self.bad_clicks += 1

            self.target = node
            length = hypot(self.x - node.x, self.y - node.y) or 1
            self.destination = (node.x + (self.x - node.x) / length * self.REACH,
                                node.y + (self.y - node.y) / length * self.REACH)
        else:
            self.target = None
            self.destination = (wx, wy)

    def move(self, x, y):
        pass

    def press(self, key):
        pass

    def hotkey(self, *keys):
        pass

    def check_failsafe(self):
        pass

    # ---------------------------------------------------------------- time

    def sleep(self, seconds):
        step = 0.05

        while seconds > 1e-9:
            dt = min(step, seconds)
            self.advance(dt)
            seconds -= dt

    def clock(self):
        return self.time

    def advance(self, dt):
        self.time += dt

        if self.destination is not None:
            dx, dy = self.destination[0] - self.x, self.destination[1] - self.y
            length = hypot(dx, dy)

            if length <= self.SPEED * dt:
                self.x, self.y = self.destination
                self.destination = None

                if self.target is not None and self.target.alive:
                    self.gathering, self.phase, self.bar_on = self.target, 0.0, True
            else:
                self.x += dx / length * self.SPEED * dt
                self.y += dy / length * self.SPEED * dt

        if self.gathering is not None:
            self.phase += dt

            if self.bar_on and self.phase >= self.CHARGE:
                self.gathering.charges -= 1
                self.phase, self.bar_on = 0.0, False

                if not self.gathering.alive:
                    self.gathering = None
            elif not self.bar_on and self.phase >= self.BETWEEN:
                self.phase, self.bar_on = 0.0, True
