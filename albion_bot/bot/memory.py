"""
Nodes gathered or given up on, so the bot does not walk back to them.

Positions are frame pixels, and the camera follows the character: every time it walks,
the world slides on screen and so must every remembered node, see shift.
"""

from math import hypot
from time import monotonic


class NodeMemory:
    # Nodes grow back after a few minutes.
    COOLDOWN = 240.0

    def __init__(self, cooldown=COOLDOWN, clock=monotonic):
        self.cooldown = cooldown
        self.clock = clock
        self.nodes = []

    def add(self, x, y, cooldown=None):
        self.nodes.append([x, y, self.clock() + (self.cooldown if cooldown is None else cooldown)])

    def shift(self, dx, dy):
        """
        :param dx: How far the world moved on screen.
        """
        for node in self.nodes:
            node[0] += dx
            node[1] += dy

    def clear(self):
        self.nodes.clear()

    def contains(self, x, y, radius):
        now = self.clock()
        self.nodes = [node for node in self.nodes if node[2] > now]
        return any(hypot(x - nx, y - ny) <= radius for nx, ny, _ in self.nodes)

    def positions(self):
        now = self.clock()
        return [(x, y) for x, y, until in self.nodes if until > now]

    def __len__(self):
        return len(self.positions())
