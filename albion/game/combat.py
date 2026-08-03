"""
Staying alive around monsters.

The bot cannot fight. It has no way to read its health, its cooldowns or its damage, so
any fight it starts is one it cannot reason about, and the honest version of a
"counterattack system" for a gatherer is not being in the fight at all. What it can do
well is notice: the model already recognises monsters so the roaming can avoid them, and
the same boxes say when one has come to the character rather than the other way round.

A monster standing next to the character for consecutive looks means it has almost
certainly aggroed, and standing still swinging an axe at a tree while being bitten is the
worst possible answer. Walking firmly away is the best one available, and against the
leashed monsters of gathering zones it works: they give up at the edge of their patch.
"""

from .. import logs
from ..geometry import distance

log = logs.get("game.combat")


class Threats:
    """Watches for monsters that have come too close, and gets the character out."""

    # Looks in a row a monster has to be close for before the bot reacts. One frame is
    # the model blinking a box into existence; two in a row is something really there.
    PATIENCE = 2

    def __init__(self, config, capture, navigator, controller, detector):
        self.config = config
        self.capture = capture
        self.navigator = navigator
        self.controller = controller
        self.detector = detector
        self.streak = 0
        self.fled = 0

    def check(self, detections):
        """
        Look for trouble and step away from it.

        :param detections: Everything the model found this look.
        :return: True when the character was sent away and this pass should rescan.
        """
        character = self.detector.character()
        limit = self.config.navigation.hazard_distance

        close = [found for found in detections
                 if not found.gatherable and distance(found.screen, character) <= limit]

        if not close:
            self.streak = 0
            return False

        self.streak += 1

        if self.streak < self.PATIENCE:
            return False

        threat = min(close, key=lambda found: distance(found.screen, character))

        # Straight away from it: the character's position mirrored through the threat's,
        # clamped to the part of the window that is world rather than interface.
        rect = self.capture.rect
        settings = self.config.navigation

        away_x = character[0] + (character[0] - threat.screen[0]) * 2
        away_y = character[1] + (character[1] - threat.screen[1]) * 2

        away_x = min(max(away_x, rect.left + rect.width * settings.area_left),
                     rect.left + rect.width * settings.area_right)
        away_y = min(max(away_y, rect.top + rect.height * settings.area_top),
                     rect.top + rect.height * settings.area_bottom)

        self.streak = 0
        self.fled += 1

        log.warning("a %s is on top of the character, moving away", threat.label)
        self.navigator.walk_to(int(away_x), int(away_y))
        self.controller.wait(self.config.navigation.wait)

        return True
