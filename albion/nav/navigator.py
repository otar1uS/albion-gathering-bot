"""
Getting the character from where it is to where it should be.

Albion is driven by clicking the ground, so walking somewhere is one click and then a
wait, and the only feedback available is whether the picture changed. That makes two
things worth care: never clicking on the interface, because that opens a panel rather
than moving anybody, and covering new ground when searching rather than crossing the same
emptied clearing over and over.

The search is an expanding square. A random walk was tried first and spent its time
wandering back over ground it had already stripped; going a few steps one way, turning a
quarter, and lengthening the legs every second turn spirals outwards from wherever it
started and reaches new ground steadily.
"""

from math import cos, radians, sin
from random import uniform

from .. import logs
from ..geometry import distance

log = logs.get("nav.navigator")

# Boxes of the window holding the interface, as fractions (left, top, right, bottom).
# Clicking any of these opens a panel over the game: the portrait and buffs along the
# top, the action bar at the bottom, the minimap and the buttons in the bottom right.
INTERFACE_AREAS = (
    (0.00, 0.00, 0.22, 0.11),
    (0.30, 0.00, 1.00, 0.06),
    (0.00, 0.88, 1.00, 1.00),
    (0.78, 0.72, 1.00, 1.00),
    (0.00, 0.62, 0.20, 1.00),
)


class Navigator:
    """Walks the character about."""

    def __init__(self, config, capture, controller, detector):
        self.config = config
        self.capture = capture
        self.controller = controller
        self.detector = detector

        # The search holds a direction for a few steps so ground is really covered, and
        # the legs grow as it goes so it spirals outwards.
        self.heading = uniform(0, 360)
        self.leg = 2
        self.done = 0
        self.turns = 0

    def on_interface(self, x, y):
        """
        Whether a screen position is over the interface rather than the world.

        :param x: Screen position.
        :param y: Screen position.
        :return: True when clicking there would open a panel.
        """
        rect = self.capture.rect

        for left, top, right, bottom in INTERFACE_AREAS:
            if (rect.left + rect.width * left <= x <= rect.left + rect.width * right
                    and rect.top + rect.height * top <= y
                    <= rect.top + rect.height * bottom):
                return True

        return False

    def walk_to(self, x, y):
        """
        Send the character to a spot on screen.

        :param x: Screen position.
        :param y: Screen position.
        :return: True when the click was made.
        """
        if self.on_interface(x, y):
            log.debug("refused to walk to (%s, %s), that is the interface", x, y)
            return False

        self.controller.click(x, y)

        return True

    def hazards_near(self, point, detections):
        """
        Whether something the bot cannot fight is close to where it wants to go.

        :param point: (x, y) on screen.
        :param detections: Everything the model found.
        :return: True when a hazard is too close.
        """
        limit = self.config.navigation.hazard_distance

        return any(distance(point, found.screen) <= limit
                   for found in detections if not found.gatherable)

    def roam(self, detections=()):
        """
        Take the character somewhere else to look for resources.

        :param detections: What is in sight, so hazards can be avoided.
        :return: True when the character was sent walking.
        """
        settings = self.config.navigation
        rect = self.capture.rect

        if self.done >= self.leg:
            self.done = 0
            self.turns += 1
            self.heading = (self.heading + 90) % 360

            # Every second turn, so the two sides of the square grow together and the
            # path spirals rather than drifting off in one direction.
            if self.turns % 2 == 0:
                self.leg += 1

            log.debug("turning, heading %.0f, legs of %d", self.heading, self.leg)

        # A wobble so a leg blocked by a rock does not have the character pushing into it
        # over and over in exactly the same place.
        heading = radians(self.heading + uniform(-12, 12))
        reach = min(rect.width, rect.height) * settings.radius

        character_x, character_y = self.detector.character()

        # The camera looks at the world from an angle, so a pixel up the screen covers
        # more world than a pixel across it and the vertical part is squashed to match.
        target_x = character_x + cos(heading) * reach
        target_y = character_y + sin(heading) * reach * settings.vertical_squash

        target_x = min(max(target_x, rect.left + rect.width * settings.area_left),
                       rect.left + rect.width * settings.area_right)
        target_y = min(max(target_y, rect.top + rect.height * settings.area_top),
                       rect.top + rect.height * settings.area_bottom)

        target = (int(target_x), int(target_y))

        if self.hazards_near(target, detections):
            # Straight into something unfriendly. Turning early costs one step and is
            # cheaper than the fight.
            log.info("a hazard is that way, turning instead")
            self.done = self.leg
            return False

        self.done += 1

        log.info("nothing in sight, walking to %s", target)

        return self.walk_to(*target)

    def reset(self):
        """Start the search again from wherever the character now is."""
        self.heading = uniform(0, 360)
        self.leg = 2
        self.done = 0
        self.turns = 0
