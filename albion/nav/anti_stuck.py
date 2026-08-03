"""
Noticing that the bot has stopped getting anywhere, and doing something about it.

A gathering bot fails quietly rather than loudly. It does not crash; it stands against a
rock clicking through it, or walks a circle between two clearings it has already stripped,
or waits on a node that is never going to give anything. Each of those looks like a
working bot to anything that only checks for exceptions, and each of them wastes a night.

The three shapes are worth naming because the recovery differs:

  Repeater  the same action over and over with nothing changing. Standing against
            scenery clicking past it is this one.
  Wanderer  busy, moving, covering ground, finding nothing. An emptied zone is this.
  Looper    alternating between a few states without settling. Walking to a node,
            failing to reach it, picking the same node again is this.

Recovery escalates rather than repeating: a nudge, then a longer walk somewhere else,
then a full reset of the search. If none of it helps the bot says so and stops, because a
bot that cannot tell it is beaten will keep going until morning.
"""

from collections import deque
from enum import Enum
from random import uniform
from time import time

from .. import logs

log = logs.get("nav.anti_stuck")


class Stuck(Enum):
    """Nothing wrong."""

    NONE = "none"
    REPEATER = "repeater"
    WANDERER = "wanderer"
    LOOPER = "looper"


class AntiStuck:
    """Watches for the bot going nowhere and escalates a way out."""

    def __init__(self, config, capture, controller, navigator, motion):
        self.config = config
        self.capture = capture
        self.controller = controller
        self.navigator = navigator
        self.motion = motion

        self.recent_targets = deque(maxlen=8)
        self.recent_states = deque(maxlen=12)
        self.moving_since = time()
        self.attempts = 0
        self.harvested = 0

    def note_state(self, state):
        """
        Record which state the bot is in, for spotting loops.

        :param state: State name.
        """
        self.recent_states.append(state)

    def note_target(self, point):
        """
        Record what the bot decided to go for.

        :param point: (x, y) on screen, or None.
        """
        self.recent_targets.append(point)

    def note_progress(self):
        """Something actually worked, so the counters start again."""
        self.harvested += 1
        self.attempts = 0
        self.moving_since = time()
        self.recent_targets.clear()
        self.recent_states.clear()

    def note_motion(self, change):
        """
        Record how much the picture moved.

        :param change: Mean change from Motion.add.
        """
        if self.motion.moving(change):
            self.moving_since = time()

    def diagnose(self, empty_scans):
        """
        Work out whether the bot is getting anywhere.

        :param empty_scans: How many scans in a row have found nothing.
        :return: Which kind of stuck, Stuck.NONE when all is well.
        """
        settings = self.config.anti_stuck

        # The screen has not changed while the bot believed it was walking somewhere.
        if time() - self.moving_since > settings.stuck_after:
            return Stuck.REPEATER

        # Going for the same spot over and over without ever finishing with it.
        targets = [t for t in self.recent_targets if t is not None]

        if len(targets) >= 4:
            first = targets[-4]
            if all(abs(t[0] - first[0]) < 30 and abs(t[1] - first[1]) < 30
                   for t in targets[-4:]):
                return Stuck.LOOPER

        # Moving about happily and finding nothing anywhere.
        if empty_scans >= settings.empty_scans_before_recovery:
            return Stuck.WANDERER

        return Stuck.NONE

    def recover(self, kind):
        """
        Try to get out of it, harder each time.

        :param kind: What diagnose decided.
        :return: False when the bot has run out of ideas and should stop.
        """
        settings = self.config.anti_stuck
        self.attempts += 1

        if self.attempts > settings.max_attempts:
            log.error("tried to get unstuck %d times and nothing worked, stopping rather "
                      "than repeating myself until morning", self.attempts - 1)
            return False

        log.warning("stuck (%s), recovery attempt %d of %d", kind.value, self.attempts,
                    settings.max_attempts)

        rect = self.capture.rect

        if self.attempts == 1:
            # A short step sideways, which is enough to get off a rock the character has
            # walked into and is now clicking straight through.
            self.__step(rect, 0.16)
        elif self.attempts == 2:
            # Further, and the other way, in case the first step went back into it.
            self.__step(rect, 0.30)
        else:
            # Give up on the local picture entirely and start the search somewhere new.
            log.warning("starting the search over from here")
            self.navigator.reset()
            self.__step(rect, 0.34)

        self.moving_since = time()
        self.recent_targets.clear()
        self.motion.clear()

        return True

    def __step(self, rect, reach):
        """
        Walk somewhere nearby, chosen away from the middle so it is never a no-op.

        :param rect: The game window.
        :param reach: How far, as a fraction of the window.
        """
        # Kept inside the part of the window that is world rather than interface, and
        # away from dead centre so the click is never on the character's own feet.
        x = rect.left + rect.width * uniform(0.5 - reach, 0.5 + reach)
        y = rect.top + rect.height * uniform(0.28, 0.55)

        log.info("nudging to (%d, %d)", x, y)
        self.navigator.walk_to(int(x), int(y))
        self.controller.wait(self.config.navigation.wait)
