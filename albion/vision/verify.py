"""
Asking the game whether something the model found is really a resource.

This is the single most valuable thing the bot does, and it exists because of a fact
about the training data that no amount of tuning could get around: every dataset the
weights learned from labelled only gatherable trees, so decorative ones were never marked
as anything. A forest is full of both and they are the same tree to look at, so the model
finds both and is right about neither. Measured over six trees it picked out of the
farming zone, one was a node and five were scenery.

The game itself knows the difference and says so for free. Hovering a real node writes its
name over it in near white and hovering scenery writes nothing: the node put 42 more
bright pixels on screen, the five decoys put none between them. So instead of walking
twenty seconds to a tree to find out, the cursor is rested on it for under a second first.
"""

import cv2 as cv

from .. import logs

log = logs.get("vision.verify")


class Verifier:
    """Confirms a detection against the game before the bot commits to it."""

    def __init__(self, config, capture, controller, labels=None):
        self.config = config
        self.capture = capture
        self.controller = controller
        self.labels = labels

        # What the last confirmed hover turned out to be, so the caller can apply tier
        # and enchantment rules without paying for a second hover.
        self.last_named = None

    def confirm(self, detection, clean):
        """
        Check that the game names a resource where the model believes there is one.

        :param detection: What the model found.
        :param clean: Frame taken with the cursor parked away, to compare against.
        :return: True when the game named something under the cursor.
        """
        settings = self.config.verify
        self.last_named = None

        if not settings.enabled:
            return True

        image_size = self.config.vision.image_size

        # The trunk rather than the box centre: the centre of a tall tree is canopy, and
        # the game only answers a cursor resting on the node itself.
        spot = detection.click
        x, y = self.capture.rect.to_image(*spot, image_size)

        left = max(x - settings.radius, 0)
        top = max(y - settings.radius, 0)
        right = min(x + settings.radius, image_size)
        bottom = min(y + settings.radius, image_size)

        def lit(frame):
            grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            return int((grey[top:bottom, left:right] > settings.bright).sum())

        before = lit(clean)

        self.controller.move(*spot)
        self.controller.wait(settings.settle)

        from .detector import process

        after = lit(process(self.capture.grab(), image_size))
        bright_says = after - before >= settings.jump

        # Reading the writing beats counting how much of it lit up, so when the reader is
        # available it is asked first and its answer wins. Counting bright pixels was
        # measured saying "scenery" for nodes that then gave up sixteen charges: the
        # writing is drawn over whatever is behind it, and over a pale patch of ground it
        # barely brightens anything. A name that matches a resource cannot be wrong in
        # that direction, and it carries the tier as well.
        named = bright_says

        if self.labels is not None and self.config.labels.enabled:
            self.last_named = self.labels.read(spot)

            if self.last_named is not None and self.last_named.known:
                named = True
            elif bright_says:
                # Something lit up but could not be read. Still a node, just an unknown
                # one, which the tier rules treat as "keep unless told otherwise".
                named = True

        # Parked again whatever the answer, so the tooltip the game draws under the
        # cursor does not cover the next look at the world.
        self.controller.park(self.capture.rect)

        # Logged at info rather than debug on purpose. This is the one judgement the bot
        # makes that silently decides whether it does any work at all, and a threshold
        # that has drifted looks exactly like a zone with nothing left in it. The numbers
        # say which: a real node measured +42 against a jump of 15, scenery measured 0.
        log.info("%s at %s: %+d bright (needs %+d)%s -> %s", detection.label, spot,
                 after - before, settings.jump,
                 f", read as {self.last_named}" if self.last_named else "",
                 "gatherable" if named else "scenery")

        return named
