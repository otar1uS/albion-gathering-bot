"""
Riding between nodes and getting off to gather.

Mounting in Albion is a channel: the key starts it, a few seconds pass, and any movement
at all interrupts it. The character is also thrown off automatically when it gathers, so
the bot never has to dismount deliberately, only to notice that it is no longer mounted.

There is no reliable way to read "mounted" off the screen without a template for the
mount bar, which is the kind of thing that went stale on the previous bot every time the
camera zoom changed. So the state is tracked rather than read, and every assumption is
cheap to be wrong about: believing it is mounted when it is not costs a walk, and
believing it is not when it is costs one wasted keypress.

Whether riding is worth it at all depends on the distance. Mounting takes about four
seconds and gathering throws the character off again, so for a node a few steps away the
mount costs more than the walk saves. Only trips past `ride_beyond` are ridden.
"""

from time import time

from .. import logs

log = logs.get("game.mount")


class MountManager:
    """Keeps track of whether the character is riding, and decides when it should be."""

    def __init__(self, config, controller, motion):
        self.config = config
        self.controller = controller
        self.motion = motion

        self.mounted = False
        self.last_attempt = 0.0
        self.failures = 0

    def should_ride(self, distance):
        """
        Whether a trip is long enough to be worth mounting for.

        :param distance: How far away the target is, in screen pixels.
        :return: True when the character should ride.
        """
        if not self.config.mount.enabled:
            return False

        return distance >= self.config.mount.ride_beyond

    def ride(self):
        """
        Get on the mount and wait for the animation.

        The mount only takes if the character is standing still, so this does not fight
        a walk already under way: the caller mounts first and then clicks where it is
        going.

        :return: True when the character is believed to be mounted.
        """
        settings = self.config.mount

        if self.mounted:
            return True

        if time() - self.last_attempt < settings.retry_after:
            return False

        self.last_attempt = time()

        log.info("mounting")
        self.controller.press(self.config.input.mount_key)
        self.controller.wait(settings.mount_time)

        # The animation is interrupted by movement, and the honest signal that it
        # finished is that the picture settled and then changed as the mount appears.
        # Neither is reliable enough to gate on, so the attempt is believed and the cost
        # of being wrong is one walk at running speed.
        self.mounted = True
        self.failures = 0

        return True

    def note_gathered(self):
        """
        Gathering throws the character off, so the bot stops believing it is riding.
        """
        if self.mounted:
            log.debug("gathering dismounted the character")

        self.mounted = False

    def note_interrupted(self):
        """
        Something interrupted the mount, so whatever was believed is no longer true.
        """
        self.mounted = False
        self.failures += 1

        if self.failures >= 3:
            log.warning("mounting has failed %d times, riding is off for this run. "
                        "Check the mount key in the settings, it is %r",
                        self.failures, self.config.input.mount_key)
            self.config.mount.enabled = False
