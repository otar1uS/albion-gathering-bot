"""
The bot itself: a state machine that finds a resource, proves it is real, goes to it, and
takes everything out of it.

Everything expensive the bot does is guarded by something cheap. Walking to a node costs
twenty seconds and hovering it costs under one, so nothing is walked to before the game
has confirmed it. A charge after the first is given six seconds to start rather than the
full walking timeout, because the character is already standing on the node and the only
question is whether anything is left in it.

The two failures this is built around, both of which the previous bot had:

Nodes were abandoned half chopped. Losing sight of a node was treated as having emptied
it, and the model drops a box every few frames, so the bot walked away from full trees
announcing it had finished them in exactly the time it takes to miss six scans in a row.
Here, losing sight of a node is not the same as emptying it: the last known spot is
clicked again and the game settles it, because a stump does not start a gathering
animation and a tree does.

Most trees are not gatherable. The model was trained on datasets that only ever labelled
real nodes, so scenery was never marked as anything and it finds both kinds equally.
Verification against the game's own hover label is what makes that survivable.
"""

from time import time

from .. import logs
from ..control.input import Stopped
from ..game import tiers
from ..geometry import distance
from ..nav.anti_stuck import Stuck
from .states import ALLOWED, State

log = logs.get("bot")


class Gatherer:
    """Runs the gathering loop until it is told to stop."""

    def __init__(self, config, capture, detector, verifier, controller, navigator,
                 mount, motion, anti_stuck, targets, threats=None, route_player=None,
                 on_frame=None):
        self.config = config
        self.capture = capture
        self.detector = detector
        self.verifier = verifier
        self.controller = controller
        self.navigator = navigator
        self.mount = mount
        self.motion = motion
        self.anti_stuck = anti_stuck
        self.targets = targets
        self.threats = threats
        self.route_player = route_player
        self.on_frame = on_frame

        self.state = State.SCANNING
        self.target = None
        self.empty_scans = 0
        self.last_change = None

        # Nodes already dealt with, as (x, y, when it may be looked at again). Held in
        # screen coordinates, which drift as the character walks, so this is only ever a
        # short term memory to stop the bot picking the same thing twice in a row.
        self.ignored = []

        self.charges = 0
        self.nodes = 0
        self.skipped = 0
        self.unread = 0
        self.started = time()

    # ------------------------------------------------------------------ machinery

    def go(self, state):
        """
        Move to another state, complaining about transitions that make no sense.

        :param state: State to move to.
        """
        if state is self.state:
            return

        if state not in ALLOWED[self.state]:
            log.error("%s -> %s is not a transition the machine has, going anyway",
                      self.state.value, state.value)

        log.debug("%s -> %s", self.state.value, state.value)
        self.state = state
        self.anti_stuck.note_state(state.value)

    def stop(self):
        """Ask the bot to finish, from another thread."""
        self.controller.stop_requested = True

    def run(self):
        """
        Gather until stopped.

        :return: Number of charges taken.
        """
        log.info("gathering %s", ", ".join(str(t) for t in self.targets))

        try:
            while self.state is not State.STOPPING:
                if self.config.window.keep_foreground:
                    # Clicks reach whatever is in front, so the game has to be there.
                    # Seeing no longer depends on it, but acting does.
                    self.capture.focus()

                self.capture.refresh()

                if not self.__in_game():
                    continue

                handler = {
                    State.SCANNING: self.__scan,
                    State.VERIFYING: self.__verify,
                    State.TRAVELLING: self.__travel,
                    State.GATHERING: self.__gather,
                    State.ROAMING: self.__roam,
                    State.RECOVERING: self.__recover,
                }[self.state]

                handler()
        except Stopped as reason:
            log.info("stopped: %s", reason)
        except Exception:
            log.exception("the bot fell over")
            raise
        finally:
            minutes = max((time() - self.started) / 60, 0.01)
            log.info("took %d charges from %d nodes in %.1f minutes (%.1f charges an "
                     "hour)%s", self.charges, self.nodes, minutes,
                     self.charges / minutes * 60,
                     f", skipped {self.skipped} on the tier filter" if self.skipped
                     else "")

        return self.charges

    # ------------------------------------------------------------------ states

    def __look(self, factor=1.0):
        """
        Take a picture, keep the preview fed and the motion history up to date.

        :param factor: Threshold multiplier, see Detector.look.
        :return: (detections, frame).
        """
        detections, frame = self.detector.look(factor)

        # The one place a frame enters the motion history. Adding the same frame twice
        # compares it against itself, which reads as a world where nothing ever moves,
        # and that is exactly the bug this line replaced: every walk measured 0.00 and
        # every reached node was reported as having given nothing.
        change = self.motion.add(frame)
        self.last_change = change
        self.anti_stuck.note_motion(change)

        log.debug("looked: %d detections, frame moved %s (moving over %.1f)",
                  len(detections),
                  "-" if change is None else f"{change:.2f}",
                  self.config.anti_stuck.motion_threshold)

        if self.on_frame is not None:
            self.on_frame(self.detector.draw(frame, detections))

        return detections, frame

    def __in_game(self):
        """
        Check that there is a world on screen to work in, and wait when there is not.

        A login screen, a loading screen between zones and a client that has stopped
        rendering all look the same from here: a frame that does not change at all. They
        are worth telling apart from a quiet forest, because the bot would otherwise
        spend its recovery attempts walking a character that is not there and then stop
        for the wrong reason. This is what it did the first time the account logged out
        mid run.

        Waiting rather than stopping, since a zone change is over in seconds and only a
        real disconnect needs somebody.

        :return: True when the game is being played and the loop may carry on.
        """
        settings = self.config.anti_stuck

        self.__look()

        if not self.motion.frozen(settings.frozen_samples, settings.frozen_threshold):
            return True

        log.warning("the game is not being played: the picture has not changed at all. "
                    "That is a login screen, a loading screen, or a disconnect. Waiting "
                    "up to %.0f minutes for it to come back.",
                    settings.wait_for_game_minutes)

        deadline = time() + settings.wait_for_game_minutes * 60

        while time() < deadline:
            if self.controller.stop_requested:
                self.go(State.STOPPING)
                return False

            self.controller.wait(5.0)
            self.motion.clear()

            for _ in range(settings.frozen_samples + 1):
                self.__look()

            if not self.motion.frozen(settings.frozen_samples, settings.frozen_threshold):
                log.info("the game is back, carrying on")
                self.anti_stuck.note_progress()
                self.go(State.SCANNING)
                return True

        log.error("the game never came back. Log in again and start the bot.")
        self.go(State.STOPPING)

        return False

    def __ignored(self, point):
        """
        Whether a spot has been dealt with recently.

        :param point: (x, y) on screen.
        :return: True when it should be left alone.
        """
        now = time()
        self.ignored = [entry for entry in self.ignored if entry[2] > now]
        radius = self.config.gathering.depleted_radius

        return any(distance(point, (x, y)) <= radius for x, y, _ in self.ignored)

    def __ignore(self, point, seconds):
        """
        Leave a spot alone for a while.

        :param point: (x, y) on screen.
        :param seconds: How long for.
        """
        self.ignored.append((point[0], point[1], time() + seconds))

    def __scan(self):
        """Look for something worth going to."""
        detections, frame = self.__look()
        self.last_frame = frame

        if self.threats is not None and self.threats.check(detections):
            # Whatever was in sight, being stood on by a monster comes first. The world
            # has moved by the time the character is clear, so look again.
            return

        wanted = [found for found in detections
                  if found.profile in self.targets
                  and not self.__ignored(found.screen)
                  and not self.navigator.on_interface(*found.screen)]

        if not wanted:
            self.empty_scans += 1
            self.anti_stuck.note_target(None)

            stuck = self.anti_stuck.diagnose(self.empty_scans)

            self.go(State.RECOVERING if stuck is not Stuck.NONE else State.ROAMING)
            return

        # Nearest first already, then by what the user cares about most when several
        # kinds are being gathered at once.
        wanted.sort(key=lambda found: (-found.profile.priority, found.distance))

        self.target = wanted[0]
        self.empty_scans = 0
        self.anti_stuck.note_target(self.target.screen)

        log.info("found %s at %.2f, %.0fpx away", self.target.label,
                 self.target.confidence, self.target.distance)

        self.go(State.VERIFYING)

    def __relocate(self):
        """
        Find the target again, in a frame taken just now.

        A detection is already a second or so old by the time the cursor has travelled to
        it and the tooltip has had time to appear, and a fox does not wait: the hover
        lands on the grass it was standing on. Looking again and taking the nearest one
        of the same kind costs a scan and rescues most of those.

        :return: A fresh Detection, or None when it has gone.
        """
        detections, frame = self.__look(self.config.vision.tracking_factor)
        self.last_frame = frame

        limit = self.capture.rect.width * self.config.gathering.track_radius
        near = [found for found in detections
                if found.profile is self.target.profile
                and distance(self.target.screen, found.screen) < limit]

        return min(near, key=lambda f: distance(self.target.screen, f.screen)) \
            if near else None

    def __verify(self):
        """Ask the game whether the thing the model found is really a resource."""
        stuck = self.anti_stuck.diagnose(self.empty_scans)

        if stuck is Stuck.LOOPER:
            self.go(State.RECOVERING)
            return

        confirmed = self.verifier.confirm(self.target, self.last_frame)

        if not confirmed:
            # One more go with a freshly located target before writing it off, since the
            # commonest reason a hover finds nothing is that whatever it was aiming at
            # has moved on.
            moved = self.__relocate()

            if moved is not None and moved.screen != self.target.screen:
                log.debug("it has moved to %s, hovering again", moved.screen)
                self.target = moved
                confirmed = self.verifier.confirm(moved, self.last_frame)

        if confirmed:
            named = self.verifier.last_named

            if named is not None:
                keep, why = tiers.wanted(named, self.config)

                if not keep:
                    # A tier the user asked to be left alone. Put aside for as long as an
                    # emptied node, not as long as scenery: it is a perfectly good node
                    # and the rules may be different by the time it comes round again.
                    self.skipped += 1
                    self.unread += 0 if named.known else 1

                    log.info("skipping the %s: %s", named, why)

                    if self.unread and self.unread % 8 == 0:
                        log.warning("%d nodes skipped now because their names could not "
                                    "be read. If that keeps up, either the tier filter "
                                    "is not worth its cost here or the reader needs a "
                                    "wider crop (labels.across / labels.above)",
                                    self.unread)

                    self.__ignore(self.target.screen,
                                  self.config.gathering.depleted_cooldown)
                    self.target = None
                    self.go(State.SCANNING)
                    return

                if named.known:
                    self.unread = 0
                    log.info("it is %s", named)

            self.go(State.TRAVELLING)
            return

        if tiers.restrictive(self.config):
            # Clicking on faith is fine when everything is being gathered, and wrong when
            # a tier was asked for: the whole point of the filter is not to touch the
            # wrong ones, and a node the game would not name is a node whose tier is not
            # known. Better to walk past it than to chop it and find out.
            log.info("no readable name on that %s and a tier filter is set, leaving it",
                     self.target.label)
            self.__ignore(self.target.screen, self.config.verify.scenery_cooldown)
            self.target = None
            self.go(State.SCANNING)
            return

        if self.target.confidence >= self.config.verify.trust_confidence:
            # The hover found no name, and the model is sure enough that the click is
            # worth its cost anyway: a real node starts a gathering animation and scenery
            # does not, so the walk settles what the hover could not. Without this, a
            # verifier wrong for any reason silently turns the whole bot off.
            log.info("no name on hover, but %.2f is sure enough to test the %s by "
                     "clicking it", self.target.confidence, self.target.label)
            self.go(State.TRAVELLING)
            return

        log.info("the game will not name that %s, it is scenery", self.target.label)
        self.__ignore(self.target.screen, self.config.verify.scenery_cooldown)
        self.target = None
        self.go(State.SCANNING)

    def __travel(self):
        """Go to the node, riding if it is far enough to be worth it."""
        target = self.target

        if self.mount.should_ride(target.distance) and not self.mount.mounted:
            if self.mount.ride():
                # Everything moved while the animation played, so where the node was on
                # screen is not where it is now: look again before clicking anything.
                self.go(State.SCANNING)
                return

            # Could not mount right now, and standing still until it can would be slower
            # than walking. Already mounted falls straight through too: the click rides
            # the character to the node and gathering gets it off again by itself.

        self.go(State.GATHERING)

    def __gather(self):
        """
        Take charges out of the node until it gives nothing back.

        :return: None, the state is left through go().
        """
        settings = self.config.gathering
        profile = self.target.profile

        # Two points, and they are not the same one. The click goes at the trunk, because
        # the centre of a tall tree is canopy with gaps the ground shows through and a
        # click through one of those walks the character past the tree. The box centre is
        # what the next detection is matched against, since that is what the model
        # reports. Following the node by its click point instead is what used to send the
        # second charge at the canopy.
        spot = self.target.click
        anchor = self.target.screen

        log.info("gathering the %s", profile)

        charges = 0
        quiet = 0

        for attempt in range(settings.charge_attempts):
            if self.controller.stop_requested:
                self.go(State.STOPPING)
                return

            # The first click is the one that walks the character over and gets the full
            # walking timeout. Every one after it is thrown at a node it is standing on,
            # so waiting twenty seconds to learn that a stump gives nothing back is
            # twenty seconds of standing still.
            arriving = profile.moving_timeout if attempt == 0 else settings.arrival_timeout

            if self.__take_charge(spot, profile, arriving):
                charges += 1
                quiet = 0
                self.mount.note_gathered()
            else:
                # Watching the picture is a good test of having walked somewhere and a
                # thin one for having swung an axe while standing still. Measured at one
                # tree: walking moved the frame 11 to 14, an emptied clearing 0.85 to
                # 1.0, and the gathering itself only 1.2 to 1.8, so whether a charge
                # registers comes down to which side of the threshold an ambient frame
                # happens to fall. The node still being on screen is the sounder signal
                # and it is already being asked for below, so a quiet click is not
                # counted and not acted on either. Only the first one is fatal, because
                # nothing was reached at all, and lost_attempts of them in a row means
                # the clicks are going somewhere that is not a node.
                quiet += 1

                if attempt == 0 or quiet >= settings.lost_attempts:
                    break

            self.controller.wait(settings.charge_delay)

            if attempt == 0:
                # The first charge is the one that walks the character over, and the
                # camera goes with it, so a node picked out 500px away has slid most of
                # that distance across the screen by the time it is reached. Matched
                # against where its box was before the walk it is always further away
                # than track_radius allows, so it was never found again, the click point
                # stayed where the node used to be, and every charge after the first
                # walked the character back to an empty patch of ground. Measured on one
                # tree: five charges, five full frame walks of 11 to 14 mean change
                # against an idle 0.6, one of them the real one. Standing on the node,
                # the node is wherever the character is.
                anchor = self.detector.character()

            found = self.__wait_for_node(anchor, profile)

            if found is None:
                log.info("lost sight of the %s after %d charges", profile, charges)
                break

            spot, anchor = found.click, found.screen

        if charges:
            self.charges += charges
            self.nodes += 1
            self.anti_stuck.note_progress()
            log.info("took %d charge%s from the %s", charges,
                     "" if charges == 1 else "s", profile)
            self.__ignore(spot, settings.depleted_cooldown)
        else:
            # Walked there and nothing started: whatever the model saw, the game does
            # not treat it as a node, so it is put aside for as long as scenery is.
            log.info("got nothing out of that %s, treating it as scenery", profile)
            self.__ignore(spot, self.config.verify.scenery_cooldown)

        self.target = None
        self.controller.wait(profile.delay_between_nodes)
        self.go(State.SCANNING)

    def __take_charge(self, spot, profile, arriving):
        """
        Click a node once and wait for the character to work on it.

        Whether it worked is decided by watching the picture rather than by looking for a
        gathering bar in a fixed place, which is what broke every time the camera zoom
        changed. Walking makes the whole frame move because the camera follows; arriving
        makes it settle. A click that produced neither never reached anything.

        :param spot: (x, y) on screen.
        :param profile: What is being gathered.
        :param arriving: Seconds to wait for the character to get there and start.
        :return: True when the character reached the node and worked on it.
        """
        self.controller.click(*spot)
        self.controller.park(self.capture.rect)

        deadline = time() + arriving
        walked = False

        while time() < deadline:
            if self.controller.stop_requested:
                return False

            self.__look(self.config.vision.tracking_factor)

            if self.motion.moving(self.last_change):
                walked = True

            # Settled again after having moved: the character arrived and is working.
            if walked and self.motion.still_for(2):
                self.controller.wait(self.config.gathering.poll)
                return True

            self.controller.wait(self.config.gathering.poll)

        # Never moved at all. The click did not reach the world, or there is nothing
        # there any more.
        return walked

    def __wait_for_node(self, anchor, profile):
        """
        Look for the node again, giving the model a few frames to produce it.

        Losing sight of a node is not the same as having emptied it: the model drops a box
        every few frames and a tree keeps its shape until the last charge is out of it, so
        one miss is looked at again rather than acted on. What is not done any more is
        clicking where the node was last seen when it has not been seen since arriving.
        That guess is what walked the character back across the clearing, and a look costs
        160ms against the several seconds a wrong walk costs.

        :param anchor: (x, y) on screen to search around.
        :param profile: What is being looked for.
        :return: The Detection, or None once lost_attempts looks have found nothing.
        """
        for _ in range(self.config.gathering.lost_attempts):
            found = self.__find_again(anchor, profile)

            if found is not None:
                log.debug("the %s is still there, at %s", profile, found.screen)
                return found

            self.controller.wait(self.config.gathering.poll)

        return None

    def __find_again(self, spot, profile):
        """
        Look for a node near where it was last seen.

        Held to a lower bar than picking one in the first place: the score of a tree
        wanders by ten points between frames while nothing on screen moves, so following
        it at the same threshold lost it constantly. Only a box already close to where
        the character is standing is accepted this cheaply.

        :param spot: (x, y) the box was centred on when it was last seen.
        :param profile: What is being looked for.
        :return: The Detection where it is now, None when it is not there.
        """
        detections, _ = self.__look(self.config.vision.tracking_factor)
        limit = self.capture.rect.width * self.config.gathering.track_radius

        near = [found for found in detections
                if found.profile is profile
                and distance(spot, found.screen) < limit]

        if not near:
            return None

        return min(near, key=lambda found: distance(spot, found.screen))

    def __roam(self):
        """Cover new ground, along the recorded route when there is one."""
        if self.route_player is not None:
            # The route replaces only the aimless part: it supplies the walking, and
            # everything found along the way still goes through the same verification
            # and gathering as ever.
            self.route_player.step()
            self.go(State.SCANNING)
            return

        detections, _ = self.__look()
        self.navigator.roam(detections)
        self.controller.wait(self.config.navigation.wait)
        self.go(State.SCANNING)

    def __recover(self):
        """Work around whatever has gone wrong, or give up honestly."""
        stuck = self.anti_stuck.diagnose(self.empty_scans)

        if stuck is Stuck.NONE:
            self.go(State.SCANNING)
            return

        if not self.anti_stuck.recover(stuck):
            self.go(State.STOPPING)
            return

        self.empty_scans = 0
        self.target = None
        self.go(State.SCANNING)
