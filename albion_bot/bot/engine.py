"""
The gathering loop.

    look      grab the game, find the resources
    gather    walk to the closest wanted node, wait for the gathering bar, wait for it to go
    roam      nothing in sight: follow the route, or wait where the character stands

Everything is measured in frame pixels. The camera follows the character, so whenever it
walks the world slides on screen, and the nodes the bot remembers slide with it, see
albion_bot.vision.tracking.
"""

from dataclasses import asdict, dataclass
from math import atan2, cos, radians, sin
from time import monotonic, sleep as _sleep

import cv2 as cv

from albion_bot import geometry
from albion_bot.bot.memory import NodeMemory
from albion_bot.platform.input import FailSafe
from albion_bot.vision import resources, tracking


class Stop(Exception):
    """The bot has to stop, the message says why."""


@dataclass
class Stats:
    state: str = "starting"
    gathered: int = 0
    failed: int = 0
    runtime: float = 0.0


class Bot:
    # Time in second between two looks at the gathering bar.
    BAR_POLLING = 0.2
    # The bar blinks out between two charges of a node, it has to stay gone this long for
    # the node to be over.
    BAR_GRACE = 1.5
    # Pauses of the loop, in second.
    ROUTE_POLLING = 0.15
    IDLE_POLLING = 0.8
    # Time in second between two clicks while walking a route.
    WALK_INTERVAL = 0.8
    # Distance of a walking click from the character, fraction of the frame height.
    WALK_RADIUS = 0.22
    # Distances, fractions of the frame height: a remembered node covers NODE_RADIUS, a
    # monster guards DANGER_RADIUS around it, a node the character reached is within
    # NEAR_RADIUS of it, and a node followed by the camera tracking within TRACK_RADIUS
    # of where the tracking puts it.
    NODE_RADIUS = 0.05
    TRACK_RADIUS = 0.06
    DANGER_RADIUS = 0.16
    NEAR_RADIUS = 0.3
    # A node gathered in less than QUICK_GATHER seconds QUICK_LIMIT times in a row means
    # the bags are full, or too heavy to carry more.
    QUICK_GATHER = 2.0
    QUICK_LIMIT = 3
    # After FAIL_LIMIT nodes in a row could not be reached, the route is walked for
    # MOVE_ON seconds without looking at resources, to get out of wherever it is stuck.
    FAIL_LIMIT = 4
    MOVE_ON = 6.0
    # Time in second the mount takes to be called.
    MOUNT_TIME = 3.0
    # Where the cursor rests while waiting, fraction of the window: the top edge, off
    # every node, so the tooltip of the game does not cover anything.
    PARK = (0.5, 0.015)
    PREVIEW_INTERVAL = 0.4
    FAILSAFE_INTERVAL = 0.5

    def __init__(self, window, pointer, detector, bar, settings, navigator=None, log=print,
                 should_stop=lambda: False, on_stats=None, on_preview=None, clock=monotonic, sleep=_sleep):
        """
        :param window: GameWindow, or anything with grab, focus and to_screen.
        :param pointer: Pointer driving the mouse and the keyboard.
        :param detector: Detector, or anything with detect and knows.
        :param bar: GatherBar, or anything with visible.
        :param settings: albion_bot.config.Settings.
        :param navigator: Navigator of the route to follow, None to stay.
        :param log: Where to tell what happens.
        :param should_stop: Callable returning True once the user wants the bot to stop.
        :param on_stats: Callable given a dict of Stats whenever they change.
        :param on_preview: Callable given the annotated BGR frame, None for no preview.
        """
        self.window = window
        self.pointer = pointer
        self.detector = detector
        self.bar = bar
        self.settings = settings
        self.navigator = navigator
        self.log = log
        self.should_stop = should_stop
        self.on_stats = on_stats
        self.on_preview = on_preview
        self.clock = clock
        self.sleep = sleep

        self.targets = resources.resolve_targets(settings.targets)
        unknown = [str(t) for t in self.targets if not detector.knows(t)]

        if len(unknown) == len(self.targets):
            raise ValueError(f"the model knows none of {', '.join(unknown)}, train it on them first")

        if unknown:
            log(f"The model does not know {', '.join(unknown)}, they will never be found")

        self.memory = NodeMemory(clock=clock)
        self.stats = Stats()
        self.started = None
        self.previous = None
        # Where the node walked to is, followed like the memory while walking to it.
        self.tracked = None
        self.last_walk = -1e9
        self.last_failsafe = -1e9
        self.last_preview = -1e9
        self.move_on_until = -1e9
        self.failed_in_row = 0
        self.quick_in_row = 0
        self.mount_due = False
        self.lost = False

    # ---------------------------------------------------------------- helpers

    def character(self, frame):
        return geometry.point(frame.shape, self.settings.character)

    def set_state(self, state):
        if state != self.stats.state:
            self.stats.state = state
            self.publish()

    def publish(self):
        if self.started is not None:
            self.stats.runtime = self.clock() - self.started

        if self.on_stats is not None:
            self.on_stats(asdict(self.stats))

    def check(self):
        """
        :raise Stop: When the bot has to stop.
        """
        if self.should_stop():
            raise Stop("asked to")

        if self.settings.max_minutes and self.clock() - self.started >= self.settings.max_minutes * 60:
            raise Stop(f"{self.settings.max_minutes:g} minutes are over")

        # On Wayland reading the cursor costs a process, it is not done on every call.
        if self.clock() - self.last_failsafe >= self.FAILSAFE_INTERVAL:
            self.last_failsafe = self.clock()
            self.pointer.check_failsafe()

    def park(self, shape):
        self.pointer.move(*self.window.to_screen(shape[1] * self.PARK[0], shape[0] * self.PARK[1], shape))

    def observe(self, frame):
        """
        Follow the camera from the last frame seen to this one, sliding the remembered
        nodes and the node walked to with the world. Called on every frame grabbed: two
        frames a fraction of a second apart always line up, two far apart may not.

        :return: (dx, dy) the world moved by, None when it cannot be told.
        """
        previous, self.previous = self.previous, frame

        if previous is None:
            return None

        dx, dy, response = tracking.camera_shift(previous, frame, self.character(previous))

        if response < tracking.MIN_RESPONSE:
            # Better forget than remember them at the wrong place.
            self.memory.clear()
            self.tracked = None
            return None

        self.memory.shift(dx, dy)

        if self.tracked is not None:
            self.tracked = (self.tracked[0] + dx, self.tracked[1] + dy)

        return dx, dy

    def grab(self):
        frame = self.window.grab()
        self.observe(frame)
        return frame

    # ---------------------------------------------------------------- the loop

    def run(self):
        self.started = self.clock()
        self.log(f"Gathering {', '.join(str(t) for t in self.targets)}"
                 f"{f', following the route {self.navigator.route.name}' if self.navigator else ''}")

        if not self.window.focus():
            self.log("Could not bring the game in front, click on it")

        self.sleep(0.3)

        try:
            while True:
                self.check()
                # Before anything looks at the memory, it has to match this frame.
                frame = self.grab()
                detections = self.detector.detect(frame)
                target = None if self.clock() < self.move_on_until else self.choose(frame, detections)
                self.preview(frame, detections, target)

                if target is not None:
                    self.gather(frame, target)
                else:
                    self.roam(frame, detections)

        except Stop as e:
            self.log(f"Stopped: {e}")
        except FailSafe as e:
            self.log(f"Stopped: {e}")
        finally:
            self.set_state("stopped")
            self.publish()

    def choose(self, frame, detections):
        """
        :return: The closest wanted node, not gathered yet and not guarded, None when none.
        """
        height = frame.shape[0]
        character = self.character(frame)
        monsters = [d for d in detections if d.profile is resources.MONSTER]
        candidates = []

        for detection in detections:
            if detection.profile not in self.targets:
                continue

            if self.memory.contains(*detection.center, self.NODE_RADIUS * height):
                continue

            if self.settings.avoid_monsters and any(
                    geometry.distance(m.center, detection.center) < self.DANGER_RADIUS * height for m in monsters):
                continue

            candidates.append(detection)

        if not candidates:
            return None

        return min(candidates, key=lambda d: geometry.distance(d.center, character))

    def wait_for_bar(self, timeout):
        """
        :return: The first frame showing the gathering bar, None when it never showed up.
        """
        start = self.clock()

        while self.clock() - start < timeout:
            self.check()
            self.sleep(self.BAR_POLLING)
            frame = self.grab()

            if self.bar.visible(frame):
                return frame

        return None

    def find_node(self, frame, picture, target, near=None):
        """
        Find the node walked to in a later frame: where the camera tracking says it is,
        refined by the way it looks.

        :param near: (x, y) the node has to be close to, the character once reached.
        :return: (x, y) of the node, None when it cannot be told.
        """
        height = frame.shape[0]
        guess = self.tracked

        # Tracking that puts a reached node away from the character went wrong.
        if near is not None and guess is not None and geometry.distance(guess, near) > self.NEAR_RADIUS * height:
            guess = None

        same = [d for d in self.detector.detect(frame) if d.profile is target.profile]

        if guess is not None:
            # Tracking is right to a few pixels, looking further would pick a neighbour
            # that looks alike and mark it as gathered in place of this one.
            found = tracking.reidentify(picture, same, frame, near=guess, max_distance=self.TRACK_RADIUS * height)
            return found[0].center if found is not None else guess

        if near is not None:
            found = tracking.reidentify(picture, same, frame, near=near, max_distance=self.NEAR_RADIUS * height)
            return found[0].center if found is not None else near

        return None

    def gather(self, frame, target):
        """
        Walk to a node and gather it until it is empty.

        :return: True when it has been gathered.
        """
        profile = target.profile
        picture = tracking.node_picture(frame, target)
        self.tracked = target.center
        hidden = False

        self.set_state(f"walking to a {profile}")

        try:
            if self.settings.hide_hud:
                self.pointer.hotkey("alt", "h")
                hidden = True

            self.pointer.click(*self.window.to_screen(*target.click_point, frame.shape))
            self.park(frame.shape)

            arrived = self.wait_for_bar(profile.moving_timeout)

            if arrived is None:
                position = self.find_node(self.grab(), picture, target)

                if position is not None:
                    self.memory.add(*position)

                self.stats.failed += 1
                self.failed_in_row += 1
                self.log(f"Could not reach the {profile}, skipping it")

                if self.failed_in_row >= self.FAIL_LIMIT:
                    self.failed_in_row = 0

                    if self.navigator is not None:
                        self.move_on_until = self.clock() + self.MOVE_ON
                        self.log("Too many nodes out of reach here, moving on along the route")
                    else:
                        self.log("Too many nodes out of reach, check the gathering bar calibration")

                self.publish()
                return False

            self.failed_in_row = 0
            position = self.find_node(arrived, picture, target, near=self.character(arrived))
            self.set_state(f"gathering a {profile}")

            began = last_seen = self.clock()

            while True:
                if self.clock() - began >= profile.gathering_timeout:
                    self.log(f"The {profile} took more than {profile.gathering_timeout:g}s, leaving it")
                    break

                self.check()
                self.sleep(self.BAR_POLLING)

                if self.bar.visible(self.grab()):
                    last_seen = self.clock()
                elif self.clock() - last_seen >= self.BAR_GRACE:
                    break

            duration = last_seen - began
            # Reached means the character stands next to it, the memory has to hold it
            # even when it was not found again.
            self.memory.add(*(position or self.character(arrived)))
            self.stats.gathered += 1
            self.mount_due = True
            self.log(f"Gathered a {profile} in {duration:.0f}s, {self.stats.gathered} so far")
            self.publish()

            self.quick_in_row = self.quick_in_row + 1 if duration < self.QUICK_GATHER else 0

            if self.quick_in_row >= self.QUICK_LIMIT:
                raise Stop(f"the last {self.QUICK_LIMIT} nodes ended right away, the bags are most likely "
                           f"full or too heavy")

            return True

        finally:
            if hidden:
                self.pointer.hotkey("alt", "h")

    def roam(self, frame, detections):
        """
        Nothing to gather in sight: walk the route, or wait for something to grow back.
        """
        if self.navigator is None:
            self.set_state("waiting for resources")
            self.sleep(self.IDLE_POLLING)
            return

        if self.mount_due and self.settings.mount_key:
            self.mount_due = False
            self.set_state("mounting")
            self.pointer.press(self.settings.mount_key)
            self.sleep(self.MOUNT_TIME)
            return

        self.mount_due = False
        step = self.navigator.step(frame)

        if step.status == "lost":
            if not self.lost:
                self.log("Off the route, no waypoint in sight: walk the character back onto it")
            self.lost = True
            self.set_state("lost, waiting to be put back on the route")
            self.sleep(1.0)
            return

        if self.lost:
            self.log("Back on the route")

        self.lost = False
        self.set_state(f"following {self.navigator.route.name}, waypoint "
                       f"{self.navigator.index + 1}/{len(self.navigator.route)}")

        if step.status == "walk" and self.clock() - self.last_walk >= self.WALK_INTERVAL:
            x, y = self.walk_point(frame, step.direction, detections)
            self.pointer.click(*self.window.to_screen(x, y, frame.shape))
            self.last_walk = self.clock()

        self.sleep(self.ROUTE_POLLING)

    def walk_point(self, frame, direction, detections):
        """
        Where to click to walk that way, off every node and monster: a click on them
        would gather or attack instead of walking.

        :return: (x, y) in the frame.
        """
        character = self.character(frame)
        radius = self.WALK_RADIUS * frame.shape[0]
        angle = atan2(direction[1], direction[0])
        height, width = frame.shape[:2]

        for turn in (0, 15, -15, 30, -30, 45, -45):
            x = character[0] + cos(angle + radians(turn)) * radius
            y = character[1] + sin(angle + radians(turn)) * radius
            x, y = min(max(x, 1), width - 2), min(max(y, 1), height - 2)

            if not any(self.near_box(d, x, y) for d in detections):
                return x, y

        return character[0] + cos(angle) * radius, character[1] + sin(angle) * radius

    @staticmethod
    def near_box(detection, x, y):
        """
        :return: True when a click at (x, y) could land on the detected thing, which the
                 game takes in a bit wider than the box drawn around it.
        """
        margin_x = max(detection.width * 0.35, 12)
        margin_y = max(detection.height * 0.35, 12)

        return (detection.x1 - margin_x <= x <= detection.x2 + margin_x
                and detection.y1 - margin_y <= y <= detection.y2 + margin_y)

    # ---------------------------------------------------------------- preview

    def preview(self, frame, detections, target):
        if self.on_preview is None or self.clock() - self.last_preview < self.PREVIEW_INTERVAL:
            return

        self.last_preview = self.clock()
        shown = frame.copy()
        thickness = max(frame.shape[0] // 400, 1)

        for detection in detections:
            if detection.profile is resources.MONSTER:
                color = (0, 0, 255)
            elif detection.profile in self.targets:
                color = (0, 220, 0)
            else:
                color = (160, 160, 160)

            cv.rectangle(shown, detection.box()[:2], detection.box()[2:], color, thickness)
            cv.putText(shown, f"{detection.label} {detection.confidence:.2f}",
                       (int(detection.x1), int(detection.y1) - 6), cv.FONT_HERSHEY_SIMPLEX,
                       0.5 * thickness, color, thickness)

        for x, y in self.memory.positions():
            cv.drawMarker(shown, (int(x), int(y)), (0, 0, 255), cv.MARKER_TILTED_CROSS, 24, thickness + 1)

        character = self.character(frame)
        cv.drawMarker(shown, (int(character[0]), int(character[1])), (255, 255, 255), cv.MARKER_DIAMOND,
                      24, thickness + 1)

        if target is not None:
            cv.line(shown, (int(character[0]), int(character[1])),
                    tuple(int(v) for v in target.click_point), (0, 255, 255), thickness + 1)

        self.on_preview(shown)
