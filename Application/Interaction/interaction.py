from Application.Albion import bar
from Application.Albion.detection import AlbionDetection
from Application.Interaction import pointer
from math import cos, radians, sin, sqrt
from random import uniform
from time import sleep, time
import cv2 as cv


class Gathering:
    def __init__(self, x, y, resource):
        self.x = x
        self.y = y
        self.resource = resource

    def __str__(self):
        return f"{self.resource} at ({self.x}x, {self.y}y)"


class Interaction:
    # Time in second between two reads of the bar, keeps the CPU quiet.
    BAR_POLLING = 0.2

    # Time in second between two looks at where the mouse is, see __wait_polling.
    FAILSAFE_POLLING = 1.0

    # A gathered node stays on screen for a while, so it is skipped during that
    # time to stop the bot from clicking the same empty node over and over. The
    # cooldown has to stay above the timeouts of a profile, otherwise a node comes
    # back while the bot is still busy giving up on the next one.
    DEPLETED_RADIUS = 60
    DEPLETED_COOLDOWN = 120

    # Where the cursor is parked between two nodes, so the tooltip of the game stops
    # covering what the model is looking at. As a fraction of the window, and kept well
    # clear of the top left corner: that corner is the panic button, and parking ten
    # pixels away from it had the bot stopping itself every few minutes.
    PARKING_X, PARKING_Y = 0.03, 0.10

    # A node is done when the model stops finding it, a chopped tree being a stump and
    # an emptied rock a hole, neither of which the model was ever shown. One frame
    # without it is not enough, the model drops a node here and there, so several in a
    # row are asked for.
    #
    # The radius is a fraction of the window rather than a number of pixels, because the
    # camera follows the character walking to the node and the node slides right across
    # the screen on the way. A fixed 170 pixels was a fifteenth of the way across a 2560
    # wide screen: the node left it within two seconds and every node looked gathered
    # the moment it was clicked.
    TRACK_RADIUS = 0.18
    TRACK_MISSES = 6
    TRACK_POLLING = 0.5

    # Finding a node again is held to a lower bar than picking one in the first place.
    # The confidence the model gives one tree wanders by a good ten points from frame to
    # frame while nothing on screen moves, measured over twelve consecutive frames of a
    # standing character: the same tree scored between 0.25 and 0.35 around a 0.30
    # threshold. Following it at the same threshold therefore lost it every few frames,
    # six of those in a row read as the node being gone and the bot walked away from a
    # full tree announcing it had emptied it in 3.1s, which is TRACK_MISSES of polling
    # and nothing else. Only a box already close to a node the bot is standing at is
    # accepted this cheaply, so a wrong one has to be in the right place to cost anything.
    TRACK_CONFIDENCE_FACTOR = 0.5

    # A node holds several charges, and one click takes one of them: a big tree with
    # fifty wood in it hands over ten and stays standing. The bot used to click once
    # and walk off, leaving most of every node behind, so the node is clicked again
    # until the model stops finding it or this many charges have been taken.
    CHARGE_ATTEMPTS = 16

    # Time in second left between two charges, the character needs a moment to swing
    # again and clicking through the animation is ignored by the game.
    CHARGE_DELAY = 0.8

    # How long a charge after the first waits for the character to start working. The
    # character is already standing on the node by then, so it either starts swinging
    # within a couple of seconds or there is nothing left in it, and this is what tells
    # an emptied node from a full one. The walking timeout would answer the same
    # question twenty seconds later, once a node in five is empty that is most of the
    # time the bot spends.
    CHARGE_ARRIVAL_TIMEOUT = 6

    # Confirming a node by hovering it before walking anywhere. The game draws the name
    # of a resource over it in near white when the cursor is on it and draws nothing at
    # all over scenery, which is the only thing on screen that knows the difference.
    # Measured over six trees the model found, the one real node put 42 more bright
    # pixels on screen and the five scenery ones put none, so anything over a handful
    # settles it. The radius is generous because the name is drawn above the node and
    # is wider than the trunk it belongs to.
    LABEL_RADIUS = 90
    LABEL_BRIGHT = 200
    LABEL_JUMP = 15
    LABEL_SETTLE = 0.8

    # A tree the game refuses to name is scenery and will still be scenery in a minute,
    # so it is put aside for longer than a node that was merely emptied.
    SCENERY_COOLDOWN = 300

    # How many scans in a row may fail to find a node the character is standing at
    # before it is given up on. One miss is the model blinking, several in a row on a
    # node that also refuses to be gathered is a stump.
    NODE_LOST_ATTEMPTS = 3

    # Roaming. Once everything in sight has been gathered the character has to be taken
    # somewhere else, otherwise the bot stands in an emptied clearing until it is
    # stopped, which is most of the point of a gathering bot gone.
    #
    # The direction is held for a few moves so ground is really covered instead of the
    # character shuffling around one spot, and it is turned once the bot keeps finding
    # nothing. The vertical part is squashed because the camera looks at the world from
    # an angle, so a pixel up the screen is more world than a pixel across it.
    ROAM_RADIUS = 0.22
    ROAM_VERTICAL_SQUASH = 0.6
    ROAM_WAIT = 3.0
    ROAM_DIRECTIONS = 8
    ROAM_HAZARD_DISTANCE = 130

    # Part of the game window that can be clicked to walk, as fractions of it. The
    # interface is left out: the portrait and the icons along the top, the action bar
    # at the bottom, the minimap and the buttons in the bottom right corner. Clicking
    # any of those opens a panel over the game instead of moving the character.
    ROAM_AREA_LEFT, ROAM_AREA_TOP = 0.08, 0.16
    ROAM_AREA_RIGHT, ROAM_AREA_BOTTOM = 0.74, 0.60

    # Boxes of the window holding the interface, as (left, top, right, bottom) fractions:
    # the bar of icons along the top, the portrait in the corner, the action bar at the
    # bottom and the minimap. The model does find resources in there, the minimap draws
    # little coloured nodes after all, and clicking one opens a panel or drags the map
    # instead of gathering anything.
    INTERFACE_AREAS = (
        (0.00, 0.00, 1.00, 0.06),
        (0.00, 0.00, 0.16, 0.09),
        (0.18, 0.86, 0.76, 1.00),
        (0.75, 0.64, 1.00, 1.00),
    )

    def __init__(self, model):
        self.model: AlbionDetection = model
        self.current_gathering: Gathering | None = None
        self.depleted = []
        self.stop_requested = False

        self.debug = self.model.debug
        self.preview = self.model.preview

        # The bot used to refuse to start without this picture. It is only an
        # accelerator now, telling that the character really reached the node, and the
        # gathering is followed through the model when it is missing, so an uncalibrated
        # bot still works instead of not running at all.
        try:
            self.img_border_resource = bar.load()
        except Exception as e:
            self.img_border_resource = None
            print(f"No usable gathering bar ({e}), following the nodes with the model instead")

        self.pointer = pointer.create()
        self.last_failsafe_check = time()

        # Where the character is being walked, and how far along the expanding square
        # the roaming is, see __roam.
        self.heading = uniform(0, 360)
        self.empty_scans = 0
        self.sweep_leg = 2
        self.sweep_done = 0
        self.sweep_turns = 0

    def toggle_ath(self):
        """
        Hide or show the whole game interface.

        Nothing calls this any more and gathering must not: alt+h takes away the entire
        HUD, and the bar telling that the character reached a node and is working on it
        is part of that HUD. Hiding it before walking to a tree left __moving waiting for
        a bar the game could no longer draw, so every node ended on "Mooving timed out"
        and the tree was left standing. It hurt the model as well, since every frame it
        was trained on has the interface on screen, so the frames taken mid gathering
        were unlike anything it had seen and it lost the node it was standing at.
        """
        self.pointer.hotkey('alt', 'h')

    def go_on_mount(self):
        self.pointer.press('a')

    def __park(self):
        """
        Put the cursor somewhere harmless between two actions, so the tooltip the game
        draws under it stops covering what the model is looking at.
        """
        window = self.model.window_capture.window

        self.pointer.move(int(window.left + window.width * self.PARKING_X),
                          int(window.top + window.height * self.PARKING_Y))

    def __is_mining(self):
        return bar.is_visible(self.model.window_capture, self.img_border_resource)

    def __wait_polling(self, duration=None):
        """
        Wait, keeping the debug window alive instead of letting Windows mark it as not
        responding while the character gathers. The window eats the keys it is given,
        so q is remembered here to stop the bot in the middle of a node instead of
        being lost.

        :param duration: Time in second to wait, the bar polling when left to None.
        """
        duration = self.BAR_POLLING if duration is None else duration

        if self.preview:
            if cv.waitKey(int(duration * 1000)) == ord('q'):
                self.stop_requested = True
        else:
            sleep(duration)

        # Gathering a node takes tens of seconds, and the mouse thrown in the corner has
        # to stop the bot during that time too, and not only between two nodes. Asking
        # the compositor costs a process, so it is not asked on every read of the bar.
        if time() - self.last_failsafe_check >= self.FAILSAFE_POLLING:
            self.last_failsafe_check = time()
            self.pointer.check_failsafe()

            # Gathering a node is the longest the bot goes without touching the game,
            # and the bar it is watching for is read off the screen, so a window coming
            # up over it during that minute would read as the node being finished.
            self.model.window_capture.focus()

    def __mining(self, timeout):
        """
        Wait for the resource bar to disappear, meaning the node is depleted.

        :param timeout: Maximum time in second spent on the node.
        :return: True when the node has been depleted before the timeout.
        """
        if self.debug:
            print("Start Minning...")

        start = time()

        while self.__is_mining() is True:
            if self.stop_requested:
                return False

            if time() - start > timeout:
                if self.debug:
                    print("Minning timed out")
                return False

            self.__wait_polling()

        if self.debug:
            print(f"Minning completed in {time() - start:.1f}s")

        return True

    def __moving(self, timeout):
        """
        Wait for the resource bar to appear, meaning the character reached the node.

        :param timeout: Maximum time in second spent walking.
        :return: True when the node has been reached before the timeout.
        """
        if self.debug:
            print("Start mooving...")

        start = time()

        while self.__is_mining() is False:
            if self.stop_requested:
                return False

            if time() - start > timeout:
                if self.debug:
                    print("Mooving timed out")
                return False

            self.__wait_polling()

        if self.debug:
            print(f"Mooving completed in {time() - start:.1f}s")

        return True

    def __track(self, x, y, resource, timeout):
        """
        Follow a node until the model stops finding it, which is what a gathered node
        looks like, and the way the bot knows it is done without needing a picture of
        the gathering bar.

        :param x: Screen position the node was clicked at.
        :param y: Screen position the node was clicked at.
        :param resource: ResourceProfile of the node.
        :param timeout: Maximum time in second spent on it.
        :return: True when the node went away before the timeout.
        """
        if self.debug:
            print("Following the node...")

        start = time()
        misses = 0
        node_x, node_y = x, y

        while time() - start < timeout:
            if self.stop_requested:
                return False

            closest = self.find_node(node_x, node_y, resource)

            if closest is None:
                misses += 1

                if misses >= self.TRACK_MISSES:
                    if self.debug:
                        print(f"Node gathered in {time() - start:.1f}s")
                    return True
            else:
                misses = 0
                node_x, node_y = closest

            self.__wait_polling(self.TRACK_POLLING)

        if self.debug:
            print("Gave up on the node, it is still there")

        return False

    def find_node(self, x, y, resource):
        """
        Look for a node of a kind again, around where it was last seen.

        :param x: Screen position it was last seen at.
        :param y: Screen position it was last seen at.
        :param resource: ResourceProfile to look for.
        :return: (x, y) where it is now, None when it is not there any more.
        """
        detections, _, _ = self.model.scan(self.TRACK_CONFIDENCE_FACTOR)

        window = self.model.window_capture.window
        closest, closest_distance = None, window.width * self.TRACK_RADIUS

        for found_x, found_y, found_resource in detections:
            if found_resource is not resource:
                continue

            distance = sqrt((found_x - x) ** 2 + (found_y - y) ** 2)

            if distance < closest_distance:
                closest, closest_distance = (found_x, found_y), distance

        return closest

    def on_interface(self, x, y):
        """
        Check whether a screen position falls on the interface of the game rather than
        on the world, so a resource the model believes it sees on the minimap is left
        alone instead of being clicked.

        :param x: Screen position.
        :param y: Screen position.
        :return: True when the position is over the interface.
        """
        window = self.model.window_capture.window

        for left, top, right, bottom in self.INTERFACE_AREAS:
            if (window.left + window.width * left <= x <= window.left + window.width * right
                    and window.top + window.height * top <= y <= window.top + window.height * bottom):
                return True

        return False

    def __near_hazard(self, x, y):
        """
        Check whether a spot is close to something the model knows but the bot cannot
        gather, a monster being the one that matters.

        :param x: Screen position to walk to.
        :param y: Screen position to walk to.
        :return: True when walking there is asking for a fight.
        """
        for hazard_x, hazard_y in self.model.last_hazards:
            if sqrt((x - hazard_x) ** 2 + (y - hazard_y) ** 2) <= self.ROAM_HAZARD_DISTANCE:
                return True

        return False

    def __roam(self):
        """
        Walk the character to another spot, so the bot looks for nodes somewhere else
        instead of standing where it has already taken everything.

        :return: True when the character has been sent walking.
        """
        window = self.model.window_capture.window
        character_x, character_y = self.model.character_screen_position()

        radius = min(window.width, window.height) * self.ROAM_RADIUS

        left = window.left + window.width * self.ROAM_AREA_LEFT
        right = window.left + window.width * self.ROAM_AREA_RIGHT
        top = window.top + window.height * self.ROAM_AREA_TOP
        bottom = window.top + window.height * self.ROAM_AREA_BOTTOM

        # An expanding square: a few moves one way, a quarter turn, a few moves the next,
        # and the legs grow every second turn. Wandering off in random directions kept
        # crossing the same emptied ground, this covers an area outwards from where it
        # started instead. The wobble is there so a leg blocked by a rock does not have
        # the character pushing into it over and over.
        if self.sweep_done >= self.sweep_leg:
            self.sweep_done = 0
            self.sweep_turns += 1
            self.heading += 90

            if self.sweep_turns % 2 == 0:
                self.sweep_leg += 1

        self.sweep_done += 1
        self.heading += uniform(-12, 12)

        for attempt in range(self.ROAM_DIRECTIONS):
            angle = radians(self.heading + attempt * (360 / self.ROAM_DIRECTIONS))

            target_x = character_x + radius * cos(angle)
            target_y = character_y + radius * sin(angle) * self.ROAM_VERTICAL_SQUASH

            target_x = min(max(target_x, left), right)
            target_y = min(max(target_y, top), bottom)

            if not self.__near_hazard(target_x, target_y):
                break
        else:
            # Every way out is next to something unfriendly, better to stand still and
            # look again than to walk into it.
            if self.debug:
                print("Surrounded by monsters, waiting instead of walking")

            self.__wait_polling(self.ROAM_WAIT)
            return False

        if self.debug:
            print(f"Nothing in sight, walking to ({int(target_x)}, {int(target_y)})")

        self.pointer.left_click(int(target_x), int(target_y))
        self.__park()

        waited = 0.0

        while waited < self.ROAM_WAIT and not self.stop_requested:
            self.__wait_polling(self.TRACK_POLLING)
            waited += self.TRACK_POLLING

        return True

    def is_depleted(self, x, y):
        """
        Check if a node has already been gathered recently.

        :param x: Screen position of the node.
        :param y: Screen position of the node.
        :return: True when the node has to be skipped.
        """
        now = time()

        self.depleted = [node for node in self.depleted if node[2] > now]

        for node_x, node_y, _ in self.depleted:
            if sqrt((x - node_x) ** 2 + (y - node_y) ** 2) <= self.DEPLETED_RADIUS:
                return True

        return False

    def __charge(self, x, y, resource, arrival_timeout=None):
        """
        Take one charge out of a node, one click and the wait that follows it.

        :param x: Screen position of the node.
        :param y: Screen position of the node.
        :param resource: ResourceProfile of the node.
        :param arrival_timeout: Time in second to wait for the character to start
                                working on the node, the walking timeout of the profile
                                when left to None.
        :return: True when the character reached the node and gathered from it.
        """
        self.pointer.left_click(x, y)
        self.__park()

        if self.img_border_resource is not None:
            # The bar says the character really reached the node, so a click that
            # landed on the ground is given up on after the walking timeout instead
            # of holding the bot for the whole gathering one as well.
            timeout = resource.moving_timeout if arrival_timeout is None else arrival_timeout

            return self.__moving(timeout) and self.__mining(resource.gathering_timeout)

        return self.__track(x, y, resource, resource.moving_timeout + resource.gathering_timeout)

    def gathering(self, x, y, resource):
        """
        Walk to a node and gather it until it is really empty.

        A node holds several charges and a click takes one of them, so it is clicked
        again as long as the model still finds it standing there. Stopping after the
        first one, which is what the bot used to do, leaves most of a big tree behind.

        :param x: Screen position of the node.
        :param y: Screen position of the node.
        :param resource: ResourceProfile of the node.
        :return: Number of charges taken out of the node.
        """
        self.current_gathering = Gathering(x, y, resource)

        if self.debug:
            print(f"Gathering {self.current_gathering}")

        node_x, node_y = x, y
        charges = 0
        misses = 0

        try:
            for attempt in range(self.CHARGE_ATTEMPTS):
                # The first click is the one that walks the character over, and it is
                # given the whole walking timeout. Every one after it is thrown at a node
                # the character is already standing on, so waiting twenty seconds to find
                # out that a stump gives nothing back is twenty seconds of standing still.
                arrival = None if attempt == 0 else self.CHARGE_ARRIVAL_TIMEOUT

                if not self.__charge(node_x, node_y, resource, arrival):
                    # The character did not start working on it, which is what an emptied
                    # node looks like. Clicking the same spot again would only repeat it.
                    if self.debug and charges > 0:
                        print(f"Nothing left to take after {charges} "
                              f"charge{'s' if charges > 1 else ''}")
                    break

                charges += 1
                misses = 0

                if self.stop_requested:
                    break

                self.__wait_polling(self.CHARGE_DELAY)

                found = self.find_node(node_x, node_y, resource)

                if found is None:
                    # Losing sight of a node is not the same as having emptied it. A tree
                    # keeps its shape until the last charge is out of it, and the model
                    # drops one here and there, so walking off on the first miss is what
                    # left half chopped trees standing. The last known spot is clicked
                    # again instead, and the game itself settles it: the character either
                    # starts swinging, so there was something left, or it does not and
                    # the charge above breaks the loop.
                    misses += 1

                    if misses >= self.NODE_LOST_ATTEMPTS:
                        if self.debug:
                            print(f"Lost the node after {charges} "
                                  f"charge{'s' if charges > 1 else ''}")
                        break
                else:
                    misses = 0
                    node_x, node_y = found
        finally:
            self.current_gathering = None

        if charges > 0 and self.debug:
            print(f"Took {charges} charge{'s' if charges > 1 else ''} from the {resource}")

        self.depleted.append((node_x, node_y, time() + self.DEPLETED_COOLDOWN))

        return charges

    def confirmed(self, x, y, clean):
        """
        Ask the game whether there is really a resource where the model believes there
        is one, by putting the cursor on it and looking for the name the game writes over
        a node it accepts.

        A forest holds far more trees than gatherable ones and they are the same tree to
        look at, so the model finds both: measured over six it picked out of Forgotten
        Woods, one was a node and five were scenery. Walking to those five costs the
        walking timeout each and leaves the character standing against a trunk it cannot
        touch, which is most of what the bot spent its time doing.

        :param x: Screen position of the node.
        :param y: Screen position of the node.
        :param clean: Frame taken with the cursor parked away, to compare against.
        :return: True when the game named something under the cursor.
        """
        image_x, image_y = self.model.image_position(x, y)

        left = max(image_x - self.LABEL_RADIUS, 0)
        top = max(image_y - self.LABEL_RADIUS, 0)
        right = min(image_x + self.LABEL_RADIUS, self.model.IMG_SIZE)
        bottom = min(image_y + self.LABEL_RADIUS, self.model.IMG_SIZE)

        def lit(frame):
            grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            return int((grey[top:bottom, left:right] > self.LABEL_BRIGHT).sum())

        before = lit(clean)

        self.pointer.move(x, y)
        self.__wait_polling(self.LABEL_SETTLE)

        after = lit(self.model._process_image(self.model.window_capture.screenshot()))

        # Put back where the tooltip cannot cover the next look at the world, whatever
        # the answer turns out to be.
        self.__park()

        if self.debug and after - before < self.LABEL_JUMP:
            print(f"Nothing there, the game names no resource at ({x}, {y})")

        return after - before >= self.LABEL_JUMP

    def __skip(self, x, y):
        """
        Check whether a resource the model found has to be left alone, either because
        it has just been gathered or because it is not really in the world.

        :param x: Screen position of the resource.
        :param y: Screen position of the resource.
        :return: True when the resource is not worth clicking.
        """
        return self.is_depleted(x, y) or self.on_interface(x, y)

    def loop(self):
        """
        Gather the closest node over and over until the user stops the program, with
        ctrl+c, by pressing q on the debug window, or by throwing the mouse in the top
        left corner of the screen.
        """
        # Anything covering the game is what the model is shown, because the capture
        # reads the screen where the window is rather than asking the window for its
        # picture. Raising it once at the start is not enough: a notification, or a
        # console the bot is being run from, takes the front back at some point and from
        # then on the model quietly scores trees against somebody else's window. It is
        # asked for again on every pass, which costs one call to the compositor when it
        # is already in front, and nothing else.
        try:
            while not self.stop_requested:
                self.model.window_capture.focus()

                x, y, resource, img = self.model.predict(ignore=self.__skip)

                if self.preview and cv.waitKey(1) == ord('q'):
                    break

                if resource is None:
                    self.empty_scans += 1
                    self.__roam()
                    continue

                # That frame was taken with the cursor parked out of the way, so it is
                # what the node looks like untouched and the hover is compared to it.
                if not self.confirmed(x, y, img):
                    self.depleted.append((x, y, time() + self.SCENERY_COOLDOWN))
                    continue

                self.empty_scans = 0

                self.gathering(x, y, resource)

                sleep(resource.delay_between_nodes)

        except KeyboardInterrupt:
            print("Stopped")
        except pointer.FailSafe as e:
            print(str(e))
        finally:
            cv.destroyAllWindows()
