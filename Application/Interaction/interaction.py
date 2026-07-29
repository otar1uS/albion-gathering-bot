from Application.Albion import bar
from Application.Albion.detection import AlbionDetection
from Application.Interaction import pointer
from math import sqrt
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
    # covering what the model is looking at. Far enough from the corner not to trip
    # the panic button watching it.
    PARKING_X, PARKING_Y = 10, 10

    def __init__(self, model):
        self.model: AlbionDetection = model
        self.current_gathering: Gathering | None = None
        self.depleted = []
        self.stop_requested = False

        self.debug = self.model.debug
        self.preview = self.model.preview
        self.img_border_resource = bar.load()
        self.pointer = pointer.create()
        self.last_failsafe_check = time()

    def toggle_ath(self):
        self.pointer.hotkey('alt', 'h')

    def go_on_mount(self):
        self.pointer.press('a')

    def __is_mining(self):
        return bar.is_visible(self.model.window_capture, self.img_border_resource)

    def __wait_polling(self):
        """
        Wait between two reads of the bar, keeping the debug window alive instead of
        letting Windows mark it as not responding while the character gathers. The
        window eats the keys it is given, so q is remembered here to stop the bot in
        the middle of a node instead of being lost.
        """
        if self.preview:
            if cv.waitKey(int(self.BAR_POLLING * 1000)) == ord('q'):
                self.stop_requested = True
        else:
            sleep(self.BAR_POLLING)

        # Gathering a node takes tens of seconds, and the mouse thrown in the corner has
        # to stop the bot during that time too, and not only between two nodes. Asking
        # the compositor costs a process, so it is not asked on every read of the bar.
        if time() - self.last_failsafe_check >= self.FAILSAFE_POLLING:
            self.last_failsafe_check = time()
            self.pointer.check_failsafe()

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

    def gathering(self, x, y, resource):
        """
        Walk to a node and gather it until it is depleted.

        :param x: Screen position of the node.
        :param y: Screen position of the node.
        :param resource: ResourceProfile of the node.
        :return: True when the node has been fully gathered.
        """
        self.toggle_ath()
        self.current_gathering = Gathering(x, y, resource)

        if self.debug:
            print(f"Gathering {self.current_gathering}")

        try:
            self.pointer.left_click(self.current_gathering.x, self.current_gathering.y)
            self.pointer.move(self.PARKING_X, self.PARKING_Y)

            gathered = self.__moving(resource.moving_timeout) and self.__mining(resource.gathering_timeout)
        finally:
            # Leaves the game interface the way it was found, even on a crash.
            self.toggle_ath()

        self.depleted.append((x, y, time() + self.DEPLETED_COOLDOWN))

        return gathered

    def loop(self):
        """
        Gather the closest node over and over until the user stops the program, with
        ctrl+c, by pressing q on the debug window, or by throwing the mouse in the top
        left corner of the screen.
        """
        try:
            while not self.stop_requested:
                x, y, resource, _ = self.model.predict(ignore=self.is_depleted)

                if self.preview and cv.waitKey(1) == ord('q'):
                    break

                if resource is None:
                    if self.debug:
                        print("No resource on screen, waiting...")
                    sleep(1)
                    continue

                self.gathering(x, y, resource)

                sleep(resource.delay_between_nodes)

        except KeyboardInterrupt:
            print("Stopped")
        except pointer.FailSafe as e:
            print(str(e))
        finally:
            cv.destroyAllWindows()
