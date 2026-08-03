"""
Telling whether anything on screen is moving.

The bot needs to answer two questions it has no other way of answering: is the character
walking, and is it working on a node. Both were previously guessed at by watching for a
template of the gathering bar in a fixed corner of the frame, which broke the moment the
camera zoom changed and then reported that every node in the world had been emptied in
3.1 seconds.

Comparing consecutive frames is a cruder signal and a far more robust one. A character
walking drags the whole camera with it and the frame changes everywhere; a character
standing still leaves a frame that only breathes, the grass and the light. The numbers
below are measured: standing still with the game running gives a mean change under 1.0
between frames, and walking gives many times that.
"""

from collections import deque

import cv2 as cv
import numpy as np

from .. import logs

log = logs.get("vision.motion")


class Motion:
    """Keeps the last few frames to say whether the picture is moving."""

    def __init__(self, config, history=6):
        self.config = config
        self.frames = deque(maxlen=history)

    def clear(self):
        """Forget what has been seen, after a deliberate jump like a zone change."""
        self.frames.clear()

    def add(self, frame):
        """
        Take note of a frame.

        :param frame: BGR frame from the detector.
        :return: Mean change against the previous frame, None for the first one.
        """
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        previous = self.frames[-1] if self.frames else None
        self.frames.append(grey)

        if previous is None:
            return None

        return float(np.abs(grey.astype(np.int16) - previous.astype(np.int16)).mean())

    def moving(self, change):
        """
        Whether a measured change counts as the world moving.

        :param change: Value from add, or None.
        :return: True when something is moving.
        """
        return change is not None and change >= self.config.anti_stuck.motion_threshold

    def frozen(self, samples, epsilon):
        """
        Whether the picture has stopped changing altogether.

        This is a different question from the character standing still, and needs a far
        smaller number to answer. A character standing in a forest still gives a mean
        change around 0.6 to 1.0 between frames, because the grass moves, the light
        shifts and other players walk past. A frame that is byte identical to the last
        one several times over is not a quiet game, it is a game that is not being
        played: the login screen, a loading screen, or a client that has stopped
        rendering. Measured at the login screen: exactly 0.00, every time.

        :param samples: How many frames back to require stillness over.
        :param epsilon: Change under which two frames count as identical.
        :return: True when nothing at all is happening.
        """
        if len(self.frames) <= samples:
            return False

        recent = list(self.frames)[-(samples + 1):]

        for older, newer in zip(recent, recent[1:]):
            change = float(np.abs(newer.astype(np.int16) - older.astype(np.int16)).mean())

            if change > epsilon:
                return False

        return True

    def still_for(self, samples):
        """
        Whether the last few frames have all been the same picture.

        :param samples: How many frames back to require stillness over.
        :return: True when nothing has moved across them.
        """
        if len(self.frames) <= samples:
            return False

        recent = list(self.frames)[-(samples + 1):]

        for older, newer in zip(recent, recent[1:]):
            change = float(np.abs(newer.astype(np.int16) - older.astype(np.int16)).mean())

            if change >= self.config.anti_stuck.motion_threshold:
                return False

        return True
