import sys
from pathlib import Path

# Makes the Application package importable, so "python main.py" works from the
# Application folder as well as "python Application/main.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from time import sleep
from Application import paths
from Application.Albion import bar
from Application.Albion.detection import AlbionDetection
from Application.Capture.Factory import CaptureFactory
from Application.Interaction.interaction import Interaction
import cv2 as cv


def run():
    """
    Only show what the model detects, without touching the mouse.
    """
    sleep(2)
    model = AlbionDetection(debug=True, confidence=0.9)
    while True:
        model.predict()

        if model.debug:

            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break


def gathering(targets=None):
    """
    Gather the resources given in targets, all of them when targets is None.
    """
    sleep(2)
    model = AlbionDetection(debug=True, confidence=0.8, targets=targets)
    interaction = Interaction(model)

    interaction.loop()


def tree_gathering():
    gathering(targets=["tree"])


def calibrate_resource_bar(delay=10):
    """
    Save the picture of the gathering bar the bot looks for to know if the character
    is gathering. Start gathering a resource by hand, the picture is taken once the
    delay is over.

    :param delay: Time in second before the screenshot is taken.
    """
    capture = CaptureFactory().capture

    print(f"Start gathering a resource, the screenshot is taken in {delay}s")

    for remaining in range(delay, 0, -1):
        print(f"{remaining}...")
        sleep(1)

    bar.save(capture)

    print(f"Saved {paths.RESOURCE_BAR} and the whole frame in {paths.RESOURCE_FRAME}")
    print("Check it with check_resource_bar(), then run tree_gathering()")


def check_resource_bar():
    """
    Print how much the gathering bar is recognized, to check the picture saved by
    calibrate_resource_bar. The value has to go over bar.CONFIDENCE while gathering,
    and stay under it while doing nothing.
    """
    capture = CaptureFactory().capture
    template = bar.load()

    while True:
        found = bar.confidence(capture, template)

        print(f"{found:.2f} {'gathering' if found >= bar.CONFIDENCE else 'idle'}")

        sleep(0.5)


def testWindowsCapture():
    """
    Check the game window is found and captured.
    """
    window = CaptureFactory().capture
    print(window.window)

    cv.imshow("Founded", window.screenshot())
    cv.waitKey(0)


if __name__ == "__main__":
    #testWindowsCapture()
    #calibrate_resource_bar()
    #check_resource_bar()
    #run()
    tree_gathering()
    #gathering()
