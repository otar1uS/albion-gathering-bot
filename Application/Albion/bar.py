from Application import paths
from Application.Albion import screen
import cv2 as cv

# Region of the 640x640 image holding the bars drawn while gathering, the segmented
# charges left on the node and the progress bar under them. They sit just under the
# name of the character, which is left out on purpose: the name is on screen at all
# times, and a template holding it would match whether the character gathers or not.
#
# These are pixels of the 640x640 image the window is squeezed into, so they hold for
# any window size, but not for a different interface scale or a different camera zoom.
# To find them again, gather by hand, run the calibration, and look at the whole frame
# it leaves in images/640x640_resource.png.
TOP_X, TOP_Y = 304, 205
BOTTOM_X, BOTTOM_Y = 340, 217

# The bars are searched for in a much wider area than the picture saved of them, so
# matchTemplate has room to slide. It is this generous because the bars are drawn over
# the head of the character and how far over depends on the camera zoom: zooming in
# once moved them forty pixels up, which a tight region would have missed entirely.
MARGIN = 40

# Confidence over which the bar is considered visible.
CONFIDENCE = 0.8

# Spread of the grey levels under which a saved picture is considered flat, see usable.
FLATNESS = 5.0


def searched_region():
    """
    Bounds of the area the bars are looked for in, the saved picture plus the margin.

    :return: (left, top, right, bottom), clamped to the image.
    """
    return (max(TOP_X - MARGIN, 0),
            max(TOP_Y - MARGIN, 0),
            min(BOTTOM_X + MARGIN, screen.IMG_SIZE),
            min(BOTTOM_Y + MARGIN, screen.IMG_SIZE))


def crop(capture):
    """
    Cut the region of the game window the gathering bars are looked for in.

    :param capture: Capture bound to the game window.
    :return: Grayscale region.
    """
    left, top, right, bottom = searched_region()
    img = screen.grab(capture)

    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)[top:bottom, left:right]


def load():
    """
    Load the picture of the gathering bar to look for.

    :return: Grayscale template.
    """
    # Grayscale because the region it is compared against is grayscale too,
    # matchTemplate needs both to have the same number of channels.
    template = cv.imread(str(paths.RESOURCE_BAR), cv.IMREAD_GRAYSCALE)

    if template is None:
        raise Exception(
            f"The picture of the gathering bar is missing in {paths.RESOURCE_BAR}, "
            f"create it with the Calibrate button of the interface, or with "
            f"calibrate_resource_bar() while gathering a resource"
        )

    height, width = template.shape[:2]
    left, top, right, bottom = searched_region()

    if height > bottom - top or width > right - left:
        raise Exception(
            f"The picture of the gathering bar {width}x{height} is bigger than the "
            f"{right - left}x{bottom - top} region it is looked for in, create it again"
        )

    return template


def save(capture):
    """
    Save the picture of the gathering bar, the character has to be gathering.

    :param capture: Capture bound to the game window.
    :return: Grayscale template.
    """
    img = screen.grab(capture)
    template = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[TOP_Y:BOTTOM_Y, TOP_X:BOTTOM_X]

    paths.IMAGES.mkdir(exist_ok=True)

    # The whole frame is kept around to crop the bar again by hand if needed.
    cv.imwrite(str(paths.RESOURCE_FRAME), img)
    cv.imwrite(str(paths.RESOURCE_BAR), template)

    return template


def usable(template):
    """
    Tell whether a saved picture of the bar can be matched at all.

    A calibration done while nothing was being gathered saves a flat piece of
    background, and matchTemplate answers noise on a picture with nothing in it, so the
    bot would either never start gathering or believe it never stops.

    :param template: Picture of the bar to check.
    :return: True when the picture holds something to look for.
    """
    return float(template.std()) >= FLATNESS


def confidence(capture, template):
    """
    Tell how much the gathering bar is recognized on screen.

    :param capture: Capture bound to the game window.
    :param template: Picture of the bar to look for.
    :return: Confidence between 0 and 1.
    """
    result = cv.matchTemplate(crop(capture), template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)

    return max_val


def is_visible(capture, template):
    """
    Check if the character is gathering.

    :param capture: Capture bound to the game window.
    :param template: Picture of the bar to look for.
    :return: True when the bar is on screen.
    """
    return confidence(capture, template) >= CONFIDENCE
