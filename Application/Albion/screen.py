import cv2 as cv

# Size of the square image the model works with.
IMG_SIZE = 640


def process(img):
    """
    Turn a screenshot into the image the model expects.

    The screenshot comes from mss as BGRA, and yolov5 expects a RGB image, so this
    conversion drops the alpha channel and swaps the red and the blue one. Colors of
    the debug window are swapped because of it, don't "fix" it or the model will stop
    detecting anything.

    :param img: Screenshot of the game window.
    :return: Processed image.
    """
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return cv.resize(img, (IMG_SIZE, IMG_SIZE))


def grab(capture):
    """
    Take a processed picture of the game window.

    :param capture: Capture bound to the game window.
    :return: Processed image.
    """
    return process(capture.screenshot())
