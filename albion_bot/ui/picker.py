"""
A window showing a screenshot of the game, to draw a box on it or to click a point.
"""

import base64
import tkinter as tk
from tkinter import ttk

import cv2 as cv

from albion_bot.platform import OWN_WINDOW_TITLES

MAX_WIDTH, MAX_HEIGHT = 1280, 720


def photo(frame, max_width, max_height):
    """
    :return: (PhotoImage, scale) of a BGR frame shrunk to fit. Tk reads PNG by itself.
    """
    scale = min(max_width / frame.shape[1], max_height / frame.shape[0], 1.0)
    shown = cv.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv.INTER_AREA)
    ok, png = cv.imencode(".png", shown, [cv.IMWRITE_PNG_COMPRESSION, 1])

    if not ok:
        raise ValueError("could not encode the picture")

    return tk.PhotoImage(data=base64.b64encode(png.tobytes())), scale


class Picker(tk.Toplevel):

    def __init__(self, parent, frame, instructions, on_done, mode="box", initial=None):
        """
        :param frame: BGR frame of the game.
        :param instructions: What to draw.
        :param on_done: Called with the box (left, top, width, height) or the point (x, y),
                        as fractions of the frame.
        :param mode: "box" or "point".
        :param initial: Current value, drawn to start with.
        """
        super().__init__(parent)
        self.title(OWN_WINDOW_TITLES[2])
        self.transient(parent)
        self.resizable(False, False)

        self.frame_shape = frame.shape
        self.on_done = on_done
        self.mode = mode
        self.image, self.scale = photo(frame, MAX_WIDTH, MAX_HEIGHT)
        self.start = None
        self.value = None

        ttk.Label(self, text=instructions, padding=8, wraplength=MAX_WIDTH).pack(anchor="w")

        self.canvas = tk.Canvas(self, width=self.image.width(), height=self.image.height(),
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")

        buttons = ttk.Frame(self, padding=8)
        buttons.pack(fill="x")
        self.ok = ttk.Button(buttons, text="Use it", command=self.accept, state="disabled")
        self.ok.pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=6)

        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.bind("<Return>", lambda _: self.accept())
        self.bind("<Escape>", lambda _: self.destroy())

        if initial is not None:
            self.value = tuple(initial)
            self.draw()
            self.ok.configure(state="normal")

        self.after(50, self.take_focus)

    def take_focus(self):
        # A grab on a window not shown yet fails on some window managers.
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

        self.focus_set()

    def size(self):
        return self.image.width(), self.image.height()

    def press(self, event):
        width, height = self.size()

        if self.mode == "point":
            self.value = (event.x / width, event.y / height)
            self.draw()
            self.ok.configure(state="normal")
        else:
            self.start = (event.x, event.y)

    def drag(self, event):
        if self.mode != "box" or self.start is None:
            return

        width, height = self.size()
        x1, x2 = sorted((self.start[0], min(max(event.x, 0), width)))
        y1, y2 = sorted((self.start[1], min(max(event.y, 0), height)))

        if x2 - x1 >= 3 and y2 - y1 >= 3:
            self.value = (x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height)
            self.draw()

    def release(self, event):
        self.drag(event)
        self.start = None

        if self.value is not None:
            self.ok.configure(state="normal")

    def draw(self):
        self.canvas.delete("mark")
        width, height = self.size()

        if self.mode == "point":
            x, y = self.value[0] * width, self.value[1] * height
            self.canvas.create_line(x - 14, y, x + 14, y, fill="yellow", width=2, tags="mark")
            self.canvas.create_line(x, y - 14, x, y + 14, fill="yellow", width=2, tags="mark")
        else:
            left, top, box_width, box_height = self.value
            self.canvas.create_rectangle(left * width, top * height, (left + box_width) * width,
                                         (top + box_height) * height, outline="yellow", width=2, tags="mark")

    def accept(self):
        if self.value is None:
            return

        value = self.value
        self.destroy()
        self.on_done(value)
