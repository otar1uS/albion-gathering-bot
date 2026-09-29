"""
The interface: pick what to gather and where, set the bot up once, watch it work.
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from albion_bot import config, paths
from albion_bot.config import ROUTE_MODES
from albion_bot.navigation import route
from albion_bot.platform import IS_WINDOWS, OWN_WINDOW_TITLES, UI_WINDOW_TITLE
from albion_bot.ui.picker import Picker, photo
from albion_bot.ui.runner import Runner, checks
from albion_bot.vision import minimap, resources

MESSAGE_POLLING = 100
CHECK_POLLING = 3000
NO_ROUTE = "(stay where the character is)"
STOP_HINT = ("F12, the Stop button, or the mouse thrown into the top left corner stop the bot"
             if IS_WINDOWS else "The Stop button, or the mouse thrown into the top left corner stop the bot")


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(UI_WINDOW_TITLE)
        self.minsize(640, 720)

        self.settings = config.load()
        self.runner = Runner()
        self.task_buttons = []
        self.preview_window = None
        self.preview_image = None
        self.minimap_image = None

        s = self.settings
        self.targets = {p.name: tk.BooleanVar(value=p.name in s.targets) for p in resources.GATHERABLE}
        self.route = tk.StringVar(value=s.route or NO_ROUTE)
        self.route_mode = tk.StringVar(value=s.route_mode)
        self.max_minutes = tk.StringVar(value=f"{s.max_minutes:g}")
        self.confidence = tk.StringVar(value=f"{s.confidence:g}")
        self.bar_threshold = tk.StringVar(value=f"{s.bar_threshold:g}")
        self.window_name = tk.StringVar(value=s.window_name)
        self.hide_hud = tk.BooleanVar(value=s.hide_hud)
        self.avoid_monsters = tk.BooleanVar(value=s.avoid_monsters)
        self.mount_key = tk.StringVar(value=s.mount_key)
        self.preview = tk.BooleanVar(value=s.preview)
        self.minimize_ui = tk.BooleanVar(value=s.minimize_ui)
        self.new_route = tk.StringVar()
        self.game = tk.StringVar(value="Looking for the game...")
        self.status = tk.StringVar(value="Idle")

        self.build()
        self.refresh_routes()
        self.poll_checks()

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(MESSAGE_POLLING, self.read_messages)

    # ---------------------------------------------------------------- layout

    def build(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        ttk.Label(root, textvariable=self.game).pack(anchor="w")
        ttk.Label(root, textvariable=self.status, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(2, 6))

        tabs = ttk.Notebook(root)
        tabs.pack(fill="x")
        tabs.add(self.build_gather(tabs), text="Gather")
        tabs.add(self.build_setup(tabs), text="Setup (once)")
        tabs.add(self.build_routes(tabs), text="Routes")
        tabs.add(self.build_settings(tabs), text="Settings")

        ttk.Label(root, text="Log").pack(anchor="w", pady=(8, 0))
        self.log_box = scrolledtext.ScrolledText(root, height=12, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)

    def task_button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.task_buttons.append(button)
        return button

    def build_gather(self, parent):
        tab = ttk.Frame(parent, padding=10)

        box = ttk.LabelFrame(tab, text="Gather", padding=8)
        box.pack(fill="x")

        for index, (name, variable) in enumerate(self.targets.items()):
            ttk.Checkbutton(box, text=name, variable=variable).grid(row=0, column=index, sticky="w", padx=6)

        box = ttk.LabelFrame(tab, text="Where", padding=8)
        box.pack(fill="x", pady=8)
        self.route_box = ttk.Combobox(box, textvariable=self.route, state="readonly", width=34)
        self.route_box.grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.route_mode, values=ROUTE_MODES, state="readonly",
                     width=14).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(box, text="Stop after").grid(row=1, column=0, sticky="w", pady=(8, 0))
        row = ttk.Frame(box)
        row.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Spinbox(row, from_=0, to=600, increment=15, width=6, textvariable=self.max_minutes).pack(side="left")
        ttk.Label(row, text="minutes, 0 for never").pack(side="left", padx=4)

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x")
        self.task_button(buttons, "Start", self.start).pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.runner.stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Checkbutton(buttons, text="Show what the model sees", variable=self.preview).pack(side="right")

        ttk.Label(tab, text=STOP_HINT, foreground="grey").pack(anchor="w", pady=(8, 0))

        self.checks_frame = ttk.LabelFrame(tab, text="Ready to run", padding=8)
        self.checks_frame.pack(fill="x", pady=(8, 0))

        return tab

    def build_setup(self, parent):
        tab = ttk.Frame(parent, padding=10)

        box = ttk.LabelFrame(tab, text="1. Gathering bar, needed", padding=8)
        box.pack(fill="x")
        ttk.Label(box, wraplength=560, justify="left", text=(
            "Press Calibrate, start gathering a node in the game before the countdown ends, then draw a "
            "box around the bar shown while gathering. Test shows whether it is recognized.")).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        self.task_button(row, "Calibrate", self.calibrate_bar).pack(side="left")
        self.task_button(row, "Test", self.test_bar).pack(side="left", padx=6)

        box = ttk.LabelFrame(tab, text="2. Character, if it is not in the middle", padding=8)
        box.pack(fill="x", pady=8)
        ttk.Label(box, wraplength=560, justify="left", text=(
            "Click on the feet of the character. Distances to the nodes are measured from there.")
        ).pack(anchor="w")
        self.task_button(box, "Pick the character", self.pick_character).pack(anchor="w", pady=(6, 0))

        box = ttk.LabelFrame(tab, text="3. Minimap, for routes", padding=8)
        box.pack(fill="x")
        ttk.Label(box, wraplength=560, justify="left", text=(
            "Draw a box inside the minimap, map only, no frame or buttons. If the character walks the wrong "
            "way on a route, stand in an open place and press Calibrate walking.")).pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        self.task_button(row, "Pick the minimap", self.pick_minimap).pack(side="left")
        self.task_button(row, "Calibrate walking", self.calibrate_walking).pack(side="left", padx=6)
        self.minimap_label = ttk.Label(box)
        self.minimap_label.pack(anchor="w", pady=(6, 0))

        return tab

    def build_routes(self, parent):
        tab = ttk.Frame(parent, padding=10)

        ttk.Label(tab, wraplength=560, justify="left", text=(
            "Walk to the start of the route, type a name, press Record, then walk the route through the places "
            "full of resources at a normal pace. Press Stop at its end. For a loop, end where you started. "
            "Keep the same minimap zoom when the bot follows it.")).pack(anchor="w")

        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        ttk.Entry(row, textvariable=self.new_route, width=30).pack(side="left")
        self.task_button(row, "Record by walking", self.record).pack(side="left", padx=6)

        self.routes_list = tk.Listbox(tab, height=8)
        self.routes_list.pack(fill="x")
        self.task_button(tab, "Delete the selected route", self.delete_route).pack(anchor="w", pady=(6, 0))

        return tab

    def build_settings(self, parent):
        tab = ttk.Frame(parent, padding=10)

        rows = [
            ("Confidence", ttk.Spinbox(tab, from_=0.1, to=0.95, increment=0.05, width=6, textvariable=self.confidence),
             "lower finds more nodes and makes more mistakes"),
            ("Gathering bar match", ttk.Spinbox(tab, from_=0.3, to=0.95, increment=0.05, width=6,
                                                textvariable=self.bar_threshold),
             "lower it when Test never says GATHERING"),
            ("Mount key", ttk.Entry(tab, textvariable=self.mount_key, width=6),
             "pressed after each node before walking the route, empty to walk"),
            ("Game window", ttk.Entry(tab, textvariable=self.window_name, width=28),
             "a part of its title"),
        ]

        for index, (name, widget, hint) in enumerate(rows):
            ttk.Label(tab, text=name).grid(row=index, column=0, sticky="w", pady=3)
            widget.grid(row=index, column=1, sticky="w", padx=6)
            ttk.Label(tab, text=hint, foreground="grey").grid(row=index, column=2, sticky="w")

        options = [
            (self.avoid_monsters, "Skip the nodes a monster stands next to"),
            (self.hide_hud, "Hide the game interface while gathering (alt+h), calibrate the bar the same way"),
            (self.minimize_ui, "Minimize this window while the bot works, so it never covers the game"),
        ]

        for index, (variable, text) in enumerate(options, start=len(rows)):
            ttk.Checkbutton(tab, text=text, variable=variable).grid(row=index, column=0, columnspan=3,
                                                                     sticky="w", pady=2)

        return tab

    # ---------------------------------------------------------------- state

    def write(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{text}\n")

        # The log would grow forever over a night of gathering.
        if int(self.log_box.index("end-1c").split(".")[0]) > 2000:
            self.log_box.delete("1.0", "500.0")

        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def number(self, variable, default, low, high, name):
        try:
            value = float(variable.get())
        except (ValueError, tk.TclError):
            self.write(f"{name} has to be a number, using {default}")
            value = default

        return min(max(value, low), high)

    def collect(self):
        """
        Read the widgets into the settings, and save them.
        """
        s = self.settings
        s.targets = [name for name, variable in self.targets.items() if variable.get()]
        s.route = "" if self.route.get() == NO_ROUTE else self.route.get()
        s.route_mode = self.route_mode.get()
        s.max_minutes = self.number(self.max_minutes, 0, 0, 100000, "Stop after")
        s.confidence = self.number(self.confidence, 0.5, 0.05, 0.95, "Confidence")
        s.bar_threshold = self.number(self.bar_threshold, 0.7, 0.3, 0.95, "Gathering bar match")
        s.window_name = self.window_name.get().strip()
        s.hide_hud = self.hide_hud.get()
        s.avoid_monsters = self.avoid_monsters.get()
        s.mount_key = self.mount_key.get().strip().lower()[:1]
        s.preview = self.preview.get()
        s.minimize_ui = self.minimize_ui.get()

        for name in s.validate():
            self.write(f"{name} was not valid, reset to its default")

        self.save()
        return s

    def save(self):
        try:
            config.save(self.settings)
        except OSError as e:
            self.write(f"Could not save the settings: {e}")

    def busy(self, busy, hide=False):
        for button in self.task_buttons:
            button.configure(state="disabled" if busy else "normal")

        self.stop_button.configure(state="normal" if busy else "disabled")

        if busy and hide and self.settings.minimize_ui:
            self.iconify()
        elif not busy and self.state() == "iconic":
            self.deiconify()

    def refresh_routes(self, select=None):
        names = route.names()
        self.route_box.configure(values=[NO_ROUTE] + names)
        self.routes_list.delete(0, "end")

        for name in names:
            self.routes_list.insert("end", name)

        if select is not None:
            self.route.set(select)
        elif self.route.get() not in names:
            self.route.set(NO_ROUTE)

    def refresh_checks(self):
        for child in self.checks_frame.winfo_children():
            child.destroy()

        results = checks(self.settings)
        game = results[0]
        self.game.set(f"Albion Online: {game[2]}" if game[1] else f"Albion Online is not running ({game[2]})")

        for row, (name, ok, detail) in enumerate(results):
            ttk.Label(self.checks_frame, text=f"{'✔' if ok else '✘'}  {name}",
                      foreground="green" if ok else "red").grid(row=row, column=0, sticky="w")
            ttk.Label(self.checks_frame, text=detail, foreground="grey", wraplength=420,
                      justify="left").grid(row=row, column=1, sticky="w", padx=8)

    def poll_checks(self):
        # Not behind a running bot, looking for windows is not free on Linux.
        if not self.runner.busy():
            # Follows the name being typed, not only the saved one.
            self.settings.window_name = self.window_name.get().strip() or self.settings.window_name
            self.refresh_checks()

        self.after(CHECK_POLLING, self.poll_checks)

    # ---------------------------------------------------------------- actions

    def start(self):
        s = self.collect()

        if not s.targets:
            messagebox.showinfo(UI_WINDOW_TITLE, "Tick at least one resource to gather")
            return

        if self.runner.start(s, preview=s.preview):
            self.status.set("Starting...")
            self.busy(True, hide=True)

    def calibrate_bar(self):
        s = self.collect()

        if self.runner.snapshot(s, "bar", delay=6, text="Start gathering a node now, screenshot"):
            self.busy(True, hide=True)

    def test_bar(self):
        if self.runner.test_bar(self.collect()):
            self.busy(True, hide=True)

    def pick_character(self):
        if self.runner.snapshot(self.collect(), "character"):
            self.busy(True, hide=True)

    def pick_minimap(self):
        if self.runner.snapshot(self.collect(), "minimap"):
            self.busy(True, hide=True)

    def calibrate_walking(self):
        if self.runner.calibrate_walking(self.collect()):
            self.busy(True, hide=True)

    def record(self):
        name = self.new_route.get().strip()

        try:
            route.Route.folder(name)
        except ValueError as e:
            messagebox.showinfo(UI_WINDOW_TITLE, str(e))
            return

        if name in route.names() and not messagebox.askyesno(UI_WINDOW_TITLE, f"Replace the route {name}?"):
            return

        if self.runner.record(self.collect(), name):
            # Left open for its Stop button, it must not cover the minimap.
            self.write("Keep this window away from the minimap while recording"
                       + (", F12 ends the recording too" if IS_WINDOWS else ""))
            self.busy(True)

    def delete_route(self):
        selection = self.routes_list.curselection()

        if not selection:
            return

        name = self.routes_list.get(selection[0])

        if messagebox.askyesno(UI_WINDOW_TITLE, f"Delete the route {name}?"):
            route.delete(name)
            self.write(f"Route {name} deleted")
            self.refresh_routes()

    # ---------------------------------------------------------------- results

    def on_snapshot(self, purpose, frame):
        if self.state() == "iconic":
            self.deiconify()

        if purpose == "bar":
            Picker(self, frame, "Draw a box around the gathering bar, tight, the whole bar and nothing else",
                   lambda box: self.save_bar(frame, box))
        elif purpose == "character":
            Picker(self, frame, "Click on the feet of the character", self.save_character, mode="point",
                   initial=self.settings.character)
        elif purpose == "minimap":
            Picker(self, frame, "Draw a box inside the minimap: the map only, no frame, no buttons",
                   lambda box: self.save_minimap(frame, box), initial=self.settings.minimap)

    def save_bar(self, frame, box):
        from albion_bot.vision.gather_bar import GatherBar

        try:
            bar = GatherBar.calibrate(frame, box, self.settings.bar_threshold)
        except ValueError as e:
            messagebox.showerror(UI_WINDOW_TITLE, str(e))
            return

        bar.save()
        self.write(f"Gathering bar saved in {paths.GATHER_BAR.relative_to(paths.ROOT)}, press Test to check it "
                   f"while gathering and while not")
        self.refresh_checks()

    def save_character(self, point):
        self.settings.character = [round(point[0], 4), round(point[1], 4)]
        self.save()
        self.write(f"Character at {self.settings.character[0]:.3f}, {self.settings.character[1]:.3f} of the window")

    def save_minimap(self, frame, box):
        self.settings.minimap = [round(v, 4) for v in box]
        self.save()
        self.write("Minimap saved. Routes recorded with another box have to be recorded again")

        try:
            self.minimap_image, _ = photo(minimap.crop(frame, self.settings.minimap), 240, 240)
            self.minimap_label.configure(image=self.minimap_image)
        except ValueError as e:
            self.write(f"Error: {e}")

    def show_preview(self, frame):
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = tk.Toplevel(self)
            self.preview_window.title(OWN_WINDOW_TITLES[1])
            self.preview_label = ttk.Label(self.preview_window)
            self.preview_label.pack()
            ttk.Label(self.preview_window, foreground="grey", padding=4, text=(
                "Green: to gather. Grey: ignored. Red box: monster. Red cross: already gathered. "
                "Keep this window off the game.")).pack(anchor="w")

        self.preview_image, _ = photo(frame, 800, 450)
        self.preview_label.configure(image=self.preview_image)

    def show_stats(self, stats):
        minutes = stats["runtime"] / 60
        rate = stats["gathered"] / minutes * 60 if minutes >= 1 else 0
        self.status.set(f"{stats['state'].capitalize()}  |  {minutes:.0f} min, {stats['gathered']} gathered, "
                        f"{stats['failed']} skipped, {rate:.0f}/hour")

    def read_messages(self):
        preview = None

        while not self.runner.messages.empty():
            kind, payload = self.runner.messages.get()

            if kind == "log":
                self.write(payload)
            elif kind == "error":
                self.write(f"Error: {payload}")
            elif kind == "stats":
                self.show_stats(payload)
            elif kind == "preview":
                # Only the last one is worth drawing.
                preview = payload
            elif kind == "snapshot":
                self.on_snapshot(*payload)
            elif kind == "matrix":
                self.settings.minimap_matrix = payload
                self.save()
            elif kind == "routes":
                self.refresh_routes(select=payload)
            elif kind == "done":
                self.busy(False)

                if self.status.get() == "Starting...":
                    self.status.set("Idle")

                self.refresh_routes()
                self.refresh_checks()

        if preview is not None:
            self.show_preview(preview)

        self.after(MESSAGE_POLLING, self.read_messages)

    def close(self):
        self.collect()
        self.runner.stop()
        self.destroy()


def main():
    App().mainloop()
