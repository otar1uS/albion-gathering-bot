"""
The window the bot is driven from.

Written for somebody who is not going to open a terminal: the checks that have to pass
are shown before anything is started, the settings that matter are on the same screen as
the start button, and the log is the same one that goes to the file so there is never a
question of where to look when something goes wrong.

Nothing here imports torch. Loading the model takes several seconds and the window has to
appear immediately, so the bot is built on its own thread once start is pressed.
"""

import tkinter as tk
from tkinter import messagebox, ttk

from .. import logs
from ..config import Config, TierRule
from ..nav.routes import Route
from .runner import BotRunner, preflight, resource_names

log = logs.get("app.ui")

TITLE = "Albion gathering bot"


class Window(tk.Tk):
    """The interface."""

    def __init__(self):
        super().__init__()

        self.title(TITLE)
        self.minsize(720, 620)

        self.config_data = Config.load()
        self.runner = BotRunner()

        self.targets = {
            name: tk.BooleanVar(value=name in self.config_data.targets)
            for name in resource_names()
        }

        self.tiers = {
            name: {
                "minimum": tk.StringVar(value=str(self.__rule(name).minimum)),
                "maximum": tk.StringVar(value=str(self.__rule(name).maximum)),
            }
            for name in resource_names()
        }

        self.read_labels = tk.BooleanVar(value=self.config_data.labels.enabled)
        self.gather_enchanted = tk.BooleanVar(
            value=self.config_data.gathering.gather_enchanted)
        self.only_enchanted = tk.BooleanVar(
            value=self.config_data.gathering.only_enchanted)

        self.window_title = tk.StringVar(value=self.config_data.window.title)
        self.confidence = tk.StringVar(value=str(self.config_data.vision.confidence))
        self.mount_key = tk.StringVar(value=self.config_data.input.mount_key)
        self.use_mount = tk.BooleanVar(value=self.config_data.mount.enabled)
        self.verify = tk.BooleanVar(value=self.config_data.verify.enabled)
        self.route = tk.StringVar(value=self.config_data.route or "(search outwards)")
        self.status = tk.StringVar(value="Not running")

        self.__build()
        self.__drain()

        self.protocol("WM_DELETE_WINDOW", self.__close)

    def __rule(self, name):
        """
        Tier rule for a resource, made if the settings file predates it.

        :param name: Canonical resource name.
        :return: TierRule.
        """
        if name not in self.config_data.tiers:
            self.config_data.tiers[name] = TierRule()

        return self.config_data.tiers[name]

    # ------------------------------------------------------------------ layout

    def __build(self):
        padding = {"padx": 10, "pady": 6}

        resources_box = ttk.LabelFrame(self, text="What to gather")
        resources_box.pack(fill="x", **padding)

        ttk.Label(resources_box, text="resource", foreground="grey").grid(
            row=0, column=0, padx=8, sticky="w")
        ttk.Label(resources_box, text="lowest tier", foreground="grey").grid(
            row=0, column=1, padx=8)
        ttk.Label(resources_box, text="highest tier", foreground="grey").grid(
            row=0, column=2, padx=8)

        for row, (name, variable) in enumerate(self.targets.items(), start=1):
            ttk.Checkbutton(resources_box, text=name.capitalize(),
                            variable=variable).grid(row=row, column=0, padx=8, pady=3,
                                                    sticky="w")

            for column, which in ((1, "minimum"), (2, "maximum")):
                box = ttk.Spinbox(resources_box, from_=1, to=8, width=4,
                                  textvariable=self.tiers[name][which])
                box.grid(row=row, column=column, padx=8, pady=3)

        ttk.Checkbutton(
            resources_box,
            text="Read the name the game writes on a node, which is where its tier and "
                 "enchantment come from (adds about a second per node)",
            variable=self.read_labels).grid(row=len(self.targets) + 1, column=0,
                                            columnspan=4, sticky="w", padx=8, pady=(6, 2))

        enchant = ttk.Frame(resources_box)
        enchant.grid(row=len(self.targets) + 2, column=0, columnspan=4, sticky="w",
                     padx=8, pady=(0, 6))

        ttk.Checkbutton(enchant, text="Gather enchanted",
                        variable=self.gather_enchanted).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(enchant, text="Only enchanted",
                        variable=self.only_enchanted).pack(side="left")

        settings = ttk.LabelFrame(self, text="Settings")
        settings.pack(fill="x", **padding)

        ttk.Label(settings, text="Game window").grid(row=0, column=0, sticky="w", padx=8,
                                                     pady=4)
        ttk.Entry(settings, textvariable=self.window_title, width=28).grid(
            row=0, column=1, sticky="w", padx=6)

        ttk.Label(settings, text="Confidence").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Entry(settings, textvariable=self.confidence, width=6).grid(
            row=0, column=3, sticky="w", padx=6)

        ttk.Label(settings, text="Mount key").grid(row=1, column=0, sticky="w", padx=8,
                                                   pady=4)
        ttk.Entry(settings, textvariable=self.mount_key, width=6).grid(
            row=1, column=1, sticky="w", padx=6)

        ttk.Checkbutton(settings, text="Use the mount for long trips",
                        variable=self.use_mount).grid(row=1, column=2, columnspan=2,
                                                      sticky="w", padx=8)

        ttk.Checkbutton(
            settings,
            text="Ask the game to confirm a node before walking to it (recommended, "
                 "most trees in a forest cannot be gathered)",
            variable=self.verify).grid(row=2, column=0, columnspan=4, sticky="w", padx=8,
                                       pady=4)

        ttk.Label(settings, text="Route").grid(row=3, column=0, sticky="w", padx=8,
                                               pady=4)
        self.route_box = ttk.Combobox(settings, textvariable=self.route, width=26,
                                      values=["(search outwards)"] + Route.names())
        self.route_box.grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(settings,
                  text="Record one with: python -m albion.record --name <name>").grid(
            row=3, column=2, columnspan=2, sticky="w", padx=8)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", **padding)

        self.check_button = ttk.Button(buttons, text="Check setup", command=self.__check)
        self.check_button.pack(side="left", padx=4)

        self.start_button = ttk.Button(buttons, text="Start", command=self.__start)
        self.start_button.pack(side="left", padx=4)

        self.stop_button = ttk.Button(buttons, text="Stop", command=self.__stop,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=4)

        ttk.Button(buttons, text="Save settings", command=self.__save).pack(side="left",
                                                                            padx=4)

        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=14)

        log_box = ttk.LabelFrame(self, text="Log")
        log_box.pack(fill="both", expand=True, **padding)

        self.log_view = tk.Text(log_box, height=18, wrap="word", state="disabled")
        self.log_view.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(log_box, command=self.log_view.yview)
        scroll.pack(side="right", fill="y")
        self.log_view.configure(yscrollcommand=scroll.set)

        self.__write("Press Check setup first. The bot needs Albion Online running and "
                     "your character standing in the world, not on the login screen.")

    # ------------------------------------------------------------------ actions

    def __gather_config(self):
        """
        Fold what the window says into the settings.

        :return: Config.
        """
        data = self.config_data

        data.window.title = self.window_title.get().strip() or data.window.title
        data.input.mount_key = self.mount_key.get().strip() or data.input.mount_key
        data.mount.enabled = bool(self.use_mount.get())
        data.verify.enabled = bool(self.verify.get())
        data.targets = [name for name, chosen in self.targets.items() if chosen.get()]

        chosen_route = self.route.get().strip()
        data.route = "" if chosen_route in ("", "(search outwards)") else chosen_route

        data.labels.enabled = bool(self.read_labels.get())
        data.gathering.gather_enchanted = bool(self.gather_enchanted.get())
        data.gathering.only_enchanted = bool(self.only_enchanted.get())

        for name, boxes in self.tiers.items():
            rule = self.__rule(name)

            for which, variable in boxes.items():
                try:
                    setattr(rule, which, min(max(int(variable.get()), 1), 8))
                except ValueError:
                    self.__write(f"{name} {which} tier has to be a whole number, "
                                 f"keeping {getattr(rule, which)}")

                variable.set(str(getattr(rule, which)))

            if rule.minimum > rule.maximum:
                rule.minimum, rule.maximum = rule.maximum, rule.minimum
                boxes["minimum"].set(str(rule.minimum))
                boxes["maximum"].set(str(rule.maximum))

        if data.gathering.only_enchanted and not data.labels.enabled:
            # Enchantment is read off the colour of the name, so asking for enchanted
            # only without the reader would quietly gather everything instead.
            data.labels.enabled = True
            self.read_labels.set(True)
            self.__write("Turned name reading on, it is where enchantment is read from.")

        try:
            # The box holds whatever was typed into it, so a number that is not one is
            # worth saying so about rather than crashing the bot ten seconds later.
            data.vision.confidence = min(max(float(self.confidence.get()), 0.05), 0.99)
        except ValueError:
            self.__write(f"Confidence has to be a number, keeping "
                         f"{data.vision.confidence}")

        self.confidence.set(str(data.vision.confidence))

        return data

    def __check(self):
        config = self.__gather_config()
        self.__write("")

        ok = True

        for name, passed, detail in preflight(config):
            self.__write(f"{'OK   ' if passed else 'FAILED'}  {name}: {detail}")
            ok = ok and passed

        self.__write("Ready to start." if ok
                     else "Fix what failed above, then check again.")

    def __start(self):
        config = self.__gather_config()

        if not config.targets:
            messagebox.showwarning(TITLE, "Pick at least one resource to gather.")
            return

        if self.runner.start(config, config.targets):
            self.status.set("Running. Throw the mouse into the top left corner to stop.")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")

    def __stop(self):
        self.runner.stop()
        self.status.set("Stopping...")

    def __save(self):
        path = self.__gather_config().save()
        self.__write(f"Settings saved to {path}")

    def __close(self):
        if self.runner.is_running():
            if not messagebox.askokcancel(TITLE, "The bot is running. Stop it and quit?"):
                return

            self.runner.stop()

        self.destroy()

    # ------------------------------------------------------------------ plumbing

    def __write(self, text):
        self.log_view.configure(state="normal")
        self.log_view.insert("end", text + "\n")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def __drain(self):
        """Move whatever the bot has said onto the screen, then ask again shortly."""
        try:
            while True:
                kind, text = self.runner.messages.get_nowait()

                if kind == "stopped":
                    self.status.set("Not running")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                elif text:
                    self.__write(text)
        except Exception:
            # An empty queue, which is the usual case rather than a problem.
            pass

        self.after(150, self.__drain)


def main():
    logs.setup()
    Window().mainloop()


if __name__ == "__main__":
    main()
