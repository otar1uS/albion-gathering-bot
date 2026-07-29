import sys
from pathlib import Path

# Makes the Application package importable, so "python ui.py" works from the
# Application folder as well as "python Application/ui.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from tkinter import scrolledtext, ttk

from Application.runner import BotRunner, checks, game_window, resource_names

# Time in millisecond between two reads of what the bot has to say.
MESSAGE_POLLING = 150

# Time in millisecond between two looks for the game window.
GAME_POLLING = 3000


class Interface(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Albion gathering bot")
        self.minsize(560, 620)

        self.runner = BotRunner()
        self.resources = {name: tk.BooleanVar(value=name == "tree") for name in resource_names()}
        self.window_name = tk.StringVar(value="Albion Online Client")
        # A string and not a double, the box can hold whatever the user types in it.
        self.confidence = tk.StringVar(value="0.8")
        self.preview = tk.BooleanVar(value=False)
        self.game = tk.StringVar(value="Looking for Albion Online...")

        self.__build()
        self.__refresh_checks()

        self.protocol("WM_DELETE_WINDOW", self.__close)
        self.after(MESSAGE_POLLING, self.__read_messages)
        self.after(0, self.__refresh_game)

    # ---------------------------------------------------------------- interface

    def __build(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, textvariable=self.game).pack(anchor="w")
        ttk.Separator(root).pack(fill="x", pady=8)

        gather = ttk.LabelFrame(root, text="Gather", padding=8)
        gather.pack(fill="x")

        for index, (name, selected) in enumerate(self.resources.items()):
            ttk.Checkbutton(gather, text=name, variable=selected).grid(row=index // 3, column=index % 3,
                                                                       sticky="w", padx=6, pady=2)

        settings = ttk.LabelFrame(root, text="Settings", padding=8)
        settings.pack(fill="x", pady=8)

        ttk.Label(settings, text="Confidence").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Spinbox(settings, from_=0.3, to=0.95, increment=0.05, width=6,
                    textvariable=self.confidence).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(settings, text="lower finds more resources and more mistakes").grid(row=0, column=2, sticky="w")

        ttk.Label(settings, text="Game window").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(settings, textvariable=self.window_name, width=28).grid(row=1, column=1, columnspan=2,
                                                                         sticky="w", padx=6)

        ttk.Checkbutton(settings, text="Show what the model sees, in another window",
                        variable=self.preview).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=4)

        self.start_button = ttk.Button(buttons, text="Start", command=self.__start)
        self.start_button.pack(side="left")

        self.stop_button = ttk.Button(buttons, text="Stop", command=self.__stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)

        self.calibrate_button = ttk.Button(buttons, text="Calibrate the gathering bar", command=self.__calibrate)
        self.calibrate_button.pack(side="right")

        self.checks_frame = ttk.LabelFrame(root, text="Ready to run", padding=8)
        self.checks_frame.pack(fill="x", pady=8)

        ttk.Label(root, text="Log").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(root, height=14, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

    def __refresh_checks(self):
        """
        Show what is missing before the bot can run.
        """
        for child in self.checks_frame.winfo_children():
            child.destroy()

        for row, (name, ok, detail) in enumerate(checks(self.window_name.get())):
            ttk.Label(self.checks_frame, text=f"{'v' if ok else 'x'}  {name}").grid(row=row, column=0, sticky="w")
            ttk.Label(self.checks_frame, text=detail, foreground="grey").grid(row=row, column=1, sticky="w", padx=8)

    def __refresh_game(self):
        """
        Tell if the game is running, over and over.
        """
        window = game_window(self.window_name.get())

        if window is None:
            self.game.set(f"Albion Online is not running, no window named {self.window_name.get()}")
        else:
            self.game.set(f"Albion Online is running, {window}")

        self.after(GAME_POLLING, self.__refresh_game)

    def __write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", f"{text}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- the bot

    def __confidence(self):
        """
        Read the confidence, the box can hold anything the user typed in it.

        :return: Confidence between 0 and 1.
        """
        try:
            value = float(self.confidence.get())
        except (ValueError, tk.TclError):
            value = 0.8
            self.__write(f"Confidence has to be a number, using {value}")

        return min(max(value, 0.05), 0.99)

    def __start(self):
        targets = [name for name, selected in self.resources.items() if selected.get()]

        self.__write(f"Starting on {', '.join(targets) if targets else 'nothing'}")

        if self.runner.start(targets=targets,
                             confidence=self.__confidence(),
                             window_name=self.window_name.get(),
                             preview=self.preview.get()):
            self.__running(True)

    def __stop(self):
        self.runner.stop()

    def __calibrate(self):
        if self.runner.calibrate(window_name=self.window_name.get()):
            self.__running(True)

    def __running(self, running):
        self.start_button.configure(state="disabled" if running else "normal")
        self.calibrate_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def __read_messages(self):
        """
        Show what the bot did since the last read, it runs in another thread and
        tkinter can only be touched from this one.
        """
        while not self.runner.messages.empty():
            kind, text = self.runner.messages.get()

            if kind == "stopped":
                self.__write("Stopped")
                self.__running(False)
                self.__refresh_checks()
            elif kind == "error":
                self.__write(f"Error: {text}")
            else:
                self.__write(text)

        self.after(MESSAGE_POLLING, self.__read_messages)

    def __close(self):
        self.runner.stop()
        self.destroy()


if __name__ == "__main__":
    Interface().mainloop()
