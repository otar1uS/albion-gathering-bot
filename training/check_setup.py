"""Checks the bot can run on this machine, without needing the game to be open."""

import sys
import traceback

sys.path.insert(0, r"C:\Users\otopk\albion-gathering-bot")

failures = []


def check(name, call):
    try:
        print(f"[ ok ] {name}: {call()}")
    except Exception as e:
        failures.append(name)
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


check("python", lambda: sys.version.split()[0])
check("torch", lambda: __import__("torch").__version__)
check("opencv", lambda: __import__("cv2").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("mss", lambda: __import__("mss").__version__)
check("pywin32", lambda: __import__("win32gui") and "imported")
check("pyautogui", lambda: __import__("pyautogui").__version__)
check("tkinter", lambda: __import__("tkinter").TkVersion)

# The bot's own modules.
check("game module", lambda: __import__("Application.game", fromlist=["x"]).DEFAULT_WINDOW_NAME)
check("resources", lambda: [str(p) for p in
                            __import__("Application.Albion.resources", fromlist=["x"]).PROFILES])


def pointer_check():
    from Application.Interaction import pointer
    ok, detail = pointer.availability()
    if not ok:
        raise Exception(detail)
    return detail


check("mouse backend", pointer_check)


def checks_check():
    from Application.runner import checks
    lines = []
    for name, ok, detail in checks():
        lines.append(f"\n         {'v' if ok else 'x'} {name}: {detail}")
    return "".join(lines)


check("readiness list", checks_check)


def windows_list():
    """Every visible window title, so the exact name of the game can be spotted."""
    import win32gui

    titles = []

    def collect(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                titles.append(title)

    win32gui.EnumWindows(collect, None)
    albion = [t for t in titles if "albion" in t.lower()]
    return f"{len(titles)} open windows, Albion ones: {albion or 'NONE - game not running'}"


check("window listing", windows_list)

print()
print("FAILURES:", failures if failures else "none")
