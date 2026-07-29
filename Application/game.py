from platform import system

# Name the game window is looked for under. Only a part of the title has to match, so
# the same name covers the window whatever it is exactly called.
#
# It lives in its own module, without a single import beyond the standard library, so
# the interface can read it before numpy and torch are anywhere near being loaded.


def default_window_name():
    """
    :return: Name of the game window, for this platform.
    """
    # The Windows client titles its window "Albion Online Client". The Linux one, run
    # through Flatpak, does not use that title, and the launcher sharing the name is
    # filtered out on its own, so the short name is enough and covers both clients.
    return "Albion" if system() == "Linux" else "Albion Online Client"


DEFAULT_WINDOW_NAME = default_window_name()

# Titles of the windows the bot opens itself. They are never the game, and the interface
# is literally called "Albion gathering bot", which matches the name the game is looked
# for under, so the bot would screenshot itself and hunt for trees in its own log.
UI_WINDOW_TITLE = "Albion gathering bot"
PREVIEW_WINDOW_TITLE = "Founded"
OWN_WINDOW_TITLES = (UI_WINDOW_TITLE, PREVIEW_WINDOW_TITLE)
