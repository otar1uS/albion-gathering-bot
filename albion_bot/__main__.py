"""
python -m albion_bot opens the interface.
"""

from albion_bot.platform import enable_dpi_awareness

# Before any window exists, see enable_dpi_awareness.
enable_dpi_awareness()

from albion_bot.ui.app import main  # noqa: E402

main()
