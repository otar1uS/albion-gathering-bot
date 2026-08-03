"""
Run the bot from a console.

    .venv\\Scripts\\python.exe -m albion --targets tree --minutes 5

Throwing the mouse into the top left corner of the screen stops it, as does ctrl+c.
"""

import argparse
import sys
from threading import Timer

from . import logs
from .app import session
from .config import Config


def main():
    parser = argparse.ArgumentParser(prog="albion")
    parser.add_argument("--targets", default=None,
                        help="Comma separated resource names, all of them when left out.")
    parser.add_argument("--minutes", type=float, default=None,
                        help="Stop after this long, for testing.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--route", default=None,
                        help="Walk a recorded route instead of searching outwards. "
                             "Record one with: python -m albion.record --name <name>")
    parser.add_argument("--read-labels", action="store_true",
                        help="Read the name the game writes over a node, which is what "
                             "gives its tier and whether it is enchanted.")
    parser.add_argument("--min-tier", type=int, default=None)
    parser.add_argument("--max-tier", type=int, default=None)
    parser.add_argument("--only-enchanted", action="store_true")
    parser.add_argument("--no-mount", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Do not ask the game to confirm a node before walking to "
                             "it. Most trees in a forest are scenery, so this mostly "
                             "makes the bot slower.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)

    if args.debug:
        config.log_level = "DEBUG"
    if args.no_mount:
        config.mount.enabled = False
    if args.no_verify:
        config.verify.enabled = False

    # Any tier rule implies reading the names, since the name is the only place the tier
    # is written. Asking to filter without turning the reader on would silently do
    # nothing, which is the sort of quiet no-op this project has been bitten by before.
    if args.min_tier is not None or args.max_tier is not None or args.only_enchanted:
        config.labels.enabled = True

    if args.read_labels:
        config.labels.enabled = True

    if args.only_enchanted:
        config.gathering.only_enchanted = True

    for rule in config.tiers.values():
        if args.min_tier is not None:
            rule.minimum = args.min_tier
        if args.max_tier is not None:
            rule.maximum = args.max_tier

    logs.setup(config.log_level)
    log = logs.get("main")

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None

    try:
        bot = session.build(config, targets, route=args.route)
    except Exception as error:
        log.error("%s", error)
        return 1

    if args.minutes:
        log.info("stopping by itself in %.1f minutes", args.minutes)
        timer = Timer(args.minutes * 60, bot.stop)
        timer.daemon = True
        timer.start()

    try:
        bot.run()
    except KeyboardInterrupt:
        log.info("stopped from the keyboard")

    return 0


if __name__ == "__main__":
    sys.exit(main())
