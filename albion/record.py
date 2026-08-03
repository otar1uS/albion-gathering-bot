"""
Record a farming route by walking it.

    .venv\\Scripts\\python.exe -m albion.record --name forest_lap

Then play normally: click your way around the lap you want the bot to farm. Every left
click inside the game window becomes a waypoint. Press F9 when the lap is done, and run
the bot with:

    .venv\\Scripts\\python.exe -m albion --route forest_lap
"""

import argparse
import sys

from . import logs
from .config import Config
from .nav.routes import Route, record


def main():
    parser = argparse.ArgumentParser(prog="albion.record")
    parser.add_argument("--name", required=True)
    parser.add_argument("--stop-key", default="f9")
    parser.add_argument("--list", action="store_true", help="Show recorded routes.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    logs.setup(config.log_level)
    log = logs.get("record")

    if args.list:
        for name in Route.names():
            log.info("route: %s", name)
        return 0

    route = record(config, args.name, args.stop_key)

    if route.waypoints:
        log.info("done. Run it with: python -m albion --route %s", args.name)

    return 0 if route.waypoints else 1


if __name__ == "__main__":
    sys.exit(main())
