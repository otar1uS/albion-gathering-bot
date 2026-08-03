"""
Building a bot out of its parts.

Everything is constructed here and nowhere else, so each module can be handed exactly
what it needs and none of them have to reach for a global. It is also the one place that
knows the order things have to be built in, which is worth having written down once.
"""

from .. import logs
from ..bot.gatherer import Gatherer
from ..control import input as input_module
from ..game import resources
from ..game.mount import MountManager
from ..nav.anti_stuck import AntiStuck
from ..nav.navigator import Navigator
from ..vision import capture as capture_module
from ..vision.detector import Detector
from ..vision.labels import LabelReader
from ..vision.motion import Motion
from ..vision.verify import Verifier

log = logs.get("app.session")


def build(config, targets=None, route=None, on_frame=None):
    """
    Put a bot together.

    :param config: Config.
    :param targets: Resource names to gather, the config's own when left to None.
    :param route: Name of a recorded route to walk, the config's own when left to None.
    :param on_frame: Called with each marked up frame, for a preview window.
    :return: Gatherer, ready to run.
    """
    from ..game.combat import Threats
    from ..nav.routes import Route, RoutePlayer

    wanted = resources.resolve_targets(targets if targets is not None else config.targets)

    capture = capture_module.build(config)
    log.info("game window %s", capture.rect)

    controller = input_module.build(config)
    detector = Detector(config, capture)
    motion = Motion(config)

    # Only built when tiers are actually being filtered on, so nobody who does not want
    # them pays the import of an OCR engine.
    labels = LabelReader(config, capture) if config.labels.enabled else None
    verifier = Verifier(config, capture, controller, labels)
    navigator = Navigator(config, capture, controller, detector)
    mount = MountManager(config, controller, motion)
    anti_stuck = AntiStuck(config, capture, controller, navigator, motion)
    threats = Threats(config, capture, navigator, controller, detector)

    route_name = config.route if route is None else route
    route_player = None

    if route_name:
        route_player = RoutePlayer(Route.load(route_name), capture, navigator,
                                   controller)
        log.info("walking the %r route, %d waypoints a lap", route_name,
                 len(route_player.route.waypoints))

    return Gatherer(
        config=config,
        capture=capture,
        detector=detector,
        verifier=verifier,
        controller=controller,
        navigator=navigator,
        mount=mount,
        motion=motion,
        anti_stuck=anti_stuck,
        targets=wanted,
        threats=threats,
        route_player=route_player,
        on_frame=on_frame,
    )
