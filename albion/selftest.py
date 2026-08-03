"""
Check each layer of the bot against the running game, one at a time.

Every subsystem here has a way of failing that looks exactly like working: a capture that
returns the window on top of the game, a detector filtering at a threshold that is not the
one asked for, a click that moves the cursor and never reaches the game. Each of those
cost hours to find by watching the bot behave oddly, so each one is checked directly and
the numbers printed rather than assumed.

Nothing here clicks anything unless --clicks is passed.

    .venv\\Scripts\\python.exe -m albion.selftest
"""

import argparse
import sys

from . import logs
from .config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clicks", action="store_true",
                        help="Also move the mouse, which takes over the pointer.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    logs.setup(config.log_level)
    log = logs.get("selftest")

    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<34} {detail}")
        if not ok:
            failures.append(name)
        return ok

    print("\nConfiguration")
    check("weights present", config.weights_path.exists(), str(config.weights_path))
    check("yolov5 present", config.yolov5_path.joinpath("hubconf.py").exists(),
          str(config.yolov5_path))
    check("targets resolve", True, ", ".join(config.targets) or "everything")

    print("\nCapture")
    from .vision import capture as capture_module

    try:
        capture = capture_module.build(config)
        check("game window found", True, str(capture.rect))
    except Exception as error:
        check("game window found", False, str(error))
        print("\nNothing else can be checked without the game running.")
        return 1

    raised = capture.focus()
    check("window raised", raised, "clicks need this, seeing no longer does")

    picture = capture.grab()
    filled = float((picture[:, :, :3].max(axis=2) >= 8).mean())
    check("screenshot has content", filled > 0.5,
          f"{picture.shape}, {filled:.1%} of it not black")

    print("\nDetection")
    from .vision.detector import Detector

    try:
        detector = Detector(config, capture)
    except Exception as error:
        check("model loads", False, str(error))
        return 1

    check("model loads", True, f"{len(detector.classes)} classes")
    check("suppression floor opened", detector.model.conf <= config.vision.model_floor,
          f"model.conf {detector.model.conf} (yolov5 defaults to 0.25, which would "
          f"silently become the real threshold)")
    check("augmented inference", config.vision.augment,
          "trees are unusable without it, 2 frames of 42 against 14")

    detections, frame = detector.look()
    gatherable = [d for d in detections if d.gatherable]
    check("a frame was scored", frame is not None, f"{len(detections)} detections")

    for detection in detections[:6]:
        print(f"          {detection.label:<9} {detection.confidence:.2f} at "
              f"{detection.screen}  {detection.distance:.0f}px away")

    print("\nMotion")
    from .vision.motion import Motion

    motion = Motion(config)
    motion.add(frame)
    change = motion.add(detector.look()[1])
    check("frames compare", change is not None,
          f"{change:.2f} mean change, moving is over "
          f"{config.anti_stuck.motion_threshold}")

    print("\nInput")
    from .control import input as input_module

    controller = input_module.build(config)
    check("controller built", True, type(controller).__name__)

    try:
        controller.check()
        check("panic corner clear", True, f"cursor at {controller.position()}")
    except Exception as error:
        check("panic corner clear", False, str(error))

    if args.clicks:
        controller.park(capture.rect)
        check("cursor parked", True, str(controller.position()))

        if gatherable:
            print("\nVerification against the game")
            from .vision.verify import Verifier

            verifier = Verifier(config, capture, controller)
            _, clean = detector.look()

            for detection in gatherable[:4]:
                named = verifier.confirm(detection, clean)
                print(f"          {detection.label} {detection.confidence:.2f} -> "
                      f"{'gatherable' if named else 'scenery, the game will not name it'}")
        else:
            print("\nNothing gatherable in sight to verify against.")
    else:
        print("\n  (pass --clicks to also test the pointer and node verification)")

    print()

    if failures:
        log.error("%d checks failed: %s", len(failures), ", ".join(failures))
        return 1

    log.info("every check passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
