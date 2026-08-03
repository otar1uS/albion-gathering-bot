"""
Save a fresh picture of the gathering bar and work out the threshold to match it at.

Calibrating by hand means starting to gather and hoping the screenshot lands while the
bar is up, and it says nothing about what the match is worth once a few charges are
gone: the bar counts the charges left in the node, so it loses a segment every swing
and a picture of a full one stops matching a half empty one. That is how the saved
picture went stale while still looking fine, and every node ended on "Mooving timed
out".

So the whole thing is filmed instead. The character is sent to a node, the frames from
while it worked are told apart from the frames after it stopped by how much the bar
region moved, a picture is taken from the middle of the run rather than the start, and
it is then matched against every frame of both kinds to see how far apart they really
sit. What comes out is a template and the two numbers a threshold has to fall between.

    .venv\\Scripts\\python.exe training\\calibrate_bar.py --resource tree
"""

import argparse
import sys
from pathlib import Path
from time import sleep, time

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application import paths
from Application.Albion import bar, resources
from Application.Albion.detection import AlbionDetection
from Application.Interaction.interaction import Interaction

# Grey level over which a pixel counts as part of the bar rather than as the game behind
# it. The bars are drawn as a near white outline over a forest that is mostly dark.
BRIGHT = 200

# How many bright pixels above the quiet baseline mean the bar is up. The bar is a thin
# outline a few pixels tall, so it only adds a few dozen bright pixels to a region that
# already holds the name of the character, and asking for a big jump missed it entirely.
BRIGHT_JUMP = 15

# Frames from the start of the run, while the character is still walking over, that set
# what the region looks like with no bar in it.
BASELINE_FRAMES = 8


def region(frame):
    """
    Cut the part of a frame the bar is looked for in.

    :param frame: Grayscale 640x640 frame.
    :return: The searched region of it.
    """
    left, top, right, bottom = bar.searched_region()

    return frame[top:bottom, left:right]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", default="tree")
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--roam", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true",
                        help="Measure without overwriting the saved picture.")
    args = parser.parse_args()

    profile = resources.resolve(args.resource)

    model = AlbionDetection(debug=False, preview=False, targets=[args.resource])
    interaction = Interaction(model)

    model.window_capture.focus()
    sleep(1.5)

    # Most trees the model finds are scenery the game will not let anyone touch, and
    # filming one of those produces a run with no bar in it at all, which is how the
    # earlier attempts at this kept concluding the template was fine or hopeless at
    # random. The game is asked to confirm the node first, the same way the bot does.
    nodes = []

    for attempt in range(args.roam):
        model.window_capture.focus()
        detections, image, _ = model.scan()
        found = [(x, y) for x, y, kind in detections if kind is profile]
        nodes = [(x, y) for x, y in found if interaction.confirmed(x, y, image)]

        if nodes:
            break

        print(f"  {len(found)} {profile} in sight, none of them gatherable, roaming "
              f"({attempt + 1}/{args.roam})")
        interaction._Interaction__roam()
        sleep(interaction.ROAM_WAIT)

    if not nodes:
        sys.exit(f"No gatherable {profile} found after roaming, move the character to a forest")

    print(f"Clicking {profile} at {nodes[0]}, filming for {args.seconds}s")
    interaction.pointer.left_click(*nodes[0])
    interaction._Interaction__park()

    frames = []
    start = time()

    while time() - start < args.seconds:
        # Asked for on every frame. The console this is launched from takes the front
        # back partway through otherwise, and the frames from then on are a picture of
        # that console, which is exactly how the first run of this produced a template
        # cut out of a terminal full of source code.
        model.window_capture.focus()

        frames.append(cv.cvtColor(model._process_image(model.window_capture.screenshot()),
                                  cv.COLOR_BGR2GRAY))
        sleep(0.4)

    # Which frames hold a bar is decided on the bar itself rather than on a comparison
    # with the end of the run. Taking the tail as the idle sample assumed the node ran
    # out before the filming did, and when it did not the whole classification flipped
    # over: every frame was called idle because it matched an idle reference that was
    # itself a picture of the bar. The bar is drawn as a bright outline on a dark game,
    # so counting bright pixels says whether it is there without needing a reference.
    counts = [int((region(f) > BRIGHT).sum()) for f in frames]
    floor, ceiling = min(counts), max(counts)

    # The character is still walking over for the first few seconds, which is the one
    # stretch of the run guaranteed to have no bar in it, so it sets the quiet level.
    baseline = sorted(counts[:BASELINE_FRAMES])[len(counts[:BASELINE_FRAMES]) // 2]
    cut = baseline + BRIGHT_JUMP

    print(f"\nBright pixels in the bar region, one mark a frame "
          f"({floor} to {ceiling}, quiet at {baseline}, bar over {cut}):")
    print("  " + "".join(
        " .:-=+*#@"[min(int((c - floor) / max(ceiling - floor, 1) * 8), 8)] for c in counts))

    working = [i for i, c in enumerate(counts) if c >= cut]

    # Kept whatever happens, because when this goes wrong the picture is the only thing
    # that says why, and the run costs a minute to repeat.
    out = Path("images/bar_hunt")
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(frames)), key=lambda i: counts[i])
    for tag, i in (("dimmest", order[0]), ("brightest", order[-1])):
        cv.imwrite(str(out / f"{tag}.png"), frames[i])
        crop = region(frames[i])
        cv.imwrite(str(out / f"{tag}_region.png"),
                   cv.resize(crop, None, fx=5, fy=5, interpolation=cv.INTER_NEAREST))
    print(f"Wrote {out}/brightest.png and dimmest.png with their regions")

    if not working:
        sys.exit(f"The region never rose {BRIGHT_JUMP} over the quiet level, so no bar was "
                 f"drawn: the character never reached a node it could work on.")

    print(f"\n{len(frames)} frames, {len(working)} of them with the bar up "
          f"(over {cut:.0f} bright pixels)")

    if len(working) < 5:
        sys.exit("The bar region never moved, so the character never worked on a node. "
                 "Check it can reach one and that the interface is not hidden.")

    # Halfway through the working frames, so the picture holds a bar that has already
    # lost a segment or two rather than a brand new one that only matches the first swing.
    middle = frames[working[len(working) // 2]]
    left, top, right, bottom = bar.searched_region()
    template = middle[bar.TOP_Y:bar.BOTTOM_Y, bar.TOP_X:bar.BOTTOM_X]

    print(f"Template {template.shape[1]}x{template.shape[0]}, usable {bar.usable(template)}, "
          f"spread {template.std():.1f}")

    def match(frame):
        result = cv.matchTemplate(region(frame), template, cv.TM_CCOEFF_NORMED)
        return float(cv.minMaxLoc(result)[1])

    gathering = [match(frames[i]) for i in working]
    quiet = set(range(len(frames))) - set(working)
    idle = [match(frames[i]) for i in quiet]

    print(f"\nWhile gathering: min {min(gathering):.3f}  median {sorted(gathering)[len(gathering) // 2]:.3f}"
          f"  max {max(gathering):.3f}   ({len(gathering)} frames)")

    if idle:
        print(f"While idle:     min {min(idle):.3f}  median {sorted(idle)[len(idle) // 2]:.3f}"
              f"  max {max(idle):.3f}   ({len(idle)} frames)")

    if not idle:
        print("No idle frames to compare against, run it again over a node that runs out.")
        return

    floor, ceiling = min(gathering), max(idle)

    if floor <= ceiling:
        print(f"\nThe two overlap ({floor:.3f} against {ceiling:.3f}), so no single number "
              f"tells them apart on every frame. Taking the midpoint, the bot polls the bar "
              f"many times a second and a frame read wrong here and there costs nothing.")

    threshold = round((floor + ceiling) / 2, 2)

    print(f"\nSuggested bar.CONFIDENCE = {threshold}   (it is {bar.CONFIDENCE} now)")

    if args.dry_run:
        print("Dry run, nothing written.")
        return

    paths.IMAGES.mkdir(exist_ok=True)

    if paths.RESOURCE_BAR.exists():
        stale = paths.RESOURCE_BAR.with_suffix(".stale.png")
        cv.imwrite(str(stale), cv.imread(str(paths.RESOURCE_BAR), cv.IMREAD_GRAYSCALE))
        print(f"Kept the old picture as {stale}")

    cv.imwrite(str(paths.RESOURCE_BAR), template)
    cv.imwrite(str(paths.RESOURCE_FRAME), middle)

    print(f"Wrote {paths.RESOURCE_BAR} and {paths.RESOURCE_FRAME}")
    print(f"Now set CONFIDENCE = {threshold} in Application/Albion/bar.py")


if __name__ == "__main__":
    main()
