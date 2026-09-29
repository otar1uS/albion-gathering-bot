# Albion gathering bot

Watches the game the way a person does — a screenshot, a trained eye, a mouse — and
gathers wood, stone, ore, fiber and hide on its own. No game files are touched, no
memory is read, no packets are sniffed: everything the bot knows it learned from pixels.

> Automating Albion Online is against its terms of service and accounts caught doing it
> get banned. Nothing in this design makes that impossible, only less obvious.

## Starting it

```
.venv\Scripts\python.exe -m albion.app.ui             the window
.venv\Scripts\python.exe -m albion --targets tree     the console
.venv\Scripts\python.exe -m albion.selftest           check every layer without gathering
```

Have the game running, windowed, with your character standing in the world. Press
**Check setup** in the window, then **Start**. Stop it any time by throwing the mouse
into the top left corner of the screen.

## Farming routes

Record a lap once by walking it yourself, then let the bot repeat it:

```
.venv\Scripts\python.exe -m albion.record --name forest_lap
    ...walk the lap by clicking in the game, press F9 when done...
.venv\Scripts\python.exe -m albion --route forest_lap
```

Waypoints are stored as fractions of the window, so a recorded route survives the window
being moved or resized. On playback the route only does the walking: everything found
along the way still goes through the same verification and gathering as ever, so the
route drifting a little costs nothing.

## How it works

```
albion/
  config.py        every tunable number, saved to config.json, each with its why
  vision/
    capture.py     photographs the game window itself (PrintWindow), so a window
                   covering the game does not become what the model sees
    detector.py    yolov5 weights (best_merged.pt), augmented inference, per class
                   confidence thresholds
    verify.py      hovers a found node and asks the game: a real resource gets its
                   name drawn over it, scenery gets nothing
    motion.py      frame differencing: is the character walking, working, or is the
                   game not rendering at all (login screen, loading, disconnect)
  control/input.py the mouse: move, wait, then click, because the game resolves what
                   is under the cursor before it reads the button
  game/
    resources.py   what each resource is called across datasets, its timeouts, its
                   priority, where to aim on it (trees are clicked on the trunk)
    mount.py       rides to anything far away, knows gathering dismounts by itself
    combat.py      walks away from monsters that come to the character; the bot
                   cannot read its own health, so it does not pretend it can fight
  nav/
    navigator.py   click to walk, expanding square search, never clicks the interface
    anti_stuck.py  notices repeating, wandering and looping, recovers in escalating
                   steps, and stops honestly when nothing works
    routes.py      record and replay farming laps
  bot/gatherer.py  the state machine: SCANNING, VERIFYING, TRAVELLING, GATHERING,
                   ROAMING, RECOVERING
  app/             the window, the runner thread, the wiring
```

Detections the model is unsure of are confirmed by hovering before anything walks
anywhere; detections it is very sure of are tested by clicking even when the hover finds
nothing, because the click is the truth: a real node starts a gathering animation and
scenery does not. A node is clicked until it gives nothing back, so trees are chopped to
the last charge rather than abandoned half done.

Everything it prints goes to `albion_bot.log` as well as the window, so an overnight run
leaves evidence.

## Training

The weights are `best_merged.pt`, trained on this machine's Intel Arc A770 from six
merged public datasets (`training/`, `datasets/albion_merged`). Trees are the thinnest
class in that data, which is why they get their own lower threshold and augmented
inference. To make the model better at *your* zone:

```
.venv\Scripts\python.exe training\autolabel.py collect --minutes 20   play normally
.venv\Scripts\python.exe training\autolabel.py label                  model pre-labels
    ...fix the boxes in a labelling tool...
resume_training.bat                                                   retrain
```

The previous implementation lives on in `Application/` until the new one has earned its
keep. This project began as a fork of
[Michelprogram/albion-gathering-bot](https://github.com/Michelprogram/albion-gathering-bot);
the detection idea and the yolov5 submodule arrangement come from there.

## The experimental rewrite, `albion_bot/`

A second, separate bot lives in `albion_bot/`. It was written away from the game and has only
run against a simulated one (`tests/`), never the real one. `albion/` above is the one that works.
Its README is [albion_bot/README.md](albion_bot/README.md), and CLAUDE.md explains how the two
compare and what is worth taking from it.
