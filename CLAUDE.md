# Albion gathering bot — working notes

A screen-reading gathering bot for Albion Online. A screenshot, a YOLOv5 model, a mouse.
No game memory is read, no packets are captured, no anti-cheat is touched.

Automating Albion is against its terms of service and accounts get banned for it. The
user knows and has decided; say it once when it is genuinely new information, then get on
with the work.

---

## Run it

```
.venv\Scripts\python.exe -m albion.app.ui                  the window
.venv\Scripts\python.exe -m albion --targets tree          console, one resource
.venv\Scripts\python.exe -m albion --route forest_lap      walk a recorded lap
.venv\Scripts\python.exe -m albion.selftest [--clicks]     check every layer, gathers nothing
.venv\Scripts\python.exe -m albion.record --name mylap     record a route, F9 to stop
```

`--read-labels` reads the name the game writes on a node, which is where its tier and
enchantment come from; `--min-tier`/`--max-tier`/`--only-enchanted` filter on it and turn
the reader on by themselves. `--no-mount`, `--no-verify`, `--minutes`, `--debug` are the
testing flags.

Settings live in `config.json` (`albion/config.py` holds the defaults and the reasons).
Everything printed also goes to `albion_bot.log`.

## Two copies of the repository

`~/code/game-automation/albion-gathering-bot` is the git checkout and where edits are
made. `C:\Users\otopk\albion-gathering-bot` (`/mnt/c/...` from here) is where the bot
actually runs, because the game, the GPU and the venvs are on Windows.

**After every edit, copy the changed files across before testing.** Nothing syncs
automatically, and a test that silently ran the old file has wasted an hour more than
once.

```bash
for d in . vision control game nav bot app; do
  cp albion/$d/*.py /mnt/c/Users/otopk/albion-gathering-bot/albion/$d/ 2>/dev/null
done
```

Reading `/mnt/c` from WSL **serves stale cached data**. A file that looks frozen usually
is not; check from the Windows side before believing it.

## Layout

```
albion/
  config.py          every tunable, JSON-backed, each with why it is that value
  logs.py            file + UI queue
  geometry.py        Rect, and the screen <-> 640px-model coordinate conversions
  vision/
    capture.py       PrintWindow(PW_RENDERFULLCONTENT) — the window's own pixels
    detector.py      yolov5, augmented inference, per-class thresholds, Detection
    verify.py        hover a node, see if the game writes its name over it
    motion.py        frame differencing: walking? working? game not rendering at all?
  control/input.py   move -> wait -> click. Controller backends (pyautogui / ydotool)
  game/
    resources.py     ResourceProfile: aliases, timeouts, priority, confidence, aim_depth
    mount.py         ride when far, gathering dismounts by itself
    combat.py        walk away from monsters; the bot cannot read health so cannot fight
  nav/
    navigator.py     click-to-walk, expanding-square search, interface exclusion zones
    anti_stuck.py    Repeater / Wanderer / Looper, escalating recovery, honest give-up
    routes.py        record + replay laps, waypoints as window fractions
  bot/
    states.py        State enum + the allowed transitions
    gatherer.py      the machine
  app/               ui.py (tkinter), runner.py (thread + preflight), session.py (wiring)
training/            dataset download, local Arc training, autolabel, live probes
Application/         the previous implementation, kept as fallback. Do not extend it.
```

---

## Hard-won facts — do not "clean these up"

Each of these was found by watching the live game, and each looked like correct code.

1. **`pyautogui.leftClick(x, y, interval=…)` does not wait before clicking.** `interval`
   is the gap between *repeated* clicks and it sleeps *after* the button goes down. The
   game resolves what is under the cursor before it reads the button, so the click landed
   wherever the cursor had previously been. **The character never moved.** Always
   `move()` → `wait(click_delay)` → `click()`, and move with a `duration` so the pointer
   travels rather than teleports.

2. **Never press `alt+H`.** It hides the entire HUD, including the gathering bar the bot
   then waits for, so every node ended in "moving timed out". It also puts the model
   off-distribution: every training frame has the interface on screen.

3. **yolov5's `AutoShape.conf` defaults to 0.25** and drops boxes before the bot sees
   them. That silently becomes the real threshold — a resource asking for less never gets
   the chance. Keep `model.conf` at `vision.model_floor` (0.10) and filter per class in
   `Detector.look`.

4. **Augmented inference is not optional for trees.** 2 frames of 42 hold an actionable
   tree without it, 14 with it; median best confidence 0.16 → 0.36. Costs 157ms vs 69ms,
   which is fine at ~2 FPS. Do **not** raise `image_size` above 640 to compensate — the
   weights were trained at 640 and 960 measured far worse (5/42).

5. **A frame must enter the motion history exactly once.** `Gatherer.__look` is the only
   caller of `Motion.add`. Adding a frame twice compares it with itself, which reads as a
   world where nothing ever moves; every walk measured 0.00 and every reached node was
   reported as having given nothing.

6. **Most trees in a forest are scenery.** Every dataset only ever labelled *gatherable*
   trees, so decorative ones were never marked and the model finds both. Of six it picked
   out of the farming zone, one was a node. The game itself knows: hovering a real node
   writes its name over it (+42 bright pixels measured; scenery +0, threshold +15).
   **Hover the trunk, not the box centre** — `aim_depth` 0.68 for trees; the centre of a
   tall tree is canopy with gaps the ground shows through.

7. **The verifier has false negatives, so it is not the last word.** Anything at or above
   `verify.trust_confidence` (0.60) gets clicked anyway, because the click is the truth:
   a real node starts a gathering animation and scenery does not. A node that yields zero
   charges is then put aside for the scenery cooldown. Without this fallback a verifier
   that is wrong for any reason silently switches the whole bot off — which happened.

8. **Losing sight of a node is not the same as having emptied it.** The model drops a box
   every few frames. Re-click the last known spot; `lost_attempts` (3) misses in a row
   ends it. Treating one miss as completion is what left half-chopped trees standing.

9. **A frozen frame means the game is not being played**, not that the character is
   still. A quiet forest measures 0.6–1.0 mean change between frames; a login screen,
   loading screen or disconnect measures exactly 0.00. Handled in `Gatherer.__in_game`,
   which waits rather than stopping.

10. **The gathering-bar template approach is dead.** `Application/Albion/bar.py`
    template-matched a fixed 36×12 region; it went stale the moment the camera zoom
    changed, the search region contains the always-present character nameplate which
    swamps every statistic, and it reported every node emptied in exactly 3.1s. Motion
    differencing replaced it. Do not resurrect it.

## Numbers that have been measured

| | |
|---|---|
| Best verified run | 52 charges / 5 trees / 6 min (~520 an hour), trees emptied to 16 charges |
| Inference | 69ms plain, 157ms augmented, CPU, 640px |
| Training data | tree **560** boxes vs ore **11,650** — trees are 21× thinner, hence their own 0.30 threshold |
| Model | `best_merged.pt`, val mAP50 0.878 all / 0.841 tree; early-stopped epoch 48 of 78 |
| Capture | game covered by a console: PrintWindow 2.4% black, screen-grab 41.5% |

## Testing

Test against the running game — this project has no meaningful unit tests and unverified
claims here have been wrong more often than right. `python -m albion.selftest` checks
each layer separately and prints numbers rather than verdicts. `training/probe_*.py` are
one-question diagnostics (hover response, capture method, aim point) worth copying when a
new question comes up.

When something behaves oddly, **save the frame and look at it**. Three separate
dead-ends this session were solved in one glance and lost an hour each to reasoning about
statistics instead: the terminal covering the game, the character standing on the login
screen, the canopy-versus-trunk aim.

## Style

Comments explain *why*, especially where the code looks wrong but is not (the BGR swap in
`detector.process`, the alias ordering in `resources.py`, the click sequence). Match the
surrounding prose: full sentences, no bullet-point comments, no restating the code.
Docstrings use `:param:`/`:return:`. Prefer measured statements over adjectives — "2 of
42 frames against 14" beats "much better".

---

## Where this is going

The user's reference is the GaripFsh interface (albionfishbot.com): one window, tabs down
the side, everything configurable in it. Their Gather tab, for reference:

- Live stats: Food Status, Total Gathered, Session Time, Waypoints
- Route dropdown, **Start [F5]**, **Record route [F1]**, Open Inventory, Map Editor
- Per resource (Wood/Stone/Ore/Fiber): enable + **Min/Max tier sliders T1–T8**
- Toggles: gather enchanted, only enchanted, auto fight, auto dodge, auto food, auto
  reconnect
- Logs & waypoints pane; save/load settings

**Achievable from pixels, roughly in order of value:**

1. ~~**Tier filtering.**~~ **Built** — see "Reading the node's name" below.
2. ~~**Enchanted detection.**~~ **Built**, from the colour of the same hover label.
   Never yet seen against a genuinely enchanted node, so the colours in
   `game/tiers.ENCHANTMENT_COLOURS` are still nominal.
3. Session statistics, hotkeys, save/load — the plumbing already exists.
4. Auto reconnect — `__in_game` already detects it; add the login click sequence.
5. Auto food — read the buff row; needs a template per food icon.
6. Map editor / waypoint editing on top of `nav/routes.py`.
7. Tabbed single-window UI in place of the current one-page `app/ui.py`.

**Not being built, and why.** Their site states their approach plainly: fishing works by
"listening to the game's network data", gather is "network-based", and there is a
"BattlEye bypass" using "advanced memory manipulation and process hiding" with
"kernel-level protection against detection". That is packet capture, memory manipulation
and anti-cheat evasion — the Proxy and Network tabs are the plumbing for it. This project
does not do any of it, which is a deliberate scope decision and also the reason some of
their features cannot be matched: exact tier and enchantment for every node on the map,
ESP/radar, and true world-coordinate pathfinding all come out of the packet stream, not
off the screen. Screen reading can approximate the first two for the node under the
cursor, and cannot do the rest. Say so plainly rather than promising parity.

## Reading the node's name — tier and enchantment

Built 2026-08-03, live-verified as far as noted. Two new files plus wiring:

- `game/tiers.py` — the vocabulary. `NAMES` is tier → name per resource (wood 1 rough
  logs … 8 whitewood logs, and so on; ore and fiber have no tier 1, which is why those
  tables start at 2). `identify()` normalises what was read and fuzzy-matches it at
  `CLOSENESS` 0.72, because OCR turns `l` into `1` often enough that an exact match
  throws away good readings. `normalise()` splits before a capital: the tooltip is tight
  and the engine returns `BirchLogs` as often as `Birch Logs`. `wanted()` applies the
  rules; `restrictive()` answers whether the user actually asked for anything to be
  skipped, which decides what an unreadable name means.
- `vision/labels.py` — `LabelReader`, RapidOCR lazily loaded, cropping **at native
  resolution** from `capture.grab()` rather than from the 640px model frame; measured
  0.86 confidence and 616ms that way. Enchantment comes from the stroke colour of the
  same crop.
- Config gained `TierRule` (per resource, min/max) and `LabelConfig`; `Config.from_dict`
  gained dict-of-dataclass handling to round-trip `tiers`. UI gained a min/max spinbox
  per resource and the three checkboxes. CLI gained `--read-labels --min-tier --max-tier
  --only-enchanted`, and any of the filters turns the reader on by itself.

Verified live: `T2 Rugged Hide`, `T2 BirchLogs`.

**Two facts found the hard way here, same class as the list above.**

11. **OCR is the primary verification signal now, not the bright-pixel count.** The
    first `--min-tier 4` run skipped nothing at all. The log said why: every node went
    down the trust-click path with `+0 bright`, so `last_named` was never set and the
    filter had nothing to filter on. Counting bright pixels says "scenery" for nodes
    that then give up sixteen charges — the writing is drawn over whatever is behind it
    and over pale ground it barely brightens anything. A name that matches the
    vocabulary cannot be wrong in that direction. `verify.confirm` asks the reader
    first, and the brightness only rescues an unreadable one.
12. **Trust-click is disabled while a tier filter is set.** Clicking an unnamed node on
    confidence alone is right when the user wants everything and wrong when they asked
    for T5 and up — quietly ignoring the filter is the worse of the two mistakes. Both
    halves are logged.

## Open work

**Pick up here.** `Gatherer.__relocate()` and the one-shot retry in `__verify()` are
written and syntax-checked but **have never been run**. They exist because a detection is
already a second or so old by the time the cursor has travelled to it and the tooltip has
had time to appear, and a fox does not wait: the hover lands on the grass it was standing
on. That is the remaining `+0 bright, read as ''` population — nodes that yield charges
anyway. Sync, then:

```
.venv\Scripts\python.exe -u -m albion --targets tree,hide --read-labels --minutes 6 --no-mount
```

and grep `albion_bot.log` for `read as|it is |skipping|took|charge`. Read the log file
directly rather than grepping the console — a grep pattern hid the evidence twice. If the
hover success rate rises, follow with a `--min-tier 4` run and confirm nodes are actually
skipped.

Then, in order:

- Zone retrain: `training/autolabel.py collect` then `label`, fix boxes, retrain. This is
  the real fix for trees; nothing else moves that number much.
- Tabbed single-window UI, session statistics, F5/F1 hotkeys — roadmap items 3 and 7.
- `game/combat.py` has never been exercised against a real aggro.
- Client packaging (ONNX + onnxruntime + PyInstaller) — the venv is 1.4GB, far too big to
  hand a non-technical client.

The whole `albion/` package is committed as of `c1358e9`, the relocate retry included.
Remember the Windows copy does not update itself — sync before testing.

---

## `albion_bot/`: a second bot, written 2026-09-29 away from the game

On 2026-09-29 a separate Linux checkout that had never pulled the August work above got a
full rewrite into a new package, `albion_bot/` (with `tests/`, `tools/`, `scripts/`,
`start.bat`, and its own README in `albion_bot/README.md`). The two histories met in a merge;
nothing of either side was dropped. **`albion/` stays the bot that works.** `albion_bot/` has
passed 66 tests against a simulated game (`tests/fake_game.py`) and has **never touched the
real one**.

### Where it contradicts the hard-won facts above: the facts win

- **Gathering signal.** `albion_bot/vision/gather_bar.py` template-matches the gathering bar,
  with a box the user draws and matching tolerant to window size. Fact 10 says this approach is
  dead (zoom changes, the character's nameplate swamps it). Motion differencing is what works.
- **Minimap.** `albion_bot/navigation/route.py` assumes the minimap *scrolls* under an arrow
  fixed in its centre, and records routes as minimap patches. `albion/nav/minimap.py`, measured
  live, finds a *moving* arrow on a *fixed* minimap. If that is right, `albion_bot`'s routes do
  not work as designed.
- **Capture.** `albion_bot` grabs the screen (mss) and minimizes its own window. `albion` uses
  PrintWindow, measured at 2.4% black under a covering window against 41.5% for screen grabs.
- **Scenery trees.** `albion_bot` has no hover or OCR verification (facts 6, 7, 11), so it
  would click decorative trees.
- **Model format.** `albion_bot` loads Ultralytics weights (YOLO26, OpenVINO export).
  `best_merged.pt` is a yolov5 checkpoint and **cannot** be loaded by it. Trying it live means
  retraining with `tools/train.py`.

### What in it is worth taking into `albion/`, most valuable first

1. **Camera-tracking memory: the fix for a real bug in `albion/`.** `Gatherer.__ignore`
   stores *screen* positions and never moves them, but the camera follows the character, so
   after any walk every ignored spot points at the wrong place. The bot can walk back to
   emptied nodes and skip live ones that slid under an old spot. `albion_bot/vision/tracking.py`
   (`camera_shift`, phase correlation over the middle of the frame with the character painted
   out) measures the world's slide between two frames to sub-pixel precision up to ~500px.
   `albion_bot/bot/memory.py` shifts the remembered spots with it. The lesson that cost a bug:
   **measure between every pair of consecutive frames and add the steps up.** One jump across
   half a screen aliases and put a node 750px off. It fits next to `Motion` (fact 5: each
   frame still enters `Motion` exactly once). Port it, then check it live with a saved frame
   and the ignored spots drawn on it.
2. **The simulator** (`tests/fake_game.py`): a scrolling world, nodes with charges, stumps the
   model still detects, and simulated time. It is how the memory bug above was found and
   proven fixed (40 random layouts, zero clicks on stumps). `albion` has no unit tests; a copy
   adapted to `Gatherer` would catch regressions without the game.
3. **Walk clicks that avoid boxes** (`Bot.walk_point`/`near_box` in `albion_bot/bot/engine.py`):
   route and roam clicks are turned aside when they would land on a node or monster, which
   would otherwise start a gather or an attack.
4. **An Ultralytics/OpenVINO model on the Arc.** Only as a measured comparison with
   `training/compare_models.py`, never on faith: fact 4 (augmented inference) and the tree
   thresholds were tuned on yolov5.

Do **not** take: the bar template, the mss capture, or the scrolling-minimap routes.

## Next session on Windows: start here

1. Pull, then sync the Windows copy (see "Two copies of the repository" above). The yolov5
   submodule is unchanged.
2. Carry on with **Open work** above: the never-run `__relocate()` retry. That is the
   established next step.
3. Then port item 1 of the list above (camera-tracking memory) into `albion/bot/gatherer.py`,
   verify it with saved frames, and run a timed session comparing charges per hour and
   re-clicks on emptied nodes before and after.
4. Decide with the user what happens to `albion_bot/`: mine it for parts (recommended) or
   delete it. Keeping two bots long-term costs more than it gives.

**Do not run `scripts\setup_windows.ps1` in the working `.venv`.** It installs the `xpu` torch build and newer packages into `.venv`, which is the environment `albion/` runs from. Give it its own folder first, or install by hand.

`albion_bot/` tests: `.venv\Scripts\python -m pytest -q tests`. They need `ultralytics`,
`opencv-python` and `numpy`, all already in `requirements.txt`. None of them need the game.
