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

1. **Tier filtering.** The bot already hovers every node and the game prints its name.
   OCR that label and the tier is exact: Rough/Birch/Chestnut/Pine/Cedar/Bloodoak/
   Ashenbark for wood, and so on per resource. This makes the tier sliders real and is
   the single best idea to take from their interface.
2. **Enchanted detection.** Enchanted nodes glow and their name is coloured (green/blue/
   purple/gold). Read the colour of the hover label from the same crop as the tier.
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

## Open work

- Zone retrain: `training/autolabel.py collect` then `label`, fix boxes, retrain. This is
  the real fix for trees; nothing else moves that number much.
- `game/combat.py` has never been exercised against a real aggro.
- Nothing is committed to git. The whole `albion/` package is untracked.
- Client packaging (ONNX + onnxruntime + PyInstaller) — the venv is 1.4GB, far too big to
  hand a non-technical client.
