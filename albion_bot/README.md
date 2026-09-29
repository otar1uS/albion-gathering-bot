<div id="top"></div>

<div align="center">
    <img src="../ressources/logo.png" alt="Logo" width="80" height="80" />
    <h3 align="center">Albion gathering bot</h3>
    <p align="center">Finds resources with a YOLO26 model, walks recorded routes, gathers what it sees.</p>
    <img src="../ressources/prediction.jpg" alt="prediction" />
</div>

## What it does

The bot only reads the screen and moves the mouse, like a player would. It never touches the
memory or the network traffic of the game.

1. It grabs the game window and a YOLO model finds the trees, rocks, ore, fiber, hides and
   monsters in it.
2. It clicks the closest wanted node, waits for the **gathering bar** to show up, then waits
   for it to go away, which means the node is empty.
3. It remembers the emptied nodes and follows them as the camera moves, so it never walks back
   to a stump.
4. When nothing is in sight, it walks a **route** you recorded once by walking it, and uses the
   minimap to know where it is.

It stops when you press **F12** (Windows) or **Stop**, when you throw the mouse into the top left
corner of the screen, when the time you set is over, or when nodes keep ending right away
(bags full or too heavy).

> Botting is against the Albion Online terms of service and accounts get banned for it.
> Use it at your own risk, on an account you can afford to lose.

## Setup on Windows (Intel Arc)

1. Install **Python 3.13** from python.org, with *tcl/tk* and the *py launcher* ticked.
2. Update the **Intel Arc driver**. The GPU build of torch needs a recent one.
3. In the repository folder:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```

   It creates `.venv`, installs torch for the Arc (`xpu`) and the rest, prints whether the GPU is
   seen, and runs the tests. Add `-Cpu` on a machine without an Arc.
4. Start the bot with **`start.bat`**.

Run the game **windowed** or **borderless windowed**. If the game runs as administrator, the bot
has to as well, or Windows drops its clicks.

<details>
<summary>Setup on Linux</summary>

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
scripts/start.sh
```

X11 needs nothing more. Wayland needs **Hyprland**, **grim** and **ydotool**, since no program is
allowed to read the screen or move the cursor there:

```bash
sudo pacman -S --needed tk ydotool grim
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
    | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input $USER          # then log out and back in
systemctl --user enable --now ydotoold
```

There is no global stop key on Linux. Use Stop or the corner of the screen.
</details>

## First time: the model

The bot needs a trained model in `models/`. Nobody has to take screenshots by hand for it.

### 1. Public datasets

Albion screenshots already labeled by other people are shared on
[Roboflow Universe](https://universe.roboflow.com/search?q=albion). Downloading them needs the free
API key of a Roboflow account (https://app.roboflow.com/settings/api), set in your shell only:

```powershell
$env:ROBOFLOW_API_KEY = "..."        # bash: export ROBOFLOW_API_KEY=...
.venv\Scripts\python tools\dataset.py download
.venv\Scripts\python tools\dataset.py merge
```

`merge` renames their classes (`rough log`, `fiber3`, `iron ore`, `travertine`, `Enemys`...) to
the six the bot knows, `tree stone ore fiber hide monster`, drops the rest and the duplicates, and
prints how many boxes each class got.

### 2. Train on the Arc

```powershell
.venv\Scripts\python tools\train.py --openvino
```

It trains `yolo26s` on the Arc (`xpu`), copies the weights to `models/best.pt`, and exports them to
OpenVINO, which the bot then runs on the Arc by itself. The options:
* `--model yolo26n.pt`: smaller and faster
* `--epochs`: how long to train
* `--batch 8`: if the GPU runs out of memory
* `--workers 0`: if Windows complains about dataloader workers

### 3. Teach it today's game

The public datasets are older than the current game, and some biomes and tiers are missing from
them. Once a first model exists:

```powershell
.venv\Scripts\python tools\dataset.py capture --minutes 30   # play, a screenshot every 2 s
.venv\Scripts\python tools\dataset.py autolabel              # the model labels them
.venv\Scripts\python tools\dataset.py merge
.venv\Scripts\python tools\train.py --openvino
```

The labels `autolabel` writes are guesses. Correcting `datasets/captured` in a labeling tool
(Roboflow, CVAT, labelImg) before merging makes every round much better. Each round also makes the
next guesses better.

## First time: the interface

`start.bat` opens it. The **Ready to run** list on the Gather tab says what is still missing.

**Setup tab, once:**
1. **Gathering bar**: press *Calibrate* and start gathering a node in the game before the
   countdown ends. Then draw a box around the bar the game shows while gathering. *Test* prints
   the match score: it has to go over the threshold while gathering and stay under it otherwise.
2. **Character**: only if the character is not in the middle of the window. Click on its feet.
3. **Minimap**: draw a box inside the minimap (the map only). Only needed for routes.

**Routes tab:** walk to the start, type a name, press *Record by walking*, walk the route through
places full of resources, then press Stop. For a loop, end where you started. Keep the same
**minimap zoom** when the bot follows it. If the character walks the wrong way on a route, stand
somewhere open and press *Calibrate walking* on the Setup tab.

**Gather tab:** tick the resources, pick a route or none, press **Start**. With *Show what the
model sees* ticked, a window draws the detections, the nodes remembered as gathered and the target.
Keep that window on another screen, or the bot sees it too. While the bot works, the interface
minimizes itself so it never covers the game.

**Settings tab:**
* **Confidence**: lower finds more nodes and makes more mistakes
* **Gathering bar match**: the threshold *Test* compares against
* **Mount key**: pressed before walking on after each node
* **Skip the nodes a monster stands next to**
* **Hide the game interface while gathering** (alt+h): calibrate the bar the same way

Everything is saved in `data/`: settings, calibration, routes.

## Troubleshooting

| What happens | What to do |
|---|---|
| Clicks land next to the nodes | Windows display scaling is handled, so check that the game is windowed and not running as administrator |
| It walks to a node but never gathers | *Test* the bar while gathering. Calibrate again, or lower *Gathering bar match* |
| It leaves a node before it is empty | Lower *Gathering bar match*, the bar is being missed while gathering. *Test* shows the scores |
| It finds nothing | Show what the model sees. Lower *Confidence*. Teach the model the current biome (capture, autolabel) |
| It walks the wrong way on a route | *Calibrate walking*, and use the same minimap zoom as when recording |
| "Off the route" | Walk the character back near the route, it carries on by itself |

## What it does not do

* No banking: the bot stops when the bags are full.
* No fighting: it skips nodes guarded by a monster but does not defend itself. Hides need the
  animal killed by hand.
* No zone change: a route stays inside one map.

## Development

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/ruff check albion_bot tools tests
```

The tests run the bot against a small simulated Albion (`tests/fake_game.py`): a world larger than
the screen, a camera following the character, nodes with charges, a bar that blinks out between
charges, stumps the model still mistakes for trees, and a minimap. The layout:

```
albion_bot/
  platform/    finding and grabbing the game window, mouse and keyboard, per system
  vision/      the model, the gathering bar, the minimap, camera tracking
  navigation/  routes: recording, following, calibration
  bot/         the gathering loop and the memory of gathered nodes
  ui/          the interface, the pickers, the thread running the tasks
tools/         dataset.py (download, merge, capture, autolabel) and train.py
```

## License

Distributed under the MIT License.
Original project: [Michelprogram/magic-scanner](https://github.com/Michelprogram/magic-scanner).
