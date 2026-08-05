# Retraining the model on RunPod

Written 2026-08-05, after measuring the dataset that produced `best_merged.pt`.

## Read this part first, because it decides whether the rest is worth doing

**More GPU does not fix the trees.** The tree class is bad for a countable reason and
compute is not it:

| class | train boxes | valid boxes |
|---|---|---|
| ore | 11,650 | 1,587 |
| stone | 6,588 | 1,143 |
| hide | 4,388 | 1,016 |
| fiber | 2,490 | 471 |
| **tree** | **560** | **97** |

Training the same 8,170 images again on a faster card gives the same model. What changes
the tree number is **more labelled trees from the zone the bot actually works in**, and
that is human work, not GPU work. RunPod turns a 10-hour local train into a 2-hour one
for about $2, which is worth having — but only after step 1.

There is a second reason the trees are weak, and it is the one that costs the most:
**every dataset so far labelled only gatherable trees, so decorative ones were never
marked as anything.** A forest is full of both and they look identical. The model finds
both and cannot be blamed for it. Measured live: of six trees it picked out, one was a
node and five were scenery.

## 1. Collect frames from the real zone

On the Windows machine, with the game running. **Not on the tutorial starter island** —
the current test character is standing in Forgotten Woods, which is T1–T2 saplings and
nothing like the zone the bot will be sold to work in.

```
.venv\Scripts\python.exe training\autolabel.py collect --minutes 20
```

Walk the farming circuit while it runs. It keeps a frame only when it differs from the
last kept one by more than `DIFFERENT_ENOUGH` (6.0), so standing still does not fill the
folder with a hundred copies of one tree.

Then pre-label them, so the job becomes correcting boxes rather than drawing them:

```
.venv\Scripts\python.exe training\autolabel.py label
```

This writes proper YOLO files (`0 0.537708 0.581682 ...`), a `classes.txt` and a
`data.yaml` next to the frames. `datasets/zone_test/` already holds 42 frames done this
way, as a worked example of the output.

## 2. Correct the boxes — this is the actual work

Open the folder in LabelImg, CVAT or Roboflow. **Keep the class order in `classes.txt`
exactly as it is** (`tree, stone, ore, fiber, hide, monster`) or the weights stop being
compatible.

What matters while correcting:

- **Box gatherable trees only.** A tree the game will not let you chop is background, and
  leaving it unboxed is what teaches the model the difference. This is the whole point of
  the exercise, so it is worth being strict rather than quick.
- **Box the trunk area generously**, not the canopy alone. The bot aims at
  `aim_depth` 0.68 down the box, and that number only means the trunk if the box covers
  the whole tree.
- Delete pre-labels that are wrong rather than nudging them into place.

**How many.** To take the tree class from 560 boxes to somewhere useful (~2,000) at a
realistic 3–5 gatherable trees per forest frame, that is **300 to 500 corrected frames**.
Budget a few hours. This is the part nobody can skip and no GPU shortens.

## 3. Prepare the upload

The dataset directory is 17GB but only **908MB of it is real**. The rest is yolov5's
`--cache disk` output, which is regenerated on whatever machine trains next:

```
train  jpg=801M  npy=14G
valid  jpg=90M   npy=2.3G
test   jpg=17M   npy=0
```

Merge the corrected zone frames into `datasets/albion_merged/train/`, then pack without
the cache:

```
tar czf albion_merged.tgz --exclude='*.npy' --exclude='*.cache' datasets/albion_merged
```

`data.yaml` currently holds absolute Windows paths and has to be rewritten for Linux.
Relative paths, so it works anywhere:

```yaml
train: train/images
val: valid/images
test: test/images
nc: 6
names: [tree, stone, ore, fiber, hide, monster]
```

## 4. Train on RunPod

Any CUDA card; an RTX 4090 or A40 is plenty and cheap. Pick a PyTorch template so torch
is already there.

```bash
# on the pod
git clone https://github.com/ultralytics/yolov5 && cd yolov5
pip install -r requirements.txt

tar xzf /workspace/albion_merged.tgz -C /workspace/

python train.py \
  --img 640 \
  --batch 32 \
  --epochs 150 \
  --patience 30 \
  --data /workspace/datasets/albion_merged/data.yaml \
  --weights /workspace/best_merged.pt \
  --device 0 \
  --cache disk \
  --name albion_zone
```

Two choices in there worth explaining:

- **`--weights best_merged.pt`, not `yolov5s.pt`.** Fine-tuning from the existing model
  keeps everything it already knows about ore, stone and hide while the new tree data
  lands. Starting from scratch throws away 78 epochs for no reason.
- **`--img 640`.** Do not raise it. The weights were trained at 640 and 960 measured far
  worse — 5 actionable frames of 42 against 14.

The local run used `--device xpu` through `training/train_local.py`, which exists only to
teach yolov5 about the Intel Arc card. On RunPod use `train.py` directly with
`--device 0`; none of that patching applies to CUDA.

Expect roughly 1–2 minutes an epoch on a 4090 at this dataset size, so 2–3 hours with
early stopping at `--patience 30`. Around $2.

## 5. Bring it back and prove it

```
runpodctl send yolov5/runs/train/albion_zone/weights/best.pt
```

Drop it next to `best_merged.pt` on the Windows machine and point `vision.weights` at it.

**Do not trust the mAP number.** The previous model scored 0.878 mAP50 overall and 0.841
on trees and still picked five scenery trees out of six. Measure it the way the bot
actually fails:

```
.venv\Scripts\python.exe -m albion.selftest
.venv\Scripts\python.exe -u -m albion --targets tree --read-labels --minutes 10 --debug
```

Then read `albion_bot.log` — not a grep of the console, the file — and count:

- how many `-> gatherable` against how many `-> scenery`, which is the real precision;
- charges per node, which says whether nodes are being emptied or abandoned;
- charges an hour on the final line, against the 520/hr best measured so far.

A retrain that raises mAP and does not raise the gatherable ratio has not helped.
