#!/usr/bin/env python
"""
Train the model on the Intel Arc GPU of this PC instead of Google Colab.

Colab hands out a session that dies after a few hours, which is less than one full
training run takes, so every attempt got cut off partway. This machine has an Arc A770
with 16GB and no time limit, and it turns out to be the faster of the two anyway: its
matrix cores do about 79 TFLOPS in half precision against roughly 65 on the Colab T4.

What stands in the way is that yolov5 was written when the only GPU worth having was an
Nvidia one, so it says "cuda" in the handful of places where it means "the GPU". Intel
cards are "xpu" to torch. Rather than editing the yolov5 checkout, which is a submodule
and would then differ from the copy the trained weights are meant to load into, this
script rewrites those few functions in memory and then calls yolov5's own training exactly
as Colab did. Nothing on disk changes, and there is no second version of yolov5 to keep
in step.

Usage, after training/download_datasets.py has built the merged dataset:

    .venv-train\\Scripts\\python.exe training\\train_local.py

Anything you pass is handed to yolov5, so its flags all still work:

    .venv-train\\Scripts\\python.exe training\\train_local.py --epochs 50 --batch 16
"""

import os
import sys
from pathlib import Path

# Ops Intel has not written a kernel for run on the CPU rather than stopping the run.
# Slower when it happens, and it beats losing an hour of training to one missing operator.
os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

# wandb interrupts training partway through to ask for an account, the same way it did
# on Colab. There is nothing to log to, so it is turned off before yolov5 looks for it.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import torch  # noqa: E402  imported after the environment is set, torch reads it at import

ROOT = Path(__file__).resolve().parents[1]
YOLOV5 = ROOT / "yolov5"
DATASET = ROOT / "datasets" / "albion_merged"

# Same settings the Colab notebook used, so the result is comparable to the runs already
# made there. --batch 32 fits in 16GB at 640 pixels with room to spare.
DEFAULTS = [
    "--img", "640",
    "--batch", "32",
    "--epochs", "150",
    "--patience", "30",
    "--weights", "yolov5s.pt",
    "--name", "albion_merged",
    "--exist-ok",
    "--cache", "disk",
    "--workers", "4",
    "--device", "xpu",
]


def patch_yolov5_for_xpu():
    """
    Teach the already imported yolov5 modules about Intel GPUs.

    Every replacement below falls back to the original behaviour when the device asked
    for is not an Intel one, so running this script with --device cpu or --device 0 still
    does what yolov5 would have done on its own.
    """
    import utils.torch_utils as torch_utils

    original_select_device = torch_utils.select_device
    original_autocast = torch_utils.smart_amp_autocast

    def select_device(device="", batch_size=0, newline=True):
        """Resolve --device xpu, and leave every other value to yolov5."""
        if not str(device).strip().lower().startswith("xpu"):
            return original_select_device(device, batch_size, newline)

        properties = torch.xpu.get_device_properties(0)
        print(
            f"YOLOv5 Python-{sys.version.split()[0]} torch-{torch.__version__} "
            f"XPU:0 ({properties.name}, {properties.total_memory / (1 << 20):.0f}MiB)"
        )

        return torch.device("xpu:0")

    def smart_amp_autocast(enabled, device="xpu"):
        """
        Mixed precision on the Intel GPU, in bfloat16 rather than the usual float16.

        Mixed precision is the single most important thing here: in full precision the
        A770 manages about 6 TFLOPS, and its matrix engines do 79 in either half format,
        so training without it would take a dozen times longer.

        Which half format is not a free choice on this card. float16 has so few exponent
        bits that small gradients vanish to zero, which is why training with it needs a
        loss scaler, and torch's scaler works out its scale in float64. Consumer Arc cards
        have no float64 at all, so that step stops the run outright, and Intel's emulation
        flag for it fails on this driver. bfloat16 keeps the full float32 exponent range
        and so needs no scaler and touches no float64, at identical speed. It carries
        fewer digits of precision than float16, which for training a detector is a
        non-issue and is what the large models are trained in nowadays anyway.
        """
        if str(device).lower().startswith("xpu"):
            return torch.amp.autocast("xpu", enabled=enabled, dtype=torch.bfloat16)

        return original_autocast(enabled, device=device)

    class XpuGradScaler(torch.amp.GradScaler):
        """
        The loss scaler yolov5 reaches for as torch.cuda.amp.GradScaler, held disabled.

        Nothing to scale in bfloat16, see above. A disabled scaler is not a stub: every
        one of its methods short-circuits, so scale(), unscale_(), step() and update()
        all still work and simply pass the loss and the optimizer straight through, and
        train.py needs no changes.
        """

        def __init__(self, *args, **kwargs):
            kwargs.pop("device", None)
            kwargs["enabled"] = False
            super().__init__("xpu", *args, **kwargs)

    # Modules import these names directly, so rebinding them on utils.torch_utils alone
    # would leave train.py and val.py holding the originals. Anything currently carrying
    # the original gets the replacement, which reaches every copy without naming them.
    for module in list(sys.modules.values()):
        if getattr(module, "select_device", None) is original_select_device:
            module.select_device = select_device

        if getattr(module, "smart_amp_autocast", None) is original_autocast:
            module.smart_amp_autocast = smart_amp_autocast

    torch.cuda.amp.GradScaler = XpuGradScaler

    # Housekeeping calls yolov5 makes on the Nvidia namespace. Left alone they are
    # harmless no-ops, but then the memory column of the progress bar reads 0G all run
    # and there is no way to see how close to the 16GB a batch size is.
    #
    # is_available has to answer yes for yolov5 to bother asking for the memory at all,
    # and once it does, torch's own "which device is current" helper starts routing to
    # the Nvidia namespace too and walks into an assert about not being built with CUDA.
    # current_device and device_count answer for it. A count of exactly one is the honest
    # answer and also keeps yolov5 off its multi-GPU paths, which are all guarded on
    # more than one.
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.empty_cache = lambda: torch.xpu.empty_cache()
    torch.cuda.memory_reserved = lambda *args, **kwargs: torch.xpu.memory_reserved(0)


def main():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise SystemExit(
            "No Intel GPU visible to torch. Install the GPU build with:\n"
            "  .venv-train\\Scripts\\python.exe -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/xpu"
        )

    arguments = sys.argv[1:]

    # The dataset is only required when the caller has not named one, so --data can still
    # point somewhere else for a quick test on a smaller set.
    if "--data" not in arguments:
        config = DATASET / "data.yaml"

        if not config.exists():
            raise SystemExit(
                f"No dataset at {config}\n"
                f"Build it first with:\n"
                f"  .venv-train\\Scripts\\python.exe training\\download_datasets.py"
            )

        arguments = ["--data", str(config)] + arguments

    # Only the defaults the caller left out, so every flag stays overridable.
    for index in range(0, len(DEFAULTS)):
        flag = DEFAULTS[index]

        if not flag.startswith("--") or flag in arguments:
            continue

        value = DEFAULTS[index + 1] if index + 1 < len(DEFAULTS) else None
        arguments += [flag] if value is None or value.startswith("--") else [flag, value]

    # yolov5 resolves weights, datasets and its runs folder relative to where it is run
    # from, which is how the Colab notebook drove it too.
    sys.path.insert(0, str(YOLOV5))
    os.chdir(YOLOV5)

    import train as yolov5_train

    patch_yolov5_for_xpu()

    print("training with:", " ".join(arguments), "\n")
    sys.argv = ["train.py"] + arguments

    yolov5_train.main(yolov5_train.parse_opt())


if __name__ == "__main__":
    main()
