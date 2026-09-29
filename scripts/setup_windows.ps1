# Sets the bot up on Windows, with the Intel Arc build of torch.
# Run it from the repository folder:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Add -Cpu on a machine without an Intel Arc GPU.
#
# It is for albion_bot/ and installs into .venv. Where albion/ already runs from .venv,
# it would replace its torch build, so do not run it there, see CLAUDE.md.

param([switch]$Cpu)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) {
    # torch has Windows builds for these, the newest one installed is used.
    $version = $null
    foreach ($candidate in "3.13", "3.12") {
        py "-$candidate" --version *> $null
        if ($LASTEXITCODE -eq 0) { $version = $candidate; break }
    }
    if (-not $version) { throw "Install Python 3.13 from python.org, with 'tcl/tk' and the 'py launcher' ticked" }

    Write-Host "Creating .venv with Python $version"
    py "-$version" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}

$python = ".venv\Scripts\python.exe"

function Run([string[]]$arguments) {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "failed: python $($arguments -join ' ')" }
}

Run -m, pip, install, --upgrade, pip

$index = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/xpu" }
Run -m, pip, install, torch, torchvision, --index-url, $index
Run -m, pip, install, -r, requirements.txt
Run -m, pip, install, -r, requirements-dev.txt

Run -c, "import torch; print('torch', torch.__version__, '| Intel GPU (xpu):', hasattr(torch, 'xpu') and torch.xpu.is_available())"
Run -c, "import openvino as ov; print('OpenVINO devices:', ov.Core().available_devices)"
Run -m, pytest, -q, tests

Write-Host "`nDone. Start the bot with start.bat"
