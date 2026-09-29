@echo off
rem Opens the interface of the bot. Run scripts\setup_windows.ps1 once before.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Run first: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m albion_bot
if errorlevel 1 pause
