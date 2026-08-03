@echo off
title Albion training
cd /d C:\Users\otopk\albion-gathering-bot

rem No chcp here on purpose. Everything below is redirected into a file rather than
rem drawn on this console, so the codepage buys nothing, and the one crashed launch
rem of this script so far was the one that set it. watch_training.py picks its own
rem encoding when it reads the log back, which is where the box characters matter.

echo Resuming training from the last saved epoch.
echo.
echo This window has to stay open until it finishes. Everything it prints goes
echo into training_run.log instead of here, so that watch_training.bat can show
echo it as a readable progress bar. Double-click that to see how it is going.
echo.
echo Closing this window stops the training. Restarting it picks up again from
echo the last finished epoch, nothing is lost.
echo.

rem Torch on Windows has crashed outright once here, taking the run with it, and a run
rem this long is left alone overnight. Since every epoch is saved to last.pt and --resume
rem carries on from it, a crash costs one epoch rather than the night, but only if
rem something restarts it. That is this loop. The attempt cap is there so that a fault
rem which fails immediately and every time cannot spin restarting until morning.
set /a attempt=0

:run
set /a attempt+=1
echo. >> training_run.log
echo ==== attempt %attempt% ==== >> training_run.log

rem -u is what makes the log worth watching. The progress bar redraws with carriage
rem returns and never writes a newline, so with the normal buffering it sits in an 8KB
rem buffer and the log appears frozen for minutes at a time. Unbuffered, every redraw
rem lands in the file as it happens.
.venv-train\Scripts\python.exe -u training\train_local.py --resume >> training_run.log 2>&1

if not errorlevel 1 goto finished

if %attempt% geq 8 goto giveup

echo ==== attempt %attempt% ended badly, restarting from the last saved epoch ==== >> training_run.log
timeout /t 30 /nobreak >nul
goto run

:giveup
echo.
echo Stopped after %attempt% failed attempts. The end of training_run.log says why.
goto done

:finished
echo.
echo Training finished. The model is in
echo   yolov5\runs\train\albion_merged\weights\best.pt

:done
echo.
echo Press a key to close.
pause >nul
