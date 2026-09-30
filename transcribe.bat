@echo off
rem Drag audio/video files onto this file, or run:  transcribe.bat file.m4a [--context "names, terms"] [--speakers 2]
setlocal
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PYTHONPATH=%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo LocalTranscribe is not set up yet. Run setup_windows.bat first.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Drag one or more audio/video files onto transcribe.bat to transcribe them to text.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m localtranscribe %*
pause
