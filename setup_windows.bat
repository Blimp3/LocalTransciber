@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LocalTranscribe - setup
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

rem Usage:  setup_windows.bat [auto^|best^|light^|both] [cpu]
rem   auto  (default) downloads the best model your GPU can hold, or the light one on small GPUs / CPU
rem   best  Qwen3-ASR-1.7B (needs about 7 GB of GPU memory)      light  Qwen3-ASR-0.6B (about 3 GB)
rem   both  downloads both       cpu  force the CPU-only install even if an NVIDIA GPU exists
set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=auto"
set "FORCE_CPU=%~2"

echo.
echo ============================================================
echo   LocalTranscribe - setup for Windows
echo ============================================================
echo.
echo This will:
echo   1. install "uv" (a small tool that fetches Python and the packages) if it is missing
echo   2. create a private Python environment in the folder ".venv" next to this file
echo   3. install the packages (about 3-5 GB with the NVIDIA build of PyTorch)
echo   4. download the speech models once, so that later runs work without internet
echo Nothing outside this folder, your user profile and the Hugging Face model cache is touched.
echo.

rem ---- uv -------------------------------------------------------------------------------
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo "uv" is not installed. Installing it now for your user account with the official
echo installer from https://astral.sh/uv  ^(it downloads uv.exe and adds it to your PATH^).
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto uv_failed
:have_uv
for /f "delims=" %%v in ('uv --version') do echo Using %%v

rem ---- which package set ---------------------------------------------------------------
set "REQ=requirements-windows.txt"
set "KIND=NVIDIA GPU"
nvidia-smi -L >nul 2>nul
if errorlevel 1 set "REQ=requirements-cpu.txt"
if errorlevel 1 set "KIND=CPU only - no NVIDIA GPU found"
if /i "%FORCE_CPU%"=="cpu" set "REQ=requirements-cpu.txt"
if /i "%FORCE_CPU%"=="cpu" set "KIND=CPU only - forced"
echo Install type: %KIND%  -^>  %REQ%
echo.

rem ---- environment ----------------------------------------------------------------------
if exist ".venv\Scripts\python.exe" goto have_venv
echo Creating the Python 3.11 environment...
uv venv --python 3.11 ".venv"
if errorlevel 1 goto venv_failed
:have_venv

echo Installing packages ^(this takes a few minutes the first time^)...
uv pip install --python ".venv\Scripts\python.exe" --index-strategy unsafe-best-match -r "%REQ%"
if errorlevel 1 goto pip_failed

rem ---- models ---------------------------------------------------------------------------
echo.
echo Downloading models...
".venv\Scripts\python.exe" -m localtranscribe.setup_models --model %PRESET%
if errorlevel 1 goto models_failed

echo.
echo ============================================================
echo   Setup complete.
echo   Drag audio or video files onto  transcribe.bat  to transcribe them.
echo ============================================================
pause
exit /b 0

:uv_failed
echo.
echo Could not install uv automatically. Install it by hand from https://docs.astral.sh/uv/
echo ^(for example:  winget install --id=astral-sh.uv -e^) and run this setup again.
pause
exit /b 1

:venv_failed
echo.
echo Could not create the Python environment. Check your internet connection and run this setup again.
pause
exit /b 1

:pip_failed
echo.
echo Package installation failed. Check your internet connection and run this setup again.
pause
exit /b 1

:models_failed
echo.
echo Some models could not be downloaded. Check your internet connection and run this setup again.
pause
exit /b 1
