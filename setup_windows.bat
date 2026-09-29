@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LocalTranscribe - setup
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

rem Usage:  setup_windows.bat [auto^|best^|light^|both] [cpu] [--yes]
rem   auto  (default) checks this PC, recommends a model and asks you to accept it or choose another
rem   best  Qwen3-ASR-1.7B, the most accurate (needs about 6 GB of GPU memory)     light  Qwen3-ASR-0.6B (about 3 GB)
rem   both  downloads both       cpu  force the CPU-only install even if an NVIDIA GPU exists
rem   --yes accept the recommendation without asking
rem An explicit best/light/both/cpu skips the question, but the hardware check is still shown and warns if it will not fit.
rem The choice is saved in localtranscribe_settings.json; check_hardware.bat repeats the check at any time.

echo.
echo ============================================================
echo   LocalTranscribe - setup for Windows
echo ============================================================
echo.
echo This will:
echo   1. install "uv" (a small tool that fetches Python and the packages) if it is missing
echo   2. check this PC (memory, graphics card, disk space) and recommend the model that fits it
echo   3. create a private Python environment in the folder ".venv" next to this file
echo   4. install the packages (about 1-5 GB: the NVIDIA build of PyTorch is the big one)
echo   5. download the speech models once, so that later runs work without internet
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

rem ---- hardware check: standard library only, so it runs with the bare uv Python before any big download ----
set "BASEPY="
for /f "delims=" %%p in ('uv python find 3.11 2^>nul') do set "BASEPY=%%p"
if defined BASEPY goto have_basepy
echo Fetching Python 3.11 ^(about 30 MB^) for the hardware check...
uv python install 3.11
for /f "delims=" %%p in ('uv python find 3.11 2^>nul') do set "BASEPY=%%p"
if not defined BASEPY goto python_failed
:have_basepy
echo.
"%BASEPY%" -m localtranscribe.precheck %*
if errorlevel 1 goto precheck_failed
set "REQ="
set "MODELS="
for /f "delims=" %%v in ('"%BASEPY%" -m localtranscribe.precheck --print requirements') do set "REQ=%%v"
for /f "delims=" %%v in ('"%BASEPY%" -m localtranscribe.precheck --print models') do set "MODELS=%%v"
if not defined REQ goto precheck_failed
if not defined MODELS set "MODELS=auto"
set "KIND=NVIDIA GPU"
if /i not "%REQ%"=="requirements-windows.txt" set "KIND=CPU only"
echo.
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
".venv\Scripts\python.exe" -m localtranscribe.setup_models --model %MODELS%
if errorlevel 1 goto models_failed

echo.
echo ============================================================
echo   Setup complete.
echo   Drag audio or video files onto  transcribe.bat  to transcribe them.
echo   To change the model later, run  check_hardware.bat  and then this setup again.
echo ============================================================
pause
exit /b 0

:uv_failed
echo.
echo Could not install uv automatically. Install it by hand from https://docs.astral.sh/uv/
echo ^(for example:  winget install --id=astral-sh.uv -e^) and run this setup again.
pause
exit /b 1

:python_failed
echo.
echo Could not find or fetch Python 3.11. Check your internet connection, or install Python 3.11
echo from https://www.python.org/downloads/ first, and run this setup again.
pause
exit /b 1

:precheck_failed
echo.
echo Setup stopped: see the message above. No packages or models have been downloaded.
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
