@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title LocalTranscribe - hardware check
set PYTHONUTF8=1

rem Checks this PC (memory, graphics card, disk space), recommends the speech model that fits it and saves your choice
rem in localtranscribe_settings.json. It only looks and saves; it installs and downloads nothing.
rem
rem   check_hardware.bat                 look, recommend, ask (press Enter to accept)
rem   check_hardware.bat best            choose without asking:  best ^| light ^| both ^| cpu
rem   check_hardware.bat --yes           accept the recommendation without asking
rem   check_hardware.bat --json          machine-readable result, nothing is saved ^(add --yes to save^)
rem   check_hardware.bat --no-save       look only
rem
rem Works before setup_windows.bat has run ^(it then needs uv or a Python on the PATH^), and again at any time afterwards.

rem ---- any Python does: the check needs only the standard library ---------------------------
if exist ".venv\Scripts\python.exe" goto use_venv
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
set "BASEPY="
where uv >nul 2>nul
if errorlevel 1 goto try_launcher
for /f "delims=" %%p in ('uv python find 3.11 2^>nul') do set "BASEPY=%%p"
if defined BASEPY goto use_uv
:try_launcher
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 goto use_launcher
python -c "import sys" >nul 2>nul
if not errorlevel 1 goto use_python
echo.
echo Python was not found. Run  setup_windows.bat  first: it installs everything, including this check.
pause
exit /b 1

:use_venv
".venv\Scripts\python.exe" -m localtranscribe.precheck %*
goto checked
:use_uv
"%BASEPY%" -m localtranscribe.precheck %*
goto checked
:use_launcher
py -3 -m localtranscribe.precheck %*
goto checked
:use_python
python -m localtranscribe.precheck %*

:checked
set "STATUS=%errorlevel%"
set "QUIET="
for %%a in (%*) do if /i "%%~a"=="--json" set "QUIET=1"
if defined QUIET exit /b %STATUS%
if not "%STATUS%"=="0" goto finish
echo.
if not exist ".venv\Scripts\python.exe" echo Next: run  setup_windows.bat  to install the program and download the model.
if exist ".venv\Scripts\python.exe" echo If you changed the choice, run  setup_windows.bat  again to install and download what it needs.
:finish
rem double-click: keep the window open
if "%~1"=="" pause
exit /b %STATUS%
