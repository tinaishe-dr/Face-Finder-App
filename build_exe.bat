@echo off
setlocal
cd /d "%~dp0"
title FaceFinder - build
echo ==============================================
echo  FaceFinder - building the Windows app
echo  Working in: %CD%
echo ==============================================
echo.

REM ---- 1. find Python (3.10 to 3.12 work best) ----
set "PY="
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V --version >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python --version >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython
echo Using Python: %PY%

REM ---- 2. private virtual environment next to this script ----
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
set "VPY=%CD%\.venv\Scripts\python.exe"

REM ---- 3. install everything the app and the build need ----
echo.
echo Installing requirements - the first time takes several minutes...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"%VPY%" -m pip install pyinstaller
if errorlevel 1 goto :fail

REM ---- 4. build ----
echo.
echo Building the app - this takes several minutes...
"%VPY%" -m PyInstaller --noconfirm --clean --windowed --name FaceFinder ^
  --icon "assets\icon.ico" --add-data "assets;assets" ^
  --collect-all insightface --collect-all onnxruntime ^
  --exclude-module matplotlib --exclude-module pandas --exclude-module sklearn ^
  --exclude-module tkinter --exclude-module imageio_ffmpeg --exclude-module IPython ^
  main.py
if errorlevel 1 goto :fail
if not exist "dist\FaceFinder\FaceFinder.exe" goto :fail

echo.
echo ==============================================
echo  DONE - your app is here:
echo    %CD%\dist\FaceFinder\FaceFinder.exe
echo  To move or share it, copy the WHOLE FaceFinder folder, not just the exe.
echo  Run check_build.bat to verify the build works on this PC.
echo ==============================================
explorer "dist\FaceFinder"
pause
exit /b 0

:nopython
echo.
echo ERROR: Python was not found.
echo Install Python 3.11 from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this file again.
goto :fail

:fail
echo.
echo ==============================================
echo  BUILD FAILED - read the messages above this box.
echo  Common causes: Python 3.13 or newer, so use 3.10 to 3.12; no internet;
echo  pip trying to compile insightface, which needs Microsoft C++ Build Tools.
echo ==============================================
pause
exit /b 1
