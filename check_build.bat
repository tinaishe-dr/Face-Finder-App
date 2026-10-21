@echo off
setlocal
cd /d "%~dp0"
title FaceFinder - check build
if not exist "dist\FaceFinder\FaceFinder.exe" (
  echo No build found. Run build_exe.bat first.
  pause
  exit /b 1
)
del "%LOCALAPPDATA%\FaceFinder\selftest.log" 2>nul
echo Checking the built app. The first run downloads the AI model, about 300 MB, so please wait...
start /wait "" "dist\FaceFinder\FaceFinder.exe" --selftest
echo.
if exist "%LOCALAPPDATA%\FaceFinder\selftest.log" (
  type "%LOCALAPPDATA%\FaceFinder\selftest.log"
) else (
  echo No selftest.log was written, so the app failed to start.
)
echo.
pause
