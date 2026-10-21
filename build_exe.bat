@echo off
REM Run inside the activated venv. Produces dist\FaceFinder\FaceFinder.exe (with the app icon)
pip install pyinstaller
pyinstaller --noconfirm --windowed --name FaceFinder ^
  --icon assets\icon.ico --add-data "assets;assets" ^
  --collect-all insightface --collect-all onnxruntime ^
  main.py
