@echo off
REM Run inside the activated venv. Produces dist\FaceFinder\FaceFinder.exe
pip install pyinstaller
pyinstaller --noconfirm --windowed --name FaceFinder ^
  --collect-all insightface --collect-all onnxruntime ^
  main.py
