# FaceFinder

**Find photos of the same person across your PC — locally.**

## Overview

FaceFinder is a Python desktop app for Windows. Choose a reference photo, select where to scan, and review photos containing a similar face. Face detection and matching run on your machine; your photos and face data are never uploaded.

## Features

- **Flexible scans:** Quick searches common user folders, Full searches fixed internal drives while skipping system folders, and Custom searches folders you choose.
- **Reference face selection:** Pick the person in a group photo, or add several reference photos of the same person.
- **Adjustable matching:** Change strictness without rescanning, and review lower-scoring candidates with the All candidates filter.
- **Live results:** Browse matches as they arrive, open photos in the built-in viewer, and pause, continue, or stop a scan.
- **Faster repeat scans:** A local cache skips unchanged images; scan history lets you reopen previous results.
- **Copy export:** Copy visible results to an export folder without changing the originals.

## How It Works

1. **Choose a face.** FaceFinder detects faces in your reference photo and lets you select the person to find.
2. **Choose a scan.** Select Quick, Full, or Custom and start searching.
3. **Detect and compare.** Images are decoded, resized, and checked for faces. Each face becomes a 512-dimensional embedding, compared with the reference using cosine similarity.
4. **Review and export.** Adjust strictness, inspect candidates, and copy the results you want to keep.

The PySide6 interface in `main.py` runs scanning through a `QThread`, receiving progress and results through signals. The Qt-independent `ScanJob` in `facefinder/pipeline.py` coordinates file discovery, cache lookup, image decoding, inference, and matching. SQLite caches embeddings and images with no detected faces using each file’s path, size, and modification time.

## Tech Stack

- **Application:** Python and PySide6
- **Face detection and recognition:** InsightFace `buffalo_l`, using SCRFD and ArcFace
- **Inference:** ONNX Runtime on CPU, with optional CUDA or DirectML acceleration when configured and available
- **Matching:** NumPy cosine similarity on normalized embeddings
- **Storage:** SQLite with write-ahead logging (WAL)
- **Windows packaging:** PyInstaller

## Getting Started

Use **Python 3.11** on Windows. From the project folder, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The first run downloads the AI model, roughly **300 MB**, to `%LOCALAPPDATA%\FaceFinder\models` by default. Once the model is available locally, scanning works offline. The storage location can be changed in Settings.

If InsightFace fails to build during installation, install **Microsoft C++ Build Tools** and retry with Python 3.11.

## Building Windows EXE

Double-click `build_exe.bat` in the project folder. It finds Python, prepares the virtual environment, installs dependencies and PyInstaller, and builds:

```text
dist\FaceFinder\FaceFinder.exe
```

Copy the **entire `dist\FaceFinder` folder** when moving or sharing the app. Run `check_build.bat` to verify that the packaged app can load the AI model on the target PC.

For startup diagnostics, run:

```powershell
.\dist\FaceFinder\FaceFinder.exe --selftest
```

Logs are saved to `%LOCALAPPDATA%\FaceFinder\selftest.log` and `%LOCALAPPDATA%\FaceFinder\crash.log`. To supply models offline, copy the existing `models` folder from the app’s storage folder next to the executable.

## Privacy

Photos, face embeddings, the scan cache, and history stay on your machine. No cloud service is used for face detection or matching. Initial dependency installation and the first model download require internet access; subsequent scans can run offline.

Copy export leaves source photos unchanged. Cached face data and history remain in the configured local storage folder.

## Limitations

- Small faces — roughly below 40 pixels after downscaling — may be missed.
- Masks, heavy sunglasses, side profiles, and large age differences can reduce matching scores.
- Identical twins may match. Results are candidates for review, not proof of identity.
- Scan speed depends on image count, storage speed, and available hardware. GPU acceleration requires a compatible runtime and setup.
- Video scanning, automatic watch-folder updates, and automatic grouping of everyone into people albums are not implemented.
