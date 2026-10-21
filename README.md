# FaceFinder – find every photo of one person on your PC

Upload a reference photo → choose Quick / Full / Custom scan → get all photos containing that face.
**100% local**: no photos or face data ever leave the machine.

## Architecture

```
 ┌────────────── PySide6 GUI (main.py) ──────────────┐
 │ face picker · scan type · strictness · history ·  │
 │ settings (storage/export folders)                 │
 │ live results grid · export (copy)                 │
 └───────┬───────────────────────────────▲───────────┘
   QThread│ signals (progress, hit)       │
 ┌───────▼───────────────────────────────┴───────────┐
 │ ScanJob  (facefinder/pipeline.py, Qt-free)        │
 │                                                    │
 │  scanner ──► cache lookup ──► decode pool ──► AI   │
 │  (walk,      (path+size+      (4 threads,    embed │
 │  filters)     mtime, SQLite)   resize/EXIF)  512-d │
 │                  ▲                              │  │
 │                  └────── store embeddings ◄─────┘  │
 │                                  cosine match vs reference(s) ──► Hit
 └────────────────────────────────────────────────────┘
 AI: InsightFace  SCRFD detector + ArcFace (buffalo_l) via ONNX Runtime
     (CPU, or CUDA / DirectML GPU if available)
```

| Layer | Choice | Why |
|---|---|---|
| Detection | SCRFD (InsightFace) | accurate on small/side faces, fast on CPU |
| Recognition | ArcFace ResNet-50, 512-d embedding | state of the art open model; same person ⇒ high cosine similarity |
| Runtime | ONNX Runtime | no PyTorch/TensorFlow (smaller exe); GPU via DirectML on any Windows GPU |
| Matching | NumPy dot product on unit vectors | exact and instant up to ~1M faces |
| Cache | SQLite (WAL) | re-scans skip unchanged files; "no face" results cached too |
| GUI | PySide6 | native-feeling, threads + signals, packages cleanly |
| Packaging | PyInstaller | `build_exe.bat` |

### Reference face picker
Every reference photo is scanned for *all* faces. One face is used automatically; if there are several, a dialog shows the
photo with numbered boxes – click the person you want (largest face preselected). Add several photos of the same person
for better accuracy; remove or clear individual reference faces any time.

### Matching vs. all candidates
Every face scoring at least `SCORE_FLOOR` (0.30) is kept as a *candidate*; the strictness slider only decides which ones
count as *matches*. The **Show** filter above the results switches between **Matching strictness only** and **All
candidates**. In the all view, results under your strictness stay visible with a grey “· below” label, so correct-but-lower
scoring photos (side profiles, old photos, sunglasses) can be reviewed and are never lost. Moving the slider relabels
instantly without rescanning, and saved History scans reopen with all their candidates. Results are sorted best-first when a
scan ends. "Export visible" copies whatever is currently shown.

### Image viewer
Double-click a result (or select it and press **View**) to open it in the built-in viewer. Move through the results with
**← / →** (or the Previous / Next buttons; **Home / End** jump to the first / last; **Esc** closes). The viewer walks exactly
the results currently shown, in grid order, so it respects the Show filter and strictness. It shows the position ("3 / 12"),
the score, the full path and a "below strictness" flag for lower-scoring candidates, and handles files that were deleted since
the scan. "Open in default app" and "Show in folder" are one click away, and when you close it the grid is left on the last
picture you viewed. Pictures are fitted to the window (downscaled to 1600 px for speed; the default app shows full resolution).

### App icon
`assets/icon.ico` (16–256 px, bolder strokes at small sizes so it stays readable in the taskbar) and `assets/icon.png`.
They are applied to every window and dialog, to the Windows taskbar (via its own app identity, so it doesn't show the Python
icon while running from source) and to the built `.exe`. To restyle it, edit the colours/shapes in `tools/make_icon.py` and run
`python tools/make_icon.py`.

### Pause / Continue / Stop
**Pause** freezes the scan at the next image boundary (normally within a second) and flushes progress to disk; the app stays
fully usable – adjust strictness, browse and export the matches found so far. **Continue** resumes exactly where it paused.
**Stop** ends the scan for good (works while running or paused), keeps every match found so far and saves it to History.
Stopped scans are cheap to finish later: re-running them skips every image already analysed (scan cache), so only the
unscanned remainder costs time.

### Live timer & scan counter
A stats bar under the progress bar updates four times a second: **elapsed time** (HH:MM:SS), **images scanned**, **speed**
(images/s) and **candidate matches**. The clock starts once the AI model is ready and **excludes paused time**, so it shows
real scanning time. When a scan ends the final time appears in the status line ("Done – 12,430 images scanned in 18m 42s")
and is stored in History, where each entry shows its duration. Older history files are upgraded automatically.

### History
Each finished or cancelled scan is saved (reference faces, scan type, folders, strictness and every candidate match; newest
50 kept). **History** lets you re-open a past scan's results instantly (no rescanning; deleted files are skipped and counted)
or **re-run** it: the reference faces, scan type and folders are restored, then you press Start.

### Storage & settings
**Settings** lets you choose the *storage folder* (AI model, scan cache, history) – e.g. a bigger D: drive – optionally moving
existing data there – and the *export folder* where "Export matches" opens. It also shows cache size and clears it. Only
`settings.json` stays in `%LOCALAPPDATA%\FaceFinder` so the app can always find your chosen folder; if that folder is
unavailable (drive unplugged) the app falls back to the default.

### Scan types
- **Quick** – Pictures, Desktop, Downloads, Documents, Videos, OneDrive.
- **Full** – every fixed internal drive; skips Windows, Program Files, ProgramData, AppData, Recycle Bin, junctions/symlinks.
- **Custom** – folders you pick. Nested folders are de-duplicated.

### Scaling path (not built, deliberately)
- \>1M faces: replace the NumPy dot with `hnswlib` / FAISS (same embeddings, same cache).
- "Group everyone" / People view: run DBSCAN/HDBSCAN over the cached embeddings – no rescan needed.
- Throughput: process pool for decode, `onnxruntime-directml`/`-gpu`, or `buffalo_s` for ~2× speed.
- Video frames, watch-folder auto-updates, per-person albums: all consume the same `Hit` stream.

## Run
```
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
First run downloads the model (roughly 300 MB) into `%LOCALAPPDATA%\FaceFinder\models`; afterwards it works offline.
If `pip install insightface` fails to build, install "Microsoft C++ Build Tools" (or use Python 3.10/3.11).

## Build the .exe
`build_exe.bat` → `dist\FaceFinder\FaceFinder.exe` (uses `assets\icon.ico` as the exe, taskbar and window icon). For an offline installer, copy the `models` folder next to it.

## Tuning
`facefinder/config.py`: `SCORE_FLOOR` (lowest score kept as a candidate; lower it for an even wider net), `DEFAULT_THRESHOLD` (0.45; ~0.40 lenient, ~0.55 strict), `DET_SIZE` (bigger finds smaller faces),
`MODEL_NAME`, `LOADER_THREADS`.

## Known limits
Faces smaller than ~40 px after downscaling can be missed; heavy sunglasses/masks and childhood-vs-adult photos lower
scores; identical twins can match. Results are candidates to review, not proof of identity. Export copies files and never
modifies originals.
