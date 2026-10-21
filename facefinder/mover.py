"""Move matched pictures out of their folders (NOT a copy) and put them back if needed (Qt-free).

Every move appends a line to FaceFinder-moved.jsonl inside the destination folder ({"src": original path,
"name": file name inside the folder}), so the folder carries its own restore list.
"""
from __future__ import annotations
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "FaceFinder-moved.jsonl"


@dataclass
class MoveResult:
    moved: list = field(default_factory=list)      # [(src, dst)]
    failed: list = field(default_factory=list)     # [(src, error text)]
    cancelled: bool = False


@dataclass
class RestoreResult:
    restored: int = 0
    missing: int = 0         # moved file no longer in the folder (you deleted it) – nothing to restore
    occupied: int = 0        # something else now sits at the original path – left in the folder
    failed: list = field(default_factory=list)
    cancelled: bool = False


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def is_within(child: str, parent: str) -> bool:
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent)).rstrip(os.sep)
    return c == p or c.startswith(p + os.sep)


def _unique(folder: Path, name: str) -> Path:
    dst, n, stem, suf = folder / name, 1, Path(name).stem, Path(name).suffix
    while dst.exists():
        dst = folder / f"{stem} ({n}){suf}"
        n += 1
    return dst


def unique_folder(path: Path) -> Path:
    """A folder path that doesn't exist yet, so every move gets its own restore list."""
    path, n = Path(path), 2
    cand = path
    while cand.exists():
        cand = path.with_name(f"{path.name} ({n})")
        n += 1
    return cand


def move_files(paths, dest: Path, progress=None, stop: threading.Event | None = None) -> MoveResult:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    res, total = MoveResult(), len(paths)
    with open(dest / MANIFEST, "a", encoding="utf-8") as log:
        for i, src in enumerate(paths, 1):
            if stop is not None and stop.is_set():
                res.cancelled = True
                break
            dst = _unique(dest, os.path.basename(src))
            try:
                shutil.move(src, str(dst))                       # rename, or copy+delete across drives
                log.write(json.dumps({"src": src, "name": dst.name}) + "\n")
                log.flush()
                res.moved.append((src, str(dst)))
            except Exception as e:
                if os.path.exists(src) and dst.exists():         # cross-drive copy made but source couldn't be removed
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                res.failed.append((src, getattr(e, "strerror", None) or str(e)))
            if progress:
                progress(i, total)
    if not res.moved and (dest / MANIFEST).exists() and (dest / MANIFEST).stat().st_size == 0:
        (dest / MANIFEST).unlink()
        try:
            dest.rmdir()                                         # nothing moved: don't leave an empty folder behind
        except OSError:
            pass
    return res


def find_manifests(folder: Path) -> list[Path]:
    folder = Path(folder)
    if (folder / MANIFEST).exists():
        return [folder]
    return sorted(p for p in folder.iterdir() if p.is_dir() and (p / MANIFEST).exists()) if folder.is_dir() else []


def _records(folder: Path) -> list[dict]:
    out = []
    for line in (Path(folder) / MANIFEST).read_text("utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def count_records(folder: Path) -> int:
    return len(_records(folder))


def restore_all(folders, progress=None, stop: threading.Event | None = None) -> RestoreResult:
    res, recs = RestoreResult(), [(Path(f), r) for f in folders for r in _records(f)]
    keep: dict[Path, list[dict]] = {Path(f): [] for f in folders}
    for i, (folder, r) in enumerate(recs, 1):
        if stop is not None and stop.is_set():
            res.cancelled = True
            keep[folder].extend(x for f2, x in recs[i - 1:] if f2 == folder)
            break
        cur, src = folder / r["name"], r["src"]
        if not cur.exists():
            res.missing += 1
        elif os.path.exists(src):
            res.occupied += 1
            keep[folder].append(r)
        else:
            try:
                os.makedirs(os.path.dirname(src), exist_ok=True)
                shutil.move(str(cur), src)
                res.restored += 1
            except Exception as e:
                res.failed.append((src, getattr(e, "strerror", None) or str(e)))
                keep[folder].append(r)
        if progress:
            progress(i, len(recs))
    for folder, left in keep.items():                            # keep only what is still unresolved
        m = folder / MANIFEST
        if left:
            m.write_text("".join(json.dumps(x) + "\n" for x in left), "utf-8")
        else:
            m.unlink(missing_ok=True)
            try:
                folder.rmdir()                                   # tidy up if the folder is now empty
            except OSError:
                pass
    return res
