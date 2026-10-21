"""File discovery: Quick / Full / Custom scan roots and a filtered recursive walk."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from . import config


def quick_roots() -> list[str]:
    home = Path.home()
    names = ["Pictures", "Desktop", "Downloads", "Documents", "Videos", "OneDrive"]
    return normalise_roots([str(home / n) for n in names])


def full_roots() -> list[str]:
    if sys.platform != "win32":
        return normalise_roots([str(Path.home())])
    import ctypes
    k32 = ctypes.windll.kernel32
    mask, roots = k32.GetLogicalDrives(), []
    for i in range(26):
        if mask & (1 << i):
            root = f"{chr(65 + i)}:\\"
            if k32.GetDriveTypeW(root) == 3:      # DRIVE_FIXED only (no USB / network / CD)
                roots.append(root)
    return normalise_roots(roots)


def normalise_roots(roots) -> list[str]:
    """Drop missing folders and folders already covered by a parent (no double scanning)."""
    resolved = sorted({os.path.abspath(r) for r in roots if os.path.isdir(r)}, key=len)
    kept: list[str] = []
    for r in resolved:
        rl = os.path.normcase(r)
        if not any(rl == os.path.normcase(k) or rl.startswith(os.path.normcase(k).rstrip(os.sep) + os.sep)
                   for k in kept):
            kept.append(r)
    return kept


def _is_link(path: str) -> bool:
    if os.path.islink(path):
        return True
    isj = getattr(os.path, "isjunction", None)
    return bool(isj and isj(path))


def iter_images(roots, skip_system: bool = False):
    skip = set(config.JUNK_DIRS) | (config.SYSTEM_DIRS if skip_system else set())
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in skip and not _is_link(os.path.join(dirpath, d))]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in config.IMAGE_EXTS:
                    yield os.path.join(dirpath, fn)
