"""User settings + storage location. settings.json always lives in the fixed bootstrap folder so the
app can find it; everything else (AI model, scan cache, history) lives in the user-chosen storage folder."""
from __future__ import annotations
import json
import os
import shutil
import uuid
from pathlib import Path

BOOT_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "FaceFinder"
BOOT_DIR.mkdir(parents=True, exist_ok=True)
_FILE = BOOT_DIR / "settings.json"


def _load() -> dict:
    try:
        return json.loads(_FILE.read_text("utf-8"))
    except Exception:
        return {}


def _save(d: dict):
    _FILE.write_text(json.dumps(d, indent=2), "utf-8")


def storage_dir() -> Path:
    p = Path(_load().get("storage_dir") or BOOT_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:                       # e.g. external drive unplugged -> fall back safely
        return BOOT_DIR


def export_dir() -> Path:
    return Path(_load().get("export_dir") or storage_dir() / "Exports")


def set_export_dir(p: Path | None):
    d = _load()
    if p is None:
        d.pop("export_dir", None)         # follow the storage folder again
    else:
        d["export_dir"] = str(p)
    _save(d)


def model_root() -> Path:
    return storage_dir() / "models"


def cache_path(model: str) -> Path:
    return storage_dir() / f"cache_{model}.db"


def history_path() -> Path:
    return storage_dir() / "history.db"


def validate_writable(p: Path) -> str | None:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".ff_{uuid.uuid4().hex}"
        probe.write_text("x")
        probe.unlink()
        return None
    except OSError as e:
        return f"Can't write to {p}: {e.strerror or e}"


def _data_items(base: Path):
    items = [base / "models"] if (base / "models").exists() else []
    return items + list(base.glob("cache_*.db*")) + list(base.glob("history.db*"))


def cache_size() -> int:
    return sum(f.stat().st_size for f in storage_dir().glob("cache_*.db*"))


def clear_cache():
    for f in storage_dir().glob("cache_*.db*"):
        f.unlink(missing_ok=True)


def change_storage(new: Path, move: bool) -> list[str]:
    """Switch storage folder, optionally moving model/cache/history. Returns human-readable notes."""
    old, new = storage_dir().resolve(), Path(new).resolve()
    if old == new:
        return []
    err = validate_writable(new)
    if err:
        raise OSError(err)
    if old in new.parents and (new == old / "models" or (old / "models") in new.parents):
        raise OSError("Choose a folder outside the AI model folder.")
    notes = []
    if move:
        for src in _data_items(old):
            dst = new / src.name
            if dst.exists():
                notes.append(f"Kept the existing {src.name} already in the new folder.")
                continue
            shutil.move(str(src), str(dst))
    d = _load()
    d["storage_dir"] = str(new)
    _save(d)
    return notes
