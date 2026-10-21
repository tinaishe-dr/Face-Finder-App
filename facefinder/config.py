"""Central tunables. Everything speed/accuracy related lives here."""
import os
from pathlib import Path

APP_NAME = "FaceFinder"
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / APP_NAME
MODEL_ROOT = DATA_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "buffalo_l"        # "buffalo_s" = smaller/faster, slightly less accurate
DET_SIZE = (800, 800)           # detector input; larger finds smaller faces but is slower
MAX_SIDE = 1600                 # images are downscaled to this before detection
MIN_FILE_BYTES = 20 * 1024      # skip icons / tiny assets
MIN_IMAGE_SIDE = 80
MIN_DET_SCORE = 0.5

LOADER_THREADS = 4              # parallel decode/resize workers
PREFETCH = 16                   # images decoded ahead of the AI model

SCORE_FLOOR = 0.30              # hits below this are never reported
DEFAULT_THRESHOLD = 0.45        # cosine similarity; higher = stricter
THUMB = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}

# Never descended into, in any scan type
JUNK_DIRS = {"node_modules", ".git", "__pycache__", "site-packages", ".venv", "venv",
             "$recycle.bin", "system volume information", "recovery"}
# Additionally skipped in Full Scan (OS / program files hold no personal photos)
SYSTEM_DIRS = {"windows", "program files", "program files (x86)", "programdata",
               "appdata", "windowsapps", "msocache"}
