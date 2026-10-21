"""AI layer: face detection + 512-d ArcFace embeddings via InsightFace (ONNX Runtime)."""
from __future__ import annotations
import threading
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageOps
from . import config

try:  # optional HEIC support
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


@dataclass
class Face:
    bbox: tuple          # x1, y1, x2, y2 in the (downscaled) image
    emb: np.ndarray      # L2-normalised float32[512]


def load_image(path: str) -> Image.Image | None:
    """Decode, fix EXIF rotation, downscale. Returns RGB PIL image or None if unreadable."""
    try:
        with Image.open(path) as im:
            if im.format == "JPEG":
                im.draft("RGB", (config.MAX_SIDE, config.MAX_SIDE))  # fast partial decode
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((config.MAX_SIDE, config.MAX_SIDE))
        return im
    except Exception:
        return None


class FaceEngine:
    def __init__(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        wanted = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        config.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(name=config.MODEL_NAME, root=str(config.MODEL_ROOT),
                                allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0, det_size=config.DET_SIZE)

    def embed(self, img: Image.Image) -> list[Face]:
        bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        out = []
        for f in self.app.get(bgr):
            if float(f.det_score) >= config.MIN_DET_SCORE:
                out.append(Face(tuple(float(v) for v in f.bbox),
                                np.asarray(f.normed_embedding, dtype=np.float32)))
        return out


_engine: FaceEngine | None = None
_lock = threading.Lock()


def get_engine() -> FaceEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = FaceEngine()
        return _engine
