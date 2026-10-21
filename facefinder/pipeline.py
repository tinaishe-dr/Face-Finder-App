"""Qt-free scan pipeline:  walk -> cache lookup -> parallel decode -> AI embed -> match.

Decoding runs on a thread pool (PIL releases the GIL); the AI model consumes images
in order from a bounded prefetch window, so memory stays flat on huge disks.
"""
from __future__ import annotations
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
from PIL import Image
from . import config
from .engine import load_image
from .scanner import iter_images


@dataclass
class Hit:
    path: str
    score: float
    thumb: Image.Image


class ScanJob:
    def __init__(self, engine, cache, ref_embs, roots, full_scan, on_progress, on_hit, stop: threading.Event,
                 pause: threading.Event | None = None, on_state=None):
        self.engine, self.cache = engine, cache
        self.refs = np.stack(ref_embs).astype(np.float32)          # (k, 512)
        self.roots, self.skip_system = roots, full_scan
        self.on_progress, self.on_hit, self.stop = on_progress, on_hit, stop
        self.pause = pause or threading.Event()          # set = paused, clear = running
        self.on_state = on_state or (lambda state: None)
        self.scanned = self.hits = 0
        self._last = 0.0
        self._dir = ""

    # -- matching ---------------------------------------------------------
    def _match(self, path: str, embs, img: Image.Image | None):
        if not embs:
            return
        score = float((np.stack(embs) @ self.refs.T).max())         # cosine (vectors are unit-norm)
        if score < config.SCORE_FLOOR:
            return
        img = img or load_image(path)
        if img is None:
            return
        thumb = img.copy()
        thumb.thumbnail((config.THUMB, config.THUMB))
        self.hits += 1
        self.on_hit(Hit(path, score, thumb))

    def _tick(self, path: str, force: bool = False):
        self._dir = os.path.dirname(path)
        now = time.monotonic()
        if force or now - self._last > 0.2:
            self._last = now
            self.on_progress(self.scanned, self.hits, os.path.dirname(path))

    def _finish_fresh(self, fut, path, size, mtime):
        img = fut.result()
        embs = []
        if img is not None and min(img.size) >= config.MIN_IMAGE_SIDE:
            embs = [f.emb for f in self.engine.embed(img)]
        self.cache.put(path, size, mtime, embs)                     # also caches "no faces"
        self.scanned += 1
        self._match(path, embs, img)
        self._tick(path)

    def _gate(self):
        """Pause point, checked between images. Blocks while paused; a Stop always wins."""
        if not self.pause.is_set() or self.stop.is_set():
            return
        self.cache.commit()                                         # progress is safe on disk while paused
        self.on_progress(self.scanned, self.hits, self._dir)
        self.on_state("paused")
        while self.pause.is_set() and not self.stop.is_set():
            self.stop.wait(0.1)
        if not self.stop.is_set():
            self.on_state("running")

    # -- main loop --------------------------------------------------------
    def run(self) -> tuple[int, int]:
        pending: deque = deque()
        path = ""
        with ThreadPoolExecutor(config.LOADER_THREADS) as pool:
            for path in iter_images(self.roots, self.skip_system):
                self._gate()
                if self.stop.is_set():
                    break
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size < config.MIN_FILE_BYTES:
                    continue
                cached = self.cache.get(path, st.st_size, st.st_mtime_ns)
                if cached is not None:                              # already analysed: instant
                    self.scanned += 1
                    self._match(path, cached, None)
                    self._tick(path)
                    continue
                pending.append((pool.submit(load_image, path), path, st.st_size, st.st_mtime_ns))
                if len(pending) >= config.PREFETCH:
                    self._finish_fresh(*pending.popleft())
            while pending:
                self._gate()
                if self.stop.is_set():
                    break
                self._finish_fresh(*pending.popleft())
            for item in pending:
                item[0].cancel()
        self.cache.commit()
        self._tick(path, force=True)
        return self.scanned, self.hits
