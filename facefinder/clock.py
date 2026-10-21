"""Pause-aware stopwatch + duration formatting (Qt-free)."""
from __future__ import annotations
import time


class ScanClock:
    """Counts only *active* scanning time: paused time is excluded."""

    def __init__(self, now=time.monotonic):
        self._now = now
        self.reset()

    def reset(self):
        self._acc, self._t0, self.active = 0.0, None, False

    def start(self):
        self.reset()
        self.active, self._t0 = True, self._now()

    def pause(self):
        if self._t0 is not None:
            self._acc += self._now() - self._t0
            self._t0 = None

    def resume(self):
        if self.active and self._t0 is None:
            self._t0 = self._now()

    def stop(self) -> float:
        self.pause()
        self.active = False
        return self._acc

    @property
    def running(self) -> bool:
        return self._t0 is not None

    def elapsed(self) -> float:
        return self._acc + (self._now() - self._t0 if self._t0 is not None else 0.0)


def clock_text(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def human_text(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{s}s"
