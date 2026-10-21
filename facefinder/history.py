"""Scan history (SQLite). Stores what was searched for, how, and every candidate match,
so a past scan can be re-opened instantly or re-run with the same reference + scope."""
from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
import numpy as np
from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans(id INTEGER PRIMARY KEY, started REAL, mode TEXT, roots TEXT,
    threshold REAL, scanned INTEGER, cancelled INTEGER, elapsed REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS scan_refs(scan_id INTEGER, crop BLOB, emb BLOB);
CREATE TABLE IF NOT EXISTS scan_results(scan_id INTEGER, path TEXT, score REAL);
CREATE INDEX IF NOT EXISTS res_scan ON scan_results(scan_id);
CREATE INDEX IF NOT EXISTS ref_scan ON scan_refs(scan_id);
"""


class History:
    def __init__(self, path_fn):
        self._path_fn = path_fn           # callable: storage folder can change at runtime

    @contextmanager
    def _db(self):
        db = sqlite3.connect(str(self._path_fn()))
        db.executescript(_SCHEMA)
        if "elapsed" not in {r[1] for r in db.execute("PRAGMA table_info(scans)")}:      # older history.db
            db.execute("ALTER TABLE scans ADD COLUMN elapsed REAL DEFAULT 0")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def add_scan(self, mode, roots, threshold, scanned, cancelled, refs, results, elapsed=0.0) -> int:
        """refs: [(png_bytes, emb)], results: [(path, score)]"""
        with self._db() as db:
            sid = db.execute("INSERT INTO scans(started,mode,roots,threshold,scanned,cancelled,elapsed) VALUES(?,?,?,?,?,?,?)",
                             (time.time(), mode, json.dumps(roots), threshold, scanned, int(cancelled), float(elapsed))).lastrowid
            db.executemany("INSERT INTO scan_refs VALUES(?,?,?)",
                           [(sid, c, np.asarray(e, np.float32).tobytes()) for c, e in refs])
            db.executemany("INSERT INTO scan_results VALUES(?,?,?)", [(sid, p, float(s)) for p, s in results])
            old = [r[0] for r in db.execute("SELECT id FROM scans ORDER BY id DESC LIMIT -1 OFFSET ?",
                                            (config.MAX_HISTORY,))]
            for o in old:
                self._delete(db, o)
        return sid

    @staticmethod
    def _delete(db, sid):
        for t in ("scan_results", "scan_refs"):
            db.execute(f"DELETE FROM {t} WHERE scan_id=?", (sid,))
        db.execute("DELETE FROM scans WHERE id=?", (sid,))

    def list_scans(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("""SELECT s.id, s.started, s.mode, s.roots, s.threshold, s.scanned, s.cancelled, s.elapsed,
                (SELECT COUNT(*) FROM scan_results r WHERE r.scan_id=s.id AND r.score>=s.threshold)
                FROM scans s ORDER BY s.id DESC""").fetchall()
        return [dict(id=r[0], started=r[1], mode=r[2], roots=json.loads(r[3]), threshold=r[4],
                     scanned=r[5], cancelled=bool(r[6]), elapsed=r[7] or 0.0, matches=r[8]) for r in rows]

    def get_scan(self, sid):
        return next((s for s in self.list_scans() if s["id"] == sid), None)

    def get_refs(self, sid):
        with self._db() as db:
            rows = db.execute("SELECT crop, emb FROM scan_refs WHERE scan_id=?", (sid,)).fetchall()
        return [(c, np.frombuffer(e, np.float32).copy()) for c, e in rows]

    def first_crop(self, sid):
        with self._db() as db:
            r = db.execute("SELECT crop FROM scan_refs WHERE scan_id=? LIMIT 1", (sid,)).fetchone()
        return r[0] if r else None

    def get_results(self, sid):
        with self._db() as db:
            return db.execute("SELECT path, score FROM scan_results WHERE scan_id=? ORDER BY score DESC",
                              (sid,)).fetchall()

    def delete(self, sid):
        with self._db() as db:
            self._delete(db, sid)

    def clear(self):
        with self._db() as db:
            for t in ("scan_results", "scan_refs", "scans"):
                db.execute(f"DELETE FROM {t}")
