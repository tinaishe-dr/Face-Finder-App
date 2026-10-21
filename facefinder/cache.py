"""SQLite embedding cache. Re-scans only touch new/changed files (path + size + mtime)."""
from __future__ import annotations
import sqlite3
import numpy as np


class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER);
            CREATE TABLE IF NOT EXISTS faces(id INTEGER PRIMARY KEY, path TEXT, emb BLOB);
            CREATE INDEX IF NOT EXISTS faces_path ON faces(path);
        """)
        self._pending = 0

    def get(self, path: str, size: int, mtime: int):
        row = self.db.execute("SELECT size, mtime FROM images WHERE path=?", (path,)).fetchone()
        if not row or row[0] != size or row[1] != mtime:
            return None
        rows = self.db.execute("SELECT emb FROM faces WHERE path=?", (path,)).fetchall()
        return [np.frombuffer(r[0], dtype=np.float32) for r in rows]

    def put(self, path: str, size: int, mtime: int, embs):
        self.db.execute("DELETE FROM faces WHERE path=?", (path,))
        self.db.execute("INSERT OR REPLACE INTO images VALUES(?,?,?)", (path, size, mtime))
        self.db.executemany("INSERT INTO faces(path, emb) VALUES(?,?)",
                            [(path, np.asarray(e, dtype=np.float32).tobytes()) for e in embs])
        self._pending += 1
        if self._pending >= 200:
            self.commit()

    def commit(self):
        self.db.commit()
        self._pending = 0

    def close(self):
        self.commit()
        self.db.close()
