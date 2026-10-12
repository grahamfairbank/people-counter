"""SQLite store: one row per unique person, plus a thumbnail on disk."""
import sqlite3
import threading
import time

from .config import DATA_DIR

THUMB_DIR = DATA_DIR / "thumbs"
THUMB_RETENTION_DAYS = 30


class Store:
    def __init__(self):
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(DATA_DIR / "people.db", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS persons (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   first_seen REAL NOT NULL,
                   last_seen REAL NOT NULL,
                   visits INTEGER NOT NULL DEFAULT 1,
                   merge_sim REAL)"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_first_seen ON persons(first_seen)")
        # One row per walk through the frame, with its appearance samples
        # (float32 blob, 240 values each) so re-entry matching can be re-run offline.
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS passes (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   person_id INTEGER NOT NULL,
                   first_seen REAL NOT NULL,
                   last_seen REAL NOT NULL,
                   hits INTEGER NOT NULL,
                   best_match_sim REAL,
                   merged INTEGER NOT NULL,
                   samples BLOB)"""
        )
        # Where each pass entered and left (fractions of the frame), added later.
        have = {r[1] for r in self._db.execute("PRAGMA table_info(passes)")}
        for col in ("entry_x REAL", "entry_y REAL", "exit_x REAL", "exit_y REAL", "by_position INTEGER"):
            if col.split()[0] not in have:
                self._db.execute(f"ALTER TABLE passes ADD COLUMN {col}")
        self._db.commit()

    def add(self, first_seen: float, last_seen: float) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO persons(first_seen, last_seen) VALUES (?, ?)", (first_seen, last_seen)
            )
            self._db.commit()
            return cur.lastrowid

    def touch(self, pid: int, last_seen: float):
        with self._lock:
            self._db.execute("UPDATE persons SET last_seen=? WHERE id=?", (last_seen, pid))
            self._db.commit()

    def revisit(self, pid: int, last_seen: float, sim: float):
        with self._lock:
            self._db.execute(
                "UPDATE persons SET last_seen=?, visits=visits+1, merge_sim=? WHERE id=?",
                (last_seen, sim, pid),
            )
            self._db.commit()

    def log_pass(self, pid, first_seen, last_seen, hits, best_match_sim, merged, samples, entry, exit, by_position):
        with self._lock:
            self._db.execute(
                "INSERT INTO passes(person_id, first_seen, last_seen, hits, best_match_sim, merged, samples, "
                "entry_x, entry_y, exit_x, exit_y, by_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pid, first_seen, last_seen, hits, best_match_sim, merged, samples, *entry, *exit, by_position),
            )
            self._db.commit()

    def delete(self, pid: int):
        with self._lock:
            self._db.execute("DELETE FROM persons WHERE id=?", (pid,))
            self._db.commit()

    def counts(self, start: float, end: float, bucket: float) -> dict[int, int]:
        """Unique people per bucket, keyed by bucket index from `start`."""
        with self._lock:
            rows = self._db.execute(
                "SELECT CAST((first_seen - ?) / ? AS INTEGER) AS b, COUNT(*) FROM persons "
                "WHERE first_seen >= ? AND first_seen < ? GROUP BY b",
                (start, bucket, start, end),
            ).fetchall()
        return dict(rows)

    def total_since(self, start: float) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM persons WHERE first_seen >= ?", (start,)
            ).fetchone()[0]

    def recent(self, limit: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, first_seen, last_seen, visits, merge_sim FROM persons "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("id", "first_seen", "last_seen", "visits", "merge_sim")
        return [dict(zip(keys, r)) for r in rows]

    def prune_thumbs(self):
        cutoff = time.time() - THUMB_RETENTION_DAYS * 86400
        for f in THUMB_DIR.glob("*.jpg"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
