"""Index over cached prompts.

Indices are pluggable. Implementations:

* :class:`BruteForceIndex` — exact cosine similarity via numpy. Fine for
  up to ~10k entries, which is plenty for a single-host deployment.
* :class:`HNSWIndex` — approximate nearest neighbours via hnswlib. Use
  when entries >= 10k. Optional dependency.

Both expose the same three methods used by the recycling engine:
  :meth:`search` -> Match | None
  :meth:`insert` -> None
  :meth:`touch`  -> None   (update LRU timestamp)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.recycling import CacheEntry


@dataclass
class Match:
    """A best-match search result."""

    cache_id: str
    similarity: float
    token_ids: list[int]
    prompt: str


class Index:
    """Abstract base."""

    def search(self, embedding: list[float]) -> Match | None:
        raise NotImplementedError

    def insert(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    def touch(self, cache_id: str) -> None:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def delete(self, cache_id: str) -> bool:
        """Delete an entry by cache_id. Returns True if it existed."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the underlying store. Idempotent; safe to skip."""

    def entries(self) -> list[dict]:
        """List stored entries (metadata only), newest use first."""
        raise NotImplementedError


class BruteForceIndex(Index):
    """Exact cosine similarity over all stored embeddings.

    Embeddings live in a numpy array. Metadata lives in SQLite for
    durability across restarts. The embedding BLOB is stored as packed
    float32 in the same SQLite DB to keep the implementation single-file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._embeddings: dict[str, np.ndarray] = {}
        self._token_counts: dict[str, int] = {}
        self._load_all()

    def close(self) -> None:
        """Checkpoint WAL and release the sqlite handle.

        Needed on Windows, where an open handle blocks file deletion
        (e.g. temp-dir cleanup in tests).
        """
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass
            self._conn.close()
            self._conn = None

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_id      TEXT PRIMARY KEY,
                prompt        TEXT NOT NULL,
                token_ids     BLOB NOT NULL,
                embedding     BLOB NOT NULL,
                n_past        INTEGER NOT NULL,
                model_id      TEXT NOT NULL,
                bytes_on_disk INTEGER NOT NULL,
                created_at    REAL NOT NULL,
                last_used_at  REAL NOT NULL,
                hit_count     INTEGER NOT NULL DEFAULT 0,
                token_count   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS cache_entries_last_used
                ON cache_entries(last_used_at);
            """
        )
        # Add token_count column if an older DB predates this schema.
        cols = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(cache_entries)"
        ).fetchall()}
        if "token_count" not in cols:
            self._conn.execute(
                "ALTER TABLE cache_entries ADD COLUMN token_count INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.commit()

    def _load_all(self) -> None:
        cur = self._conn.execute(
            "SELECT cache_id, embedding, token_count FROM cache_entries"
        )
        for cache_id, blob, token_count in cur.fetchall():
            self._embeddings[cache_id] = np.frombuffer(blob, dtype=np.float32).copy()
            self._token_counts[cache_id] = int(token_count or 0)

    def search(self, embedding: list[float]) -> Match | None:
        """Find the best match by cosine similarity.

        Note: the prefix check (compute_reuse_depth in core.recycling)
        is the authoritative gate. This search is only a retrieval
        heuristic — when multiple entries are similarly similar in
        embedding space, the longest one is preferred because it has
        more potential prefix to reuse.
        """
        with self._lock:
            if not self._embeddings:
                return None
            q = np.asarray(embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0.0:
                return None
            # Only entries whose width matches the live model are comparable.
            # Switching models changes the embedding width (2048 vs 4096 on
            # this box), and np.stack refuses mixed shapes — which turned a
            # model switch into a hard 500 on EVERY request, because two
            # legacy rows of the old width were still on disk. Skip them
            # instead: they are simply not searchable by this model.
            dim = int(q.shape[0])
            ids = [k for k, v in self._embeddings.items() if v.shape[0] == dim]
            if not ids:
                return None
            matrix = np.stack([self._embeddings[i] for i in ids])
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0
            sims = (matrix @ q) / (norms * q_norm)
            # Among candidates with similar similarity (within 5% of top),
            # prefer the one with the longest token list — more potential
            # for prefix reuse. Token counts live in their own column so we
            # fetch them in one SQL roundtrip instead of one per candidate.
            top_sim = float(np.max(sims))
            threshold = top_sim * 0.95
            candidates = np.where(sims >= threshold)[0]
            if len(candidates) > 1:
                cid_list = [ids[int(idx)] for idx in candidates]
                placeholders = ",".join("?" * len(cid_list))
                count_rows = self._conn.execute(
                    f"SELECT cache_id, token_count FROM cache_entries "
                    f"WHERE cache_id IN ({placeholders})",
                    cid_list,
                ).fetchall()
                count_map = {row[0]: row[1] for row in count_rows}
                counts = [count_map.get(cid, 0) for cid in cid_list]
                best_local = candidates[int(np.argmax(counts))]
                best_idx = int(best_local)
            else:
                best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim <= 0.0:
                return None
            best_id = ids[best_idx]
            row = self._conn.execute(
                "SELECT prompt, token_ids FROM cache_entries WHERE cache_id = ?",
                (best_id,),
            ).fetchone()
            if row is None:
                return None
            prompt, token_blob = row
            token_ids = list(np.frombuffer(token_blob, dtype=np.int32).tolist())
            return Match(
                cache_id=best_id,
                similarity=best_sim,
                token_ids=token_ids,
                prompt=prompt,
            )

    def insert(self, entry: CacheEntry) -> None:
        with self._lock:
            embedding_arr = np.asarray(entry.embedding, dtype=np.float32)
            token_arr = np.asarray(entry.token_ids, dtype=np.int32)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                    (cache_id, prompt, token_ids, embedding, n_past,
                     model_id, bytes_on_disk, created_at, last_used_at,
                     hit_count, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.cache_id,
                    entry.prompt,
                    token_arr.tobytes(),
                    embedding_arr.tobytes(),
                    entry.n_past,
                    entry.model_id,
                    entry.bytes_on_disk,
                    entry.created_at,
                    entry.last_used_at,
                    entry.hit_count,
                    int(token_arr.size),
                ),
            )
            self._conn.commit()
            self._embeddings[entry.cache_id] = embedding_arr
            self._token_counts[entry.cache_id] = int(token_arr.size)

    def touch(self, cache_id: str) -> None:
        with self._lock:
            now = time.time()
            self._conn.execute(
                "UPDATE cache_entries SET last_used_at = ?, hit_count = hit_count + 1 "
                "WHERE cache_id = ?",
                (now, cache_id),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(bytes_on_disk), 0) FROM cache_entries"
            ).fetchone()
            count, total_bytes = row
            # Report the actual stored embedding dimension so mixed-dim
            # corruption (e.g. a 4096-dim entry inserted after a 384-dim
            # one) shows up immediately instead of crashing inside
            # search() with a confusing numpy error.
            if self._embeddings:
                first = next(iter(self._embeddings.values()))
                embedding_dim = int(first.shape[0])
            else:
                embedding_dim = 0
            return {
                "entries": int(count),
                "bytes_on_disk": int(total_bytes),
                "embedding_dim": embedding_dim,
            }

    def delete(self, cache_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache_entries WHERE cache_id = ?", (cache_id,)
            )
            self._conn.commit()
            removed = cur.rowcount > 0
            if removed:
                self._embeddings.pop(cache_id, None)
                self._token_counts.pop(cache_id, None)
            return removed

    def entries(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT cache_id, prompt, token_count, bytes_on_disk, "
                "last_used_at, hit_count FROM cache_entries "
                "ORDER BY last_used_at DESC"
            ).fetchall()
            return [
                {
                    "cache_id": row[0],
                    "prompt": row[1],
                    "token_count": int(row[2] or 0),
                    "bytes_on_disk": int(row[3] or 0),
                    "last_used_at": float(row[4] or 0),
                    "hit_count": int(row[5] or 0),
                }
                for row in rows
            ]

    def evict_lru(self, target_bytes: int) -> int:
        """Evict oldest entries until the cache is at most `target_bytes`.
        Returns the number of entries evicted."""
        with self._lock:
            current = self.stats()["bytes_on_disk"]
            if current <= target_bytes:
                return 0
            evicted = 0
            cur = self._conn.execute(
                "SELECT cache_id, bytes_on_disk FROM cache_entries "
                "ORDER BY last_used_at ASC"
            )
            for cache_id, size in cur.fetchall():
                if current <= target_bytes:
                    break
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE cache_id = ?", (cache_id,)
                )
                self._embeddings.pop(cache_id, None)
                self._token_counts.pop(cache_id, None)
                current -= int(size)
                evicted += 1
            self._conn.commit()
            return evicted


class HNSWIndex(Index):
    """Approximate nearest-neighbour index using hnswlib. Optional.

    NOTE: this class is currently a thin wrapper that delegates every
    method to the in-memory :class:`BruteForceIndex`. The hnswlib
    import is verified at construction time so a missing optional
    dependency surfaces clearly, but the ANN code path is not yet
    wired up. Treat this as a placeholder until the ANN search is
    implemented. For now, prefer :class:`BruteForceIndex` directly.
    """

    def __init__(self, db_path: str | Path, dim: int = 384, max_elements: int = 100_000):
        try:
            import hnswlib  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Install the [index] extra to use HNSWIndex: "
                "pip install tokirecur[index]"
            ) from e
        self.dim = dim
        self._brute = BruteForceIndex(db_path)

    def search(self, embedding: list[float]) -> Match | None:
        return self._brute.search(embedding)

    def insert(self, entry: CacheEntry) -> None:
        self._brute.insert(entry)

    def touch(self, cache_id: str) -> None:
        self._brute.touch(cache_id)

    def stats(self) -> dict:
        return self._brute.stats()

    def delete(self, cache_id: str) -> bool:
        return self._brute.delete(cache_id)

    def entries(self) -> list[dict]:
        return self._brute.entries()

    def evict_lru(self, target_bytes: int) -> int:
        return self._brute.evict_lru(target_bytes)
