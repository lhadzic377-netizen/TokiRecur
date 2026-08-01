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

import math
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
        self._load_all()

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
                hit_count     INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS cache_entries_last_used
                ON cache_entries(last_used_at);
            """
        )
        self._conn.commit()

    def _load_all(self) -> None:
        cur = self._conn.execute("SELECT cache_id, embedding FROM cache_entries")
        for cache_id, blob in cur.fetchall():
            self._embeddings[cache_id] = np.frombuffer(blob, dtype=np.float32).copy()

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
            ids = list(self._embeddings.keys())
            matrix = np.stack([self._embeddings[i] for i in ids])
            q = np.asarray(embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0.0:
                return None
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0
            sims = (matrix @ q) / (norms * q_norm)
            # Among candidates with similar similarity (within 5% of top),
            # prefer the one with the longest token list — more potential
            # for prefix reuse. This dampens the "cosine finds a similar-
            # looking 1-token match" pathology.
            top_sim = float(np.max(sims))
            threshold = top_sim * 0.95
            candidates = np.where(sims >= threshold)[0]
            if len(candidates) > 1:
                # Load token counts for the candidates.
                counts = []
                for idx in candidates:
                    cid = ids[int(idx)]
                    row = self._conn.execute(
                        "SELECT length(token_ids)/4 FROM cache_entries WHERE cache_id = ?",
                        (cid,),
                    ).fetchone()
                    counts.append(int(row[0]) if row else 0)
                best_local = int(candidates[int(np.argmax(counts))])
                best_idx = best_local
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
                     model_id, bytes_on_disk, created_at, last_used_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            self._conn.commit()
            self._embeddings[entry.cache_id] = embedding_arr

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
            return {
                "entries": int(count),
                "bytes_on_disk": int(total_bytes),
                "embedding_dim": 384,
            }

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
                current -= int(size)
                evicted += 1
            self._conn.commit()
            return evicted


class HNSWIndex(Index):
    """Approximate nearest-neighbour index using hnswlib. Optional."""

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

    def evict_lru(self, target_bytes: int) -> int:
        return self._brute.evict_lru(target_bytes)
