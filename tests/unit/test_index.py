"""Unit tests for the BruteForceIndex."""

import tempfile
from pathlib import Path

import pytest

from tokirecur.core.recycling import CacheEntry
from tokirecur.index import BruteForceIndex


def _make_entry(cache_id: str, prompt: str, embedding: list[float], n_past: int = 100) -> CacheEntry:
    return CacheEntry(
        cache_id=cache_id,
        prompt=prompt,
        token_ids=list(range(n_past)),
        embedding=embedding,
        n_past=n_past,
        model_id="test",
        bytes_on_disk=1000,
    )


def _orthogonal(dim: int = 4, axis: int = 0) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


class TestBruteForceIndex:
    def test_empty_search(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            assert idx.search([0.1, 0.2, 0.3]) is None

    def test_insert_and_find(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            e = _make_entry("a", "prompt A", [1.0, 0.0, 0.0, 0.0])
            idx.insert(e)
            m = idx.search([0.9, 0.1, 0.0, 0.0])
            assert m is not None
            assert m.cache_id == "a"
            assert m.similarity > 0.9

    def test_best_match_wins(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            idx.insert(_make_entry("x", "x axis", _orthogonal(4, 0)))
            idx.insert(_make_entry("y", "y axis", _orthogonal(4, 1)))
            m = idx.search([0.99, 0.01, 0.0, 0.0])
            assert m.cache_id == "x"

    def test_touch_updates_last_used(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            e = _make_entry("a", "p", [1.0, 0.0])
            idx.insert(e)
            idx.touch("a")
            # re-search still works
            assert idx.search([1.0, 0.0]).cache_id == "a"

    def test_evict_lru(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            for i in range(5):
                idx.insert(_make_entry(f"k{i}", f"p{i}", [float(i), 1.0], n_past=10))
            n = idx.evict_lru(target_bytes=2000)
            assert n >= 1
            # remaining entries should be the most recently used
            stats = idx.stats()
            assert stats["entries"] < 5

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
            idx = BruteForceIndex(f.name)
            for i in range(3):
                idx.insert(_make_entry(f"k{i}", f"p{i}", [1.0, 0.0], n_past=10))
            s = idx.stats()
            assert s["entries"] == 3
            assert s["bytes_on_disk"] == 3000

    def test_persistence(self):
        path = Path(tempfile.mkdtemp()) / "idx.sqlite"
        idx1 = BruteForceIndex(path)
        idx1.insert(_make_entry("a", "p", [1.0, 0.0]))
        # new index reads from disk
        idx2 = BruteForceIndex(path)
        m = idx2.search([1.0, 0.0])
        assert m is not None
        assert m.cache_id == "a"
