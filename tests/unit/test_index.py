"""Unit tests for the BruteForceIndex."""

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


@pytest.fixture
def idx(tmp_path):
    # A real path in a temp dir: NamedTemporaryFile keeps an open handle
    # that Windows won't share with sqlite, and the index must be closed
    # before the dir is cleaned up (Windows refuses to delete open files).
    index = BruteForceIndex(tmp_path / "idx.sqlite")
    yield index
    index.close()


class TestBruteForceIndex:
    def test_empty_search(self, idx):
        assert idx.search([0.1, 0.2, 0.3]) is None

    def test_insert_and_find(self, idx):
        e = _make_entry("a", "prompt A", [1.0, 0.0, 0.0, 0.0])
        idx.insert(e)
        m = idx.search([0.9, 0.1, 0.0, 0.0])
        assert m is not None
        assert m.cache_id == "a"
        assert m.similarity > 0.9

    def test_best_match_wins(self, idx):
        idx.insert(_make_entry("x", "x axis", _orthogonal(4, 0)))
        idx.insert(_make_entry("y", "y axis", _orthogonal(4, 1)))
        m = idx.search([0.99, 0.01, 0.0, 0.0])
        assert m.cache_id == "x"

    def test_touch_updates_last_used(self, idx):
        e = _make_entry("a", "p", [1.0, 0.0])
        idx.insert(e)
        idx.touch("a")
        # re-search still works
        assert idx.search([1.0, 0.0]).cache_id == "a"

    def test_evict_lru(self, idx):
        for i in range(5):
            idx.insert(_make_entry(f"k{i}", f"p{i}", [float(i), 1.0], n_past=10))
        n = idx.evict_lru(target_bytes=2000)
        assert n >= 1
        # remaining entries should be the most recently used
        stats = idx.stats()
        assert stats["entries"] < 5

    def test_stats(self, idx):
        for i in range(3):
            idx.insert(_make_entry(f"k{i}", f"p{i}", [1.0, 0.0], n_past=10))
        s = idx.stats()
        assert s["entries"] == 3
        assert s["bytes_on_disk"] == 3000

    def test_persistence(self, tmp_path):
        path = tmp_path / "idx.sqlite"
        idx1 = BruteForceIndex(path)
        idx1.insert(_make_entry("a", "p", [1.0, 0.0]))
        idx1.close()
        # new index reads from disk
        idx2 = BruteForceIndex(path)
        m = idx2.search([1.0, 0.0])
        assert m is not None
        assert m.cache_id == "a"
        idx2.close()


class TestMixedEmbeddingWidths:
    """A model switch changes the embedding width.

    The index keeps every entry from every model it has ever served, so a
    width change leaves two shapes in the same dict. `np.stack` refuses
    mixed shapes, which made the search raise
    ``all input arrays must have the same shape`` and turned every request
    into a 500 — the failure looked like a projector bug because it first
    showed up on an image request. Rows of the wrong width are simply not
    comparable and must be skipped, not stacked.
    """

    def test_search_skips_entries_from_another_model_width(self, idx):
        idx.insert(_make_entry("old-2048", "old", _orthogonal(8, 0)))
        idx.insert(_make_entry("new-4096", "new", _orthogonal(16, 1)))
        # Query in the live model's width: the old row must not poison it.
        match = idx.search(_orthogonal(16, 1))
        assert match is not None
        assert match.cache_id == "new-4096"
        # And the reverse: querying the old width still works.
        match_old = idx.search(_orthogonal(8, 0))
        assert match_old is not None
        assert match_old.cache_id == "old-2048"

    def test_search_returns_none_when_no_width_matches(self, idx):
        idx.insert(_make_entry("old", "old", _orthogonal(8, 0)))
        assert idx.search(_orthogonal(32, 0)) is None

    def test_insert_then_reload_keeps_mixed_widths_searchable(self, tmp_path):
        db = tmp_path / "mixed.sqlite"
        first = BruteForceIndex(db)
        first.insert(_make_entry("old", "old", _orthogonal(8, 0)))
        first.close()
        # Restart, add a wider model, and search the narrow one again.
        second = BruteForceIndex(db)
        second.insert(_make_entry("new", "new", _orthogonal(16, 2)))
        assert second.search(_orthogonal(8, 0)) is not None
        assert second.search(_orthogonal(16, 2)) is not None
        second.close()
