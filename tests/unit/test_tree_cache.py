"""Unit tests for the tree cache and persistent slot pool."""

import tempfile
from pathlib import Path

import pytest

from tokirecur.cache.tree import (
    BLOCK_SIZE,
    TreeCache,
    block_id,
    block_ids_for_tokens,
)


class TestBlockID:
    def test_stable(self):
        a = block_id([1, 2, 3])
        b = block_id([1, 2, 3])
        assert a == b

    def test_distinguishes_content(self):
        a = block_id([1, 2, 3])
        b = block_id([1, 2, 4])
        assert a != b

    def test_distinguishes_order(self):
        a = block_id([1, 2, 3])
        b = block_id([3, 2, 1])
        assert a != b


class TestBlockIDsForTokens:
    def test_short(self):
        ids = block_ids_for_tokens(None, [1, 2, 3])
        assert len(ids) == 1

    def test_exact_block(self):
        ids = block_ids_for_tokens(None, list(range(BLOCK_SIZE)))
        assert len(ids) == 1

    def test_two_blocks(self):
        ids = block_ids_for_tokens(None, list(range(BLOCK_SIZE + 1)))
        assert len(ids) == 2

    def test_sharing(self):
        """Two prefixes with identical content get the same block IDs."""
        a = block_ids_for_tokens(None, [1, 2, 3, 4, 5])
        b = block_ids_for_tokens(None, [1, 2, 3, 4, 5])
        assert a == b

    def test_sharing_partial(self):
        """Two prefixes that diverge at position N produce different block IDs from that block onward."""
        a = block_ids_for_tokens(None, [1, 2, 3, 4, 5])
        b = block_ids_for_tokens(None, [1, 2, 3, 9, 10])
        # With BLOCK_SIZE > 5, both fit in one block but the content
        # differs → block IDs differ.
        assert a != b


class TestTreeCache:
    def test_register_lookup_empty(self):
        with tempfile.TemporaryDirectory() as d:
            tc = TreeCache(Path(d))
            depth = tc.lookup("synth", list(range(100)))
            assert depth == 0

    def test_register_lookup_with_data(self):
        with tempfile.TemporaryDirectory() as d:
            tc = TreeCache(Path(d))
            tc.register("c1", list(range(100)), save_fn=lambda bid: b"data")
            depth = tc.lookup("c1", list(range(100)))
            assert depth == 100

    def test_register_shares_blocks(self):
        """Two entries sharing a prefix share the leading blocks."""
        with tempfile.TemporaryDirectory() as d:
            tc = TreeCache(Path(d))
            # c1 has 200 tokens, c2 has 150 tokens — first 128 tokens are
            # identical between the two.
            tc.register("c1", list(range(200)), save_fn=lambda bid: b"x" * 64)
            tc.register("c2", list(range(150)), save_fn=lambda bid: b"y" * 32)
            stats = tc.stats()
            # c1 spans 4 blocks (0-63, 64-127, 128-191, 192-199).
            # c2 spans 3 blocks (0-63, 64-127, 128-149).
            # Blocks 0 and 1 are identical between the two → shared.
            # Block 2 differs (192 vs 22 tokens of content) → not shared.
            # Block 3 only exists for c1.
            assert stats.total_blocks == 7
            assert stats.unique_blocks == 5
            assert stats.block_reuse_ratio > 0.5

    def test_evict_lru(self):
        with tempfile.TemporaryDirectory() as d:
            tc = TreeCache(Path(d))
            # Two entries that share leading tokens.
            tc.register("c1", list(range(100)), save_fn=lambda bid: b"x" * 64)
            tc.register("c2", list(range(200, 300)), save_fn=lambda bid: b"y" * 32)
            tc.unregister("c1")
            deleted = tc.garbage_collect()
            # c1 had 2 blocks; both are now unreferenced.
            assert deleted == 2

    def test_stats_disk_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            tc = TreeCache(Path(d))
            tc.register("c1", list(range(100)), save_fn=lambda bid: b"x" * 100)
            stats = tc.stats()
            assert stats.disk_bytes == 200  # 2 blocks × 100 bytes