"""Unit tests for the core recycling engine."""

import pytest

from tokirecur.core.recycling import (
    cosine_similarity,
    compute_reuse_depth,
    make_cache_id,
)


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_empty(self):
        assert cosine_similarity([], []) == 0.0


class TestComputeReuseDepth:
    def test_identical(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        assert compute_reuse_depth(a, b) == 5

    def test_prefix_match(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 9, 10, 11]
        assert compute_reuse_depth(a, b) == 3

    def test_no_match(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        assert compute_reuse_depth(a, b) == 0

    def test_cached_shorter(self):
        a = [1, 2, 3]
        b = [1, 2, 3, 4, 5]
        assert compute_reuse_depth(a, b) == 3

    def test_new_shorter(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3]
        assert compute_reuse_depth(a, b) == 3

    def test_empty(self):
        assert compute_reuse_depth([], [1, 2, 3]) == 0
        assert compute_reuse_depth([1, 2, 3], []) == 0
        assert compute_reuse_depth([], []) == 0


class TestMakeCacheId:
    def test_deterministic(self):
        a = make_cache_id("hello", "llama-3.1-8b")
        b = make_cache_id("hello", "llama-3.1-8b")
        assert a == b

    def test_different_prompt(self):
        a = make_cache_id("hello", "llama-3.1-8b")
        b = make_cache_id("world", "llama-3.1-8b")
        assert a != b

    def test_different_model(self):
        a = make_cache_id("hello", "llama-3.1-8b")
        b = make_cache_id("hello", "qwen-9b")
        assert a != b

    def test_length(self):
        assert len(make_cache_id("x", "y")) == 24
