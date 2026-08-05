"""Tests for TreeCache/PersistentSlotPool integration into RecyclingEngine.

Routing is purely in-memory slot reuse: the engine must never call
``backend.cache_save`` / ``backend.cache_load`` (disk save/restore was
removed because it corrupted KV precision — cached outputs diverged from
baseline). These tests verify the engine works end-to-end against a fake
backend without any disk round-trip, that repeated prompts hit the
in-memory slot cache, and that the default components still construct.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from tokirecur.cache.tree import TreeCache
from tokirecur.core.recycling import RecyclingEngine


@dataclass
class _Backend:
    """Tracks saves/loads explicitly so tests can assert they never fire."""
    slot_kv: dict[int, list[int]] = field(default_factory=dict)
    saves: list[tuple[int, str]] = field(default_factory=list)
    loads: list[tuple[int, str]] = field(default_factory=list)

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def cache_info(self, slot_id: int) -> dict[str, Any]:
        return {"n_past": len(self.slot_kv.get(slot_id, [])), "model_id": "fake"}

    def complete_native(
        self,
        prompt: str,
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
    ) -> dict[str, Any]:
        toks = self.tokenize(prompt)
        prev = self.slot_kv.get(slot_id, [])
        reuse = 0
        for a, b in zip(prev, toks, strict=False):
            if a == b:
                reuse += 1
            else:
                break
        new_prefill = max(1, len(toks) - reuse)
        self.slot_kv[slot_id] = toks
        return {
            "text": "ok",
            "tokens_generated": 4,
            "prefill_tokens": new_prefill,
            "ttft_ms": 50.0,
            "total_ms": 100.0,
            "prefill_tokens_per_sec": new_prefill / 0.05,
        }

    def cache_save(self, slot_id: int, cache_id: str) -> dict[str, Any]:
        self.saves.append((slot_id, cache_id))
        return {"n_past": len(self.slot_kv.get(slot_id, [])), "bytes": 0, "model_id": "fake"}

    def cache_load(self, slot_id: int, cache_id: str) -> dict[str, Any]:
        self.loads.append((slot_id, cache_id))
        return {"n_past": 0, "model_id": "fake"}

    def clear_slot(self, slot_id: int) -> dict[str, Any]:
        self.slot_kv.pop(slot_id, None)
        return {"cleared": True}


class _BackendNoDisk:
    """A backend that doesn't even implement cache_save/cache_load."""

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def cache_info(self, slot_id: int) -> dict[str, Any]:
        return {"n_past": 0, "model_id": "fake"}

    def complete_native(self, prompt, *, slot_id, max_tokens=256, temperature=0.0, cache_prompt=True):
        toks = self.tokenize(prompt)
        return {
            "text": "ok",
            "tokens_generated": 4,
            "prefill_tokens": len(toks),
            "ttft_ms": 50.0,
            "total_ms": 100.0,
            "prefill_tokens_per_sec": len(toks) / 0.05,
        }


class _Index:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def touch(self, cache_id: str) -> None:
        pass


@pytest.fixture
def engine(tmp_path):
    """Build an engine with a fresh tree cache rooted at tmp_path."""
    backend = _Backend()
    index = _Index()
    tree = TreeCache(tmp_path)
    from tokirecur.cache.persistent_pool import PersistentSlotPool
    pool = PersistentSlotPool(max_slots=4, min_reuse_for_pooling=100)
    eng = RecyclingEngine(
        backend, index, tree_cache=tree, persistent_pool=pool,
    )
    return eng, backend, index, tree


class TestInMemoryOnly:
    def test_completion_never_touches_disk(self, engine):
        """The engine must not call cache_save/cache_load at all."""
        eng, backend, _index, _tree = engine
        eng.serve_native("hello world", max_tokens=4)
        assert backend.saves == [], "disk save must not be called"
        assert backend.loads == [], "disk restore must not be called"

    def test_repeated_prompt_hits_in_memory_slot_cache(self, engine):
        """Two consecutive identical prompts: the second lands on the
        same slot already holding the prefix, reuses it in memory, and
        still never touches disk."""
        eng, backend, _index, _tree = engine
        prompt = " ".join(f"alpha_{i}" for i in range(20))
        first = eng.serve_native(prompt, max_tokens=4)
        second = eng.serve_native(prompt, max_tokens=4)
        assert second.hit, "identical prompt should be a real hit"
        assert first.reuse_depth == 0, "first request has nothing cached"
        assert second.reuse_depth == 20, "identical prompt reuses the full prefix"
        assert backend.saves == []
        assert backend.loads == []

    def test_spill_and_reprompt_refills_without_restore(self, engine):
        """After a spill evicts a prefix, returning to the original
        prompt just re-prefills — no restore, no disk I/O."""
        eng, backend, _index, _tree = engine
        long_prompt = " ".join(f"alpha_{i}" for i in range(20))
        eng.serve_native(long_prompt, max_tokens=4)
        # Force a fresh slot by sending a long unrelated prompt.
        eng._max_context_tokens = 5
        eng.serve_native(" ".join(f"x{i}" for i in range(20)), max_tokens=4)
        eng._max_context_tokens = 10000
        result = eng.serve_native(long_prompt, max_tokens=4)
        assert result.output_text == "ok"
        assert backend.loads == [], "evicted prefix must re-prefill, not restore"
        assert backend.saves == []

    def test_backend_without_save_load_methods_works(self):
        """Regression: a backend that lacks cache_save/cache_load entirely
        (e.g. llama-server without --slot-save-path) must still work."""
        eng = RecyclingEngine(_BackendNoDisk(), _Index())
        prompt = " ".join(f"beta_{i}" for i in range(20))
        first = eng.serve_native(prompt, max_tokens=4)
        second = eng.serve_native(prompt, max_tokens=4)
        assert first.output_text == "ok"
        assert second.output_text == "ok"
        assert second.hit


class TestTreeCacheDefaults:
    def test_engine_constructs_default_tree_cache(self, tmp_path, monkeypatch):
        """No tree_cache passed → engine builds a TreeCache rooted at
        ``TOKI_SLOT_SAVE_DIR`` (or the home-dir default)."""
        monkeypatch.setenv("TOKI_SLOT_SAVE_DIR", str(tmp_path / "auto"))
        eng = RecyclingEngine(_Backend(), _Index())
        assert isinstance(eng.tree_cache, TreeCache)
        assert eng.tree_cache.cache_dir == tmp_path / "auto"
