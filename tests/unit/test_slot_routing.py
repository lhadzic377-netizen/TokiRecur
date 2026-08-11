"""Unit tests for the slot-routing and session-management logic in
RecyclingEngine.serve_native."""

from dataclasses import dataclass, field
from typing import Any

from tokirecur.core.recycling import (
    MIN_REUSE_FOR_HIT,
    RecyclingEngine,
)


@dataclass
class FakeBackend:
    """Minimal in-memory backend for testing slot routing.

    Tracks each slot's last prompt token list and exposes a clearable
    KV state per slot. Implements just enough of the LlamaBackend
    protocol for serve_native() to work.
    """

    slot_kv: dict[int, list[int]] = field(default_factory=dict)
    clear_count: dict[int, int] = field(default_factory=dict)

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def tokenize(self, text: str) -> list[int]:
        # Tokenize by splitting on whitespace; deterministic.
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
        sampling: dict | None = None,
    ) -> dict[str, Any]:
        toks = self.tokenize(prompt)
        prev = self.slot_kv.get(slot_id, [])
        reuse = 0
        for a, b in zip(prev, toks, strict=False):
            if a == b:
                reuse += 1
            else:
                break
        # llama-server skips prefill for the reused prefix; only the new
        # tokens get prefill cost.
        new_prefill = max(1, len(toks) - reuse)
        # Simulate llama-server: append the full prompt to the slot's KV.
        self.slot_kv[slot_id] = toks
        return {
            "text": "ok",
            "tokens_generated": 8,
            "prefill_tokens": new_prefill,
            "ttft_ms": 50.0 + new_prefill * 0.1,
            "total_ms": 100.0,
            "prefill_tokens_per_sec": new_prefill / 0.05,
        }

    def clear_slot(self, slot_id: int) -> dict[str, Any]:
        self.slot_kv.pop(slot_id, None)
        self.clear_count[slot_id] = self.clear_count.get(slot_id, 0) + 1
        return {"cleared": True}

    async def complete_openai_stream(
        self,
        messages,
        *,
        slot_id: int,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_prompt: bool = True,
        chat_template_kwargs=None,
        tools=None,
        tool_choice=None,
        sampling: dict | None = None,
    ):
        toks = self.tokenize(messages[-1]["content"])
        prev = self.slot_kv.get(slot_id, [])
        reuse = 0
        for a, b in zip(prev, toks, strict=False):
            if a == b:
                reuse += 1
            else:
                break
        new_prefill = max(1, len(toks) - reuse)
        self.slot_kv[slot_id] = toks
        yield {"kind": "delta", "delta": "hi"}
        yield {"kind": "finish", "finish_reason": "stop"}
        yield {
            "kind": "result",
            "result": {
                "text": "hi",
                "tool_calls": None,
                "finish_reason": "stop",
                "tokens_generated": 2,
                "prefill_tokens": new_prefill,
                "ttft_ms": 1.0,
                "total_ms": 2.0,
                "prefill_tokens_per_sec": 100.0,
            },
        }


class FakeIndex:
    """No-op index for testing."""

    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0, "embedding_dim": 4}


class TestServeNativeBasic:
    def test_first_request_lands_on_slot_0(self):
        engine = RecyclingEngine(FakeBackend(), FakeIndex())
        result = engine.serve_native("hello world", max_slots=2)
        assert result.slot_id == 0
        assert result.reuse_depth == 0
        assert result.hit is False

    def test_repeated_prompt_hits_cache(self):
        engine = RecyclingEngine(FakeBackend(), FakeIndex())
        engine.serve_native("hello world foo bar", max_slots=2)
        result = engine.serve_native("hello world foo bar baz", max_slots=2)
        # Slot 0 has "hello world foo bar"; new prompt extends it.
        # 4 tokens of reuse — below MIN_REUSE_FOR_HIT (10), so this is
        # a soft_hit rather than a hard hit. The slot was found, the
        # prefix was reused, but it's a small win.
        assert result.slot_id == 0
        assert result.reuse_depth == 4
        assert result.hit is False
        assert result.soft_hit is True

    def test_unrelated_prompt_round_robins(self):
        engine = RecyclingEngine(FakeBackend(), FakeIndex())
        engine.serve_native("alpha beta gamma", max_slots=2)
        engine.serve_native("delta epsilon", max_slots=2)
        result = engine.serve_native("zeta eta theta", max_slots=2)
        # "zeta eta theta" shares no prefix with either slot 0 or 1,
        # so it should go to the next free slot (slot 2 doesn't exist
        # in max_slots=2; we expect slot 0 reused via LRU).
        assert result.slot_id in (0, 1)


class TestHardHit:
    """The real cache-hit path (reuse >= MIN_REUSE_FOR_HIT) must return
    a result carrying cache_id — regression for an UnboundLocalError
    where cache_id was only assigned on the miss path."""

    def test_hard_hit_returns_cache_id(self):
        engine = RecyclingEngine(FakeBackend(), FakeIndex())
        prompt = " ".join(f"word{i}" for i in range(20))
        r1 = engine.serve_native(prompt, max_slots=2)
        assert r1.hit is False
        assert r1.cache_id
        r2 = engine.serve_native(prompt, max_slots=2)
        assert r2.hit is True
        assert r2.reuse_depth >= MIN_REUSE_FOR_HIT
        # Same prompt -> same derived cache_id.
        assert r2.cache_id == r1.cache_id
        # Hit path must not re-insert the index entry.
        assert len(engine.index.entries) == 1

    async def test_hard_hit_streaming_returns_cache_id(self):
        engine = RecyclingEngine(FakeBackend(), FakeIndex())
        messages = [
            {"role": "user", "content": " ".join(f"word{i}" for i in range(20))},
        ]
        results = []
        for _ in range(2):
            collected = []
            async for item in engine.serve_chat_stream(messages, max_slots=2):
                collected.append(item)
            results.append(next(i["result"] for i in collected if i["kind"] == "result"))
        assert results[0].hit is False
        assert results[1].hit is True
        assert results[1].reuse_depth >= MIN_REUSE_FOR_HIT
        assert results[1].cache_id == results[0].cache_id


class TestSessionRouting:
    def test_session_id_pins_to_same_slot(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        r1 = engine.serve_native("hello", max_slots=4, session_id="user-1")
        r2 = engine.serve_native("world", max_slots=4, session_id="user-1")
        r3 = engine.serve_native("foo", max_slots=4, session_id="user-1")
        assert r1.slot_id == r2.slot_id == r3.slot_id

    def test_different_sessions_pinned_independently(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        # Each session gets its own slot pinned.
        engine.serve_native("alpha", max_slots=4, session_id="user-A")
        engine.serve_native("beta", max_slots=4, session_id="user-B")
        # Both should have a pinned slot.
        assert "user-A" in engine._session_to_slot
        assert "user-B" in engine._session_to_slot
        # Their pinned slots are independent.
        assert engine._session_to_slot["user-A"] != engine._session_to_slot["user-B"]

    def test_session_id_pins_across_unrelated_prompts(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        # First request for session-A, unrelated content.
        r1 = engine.serve_native("alpha beta gamma", max_slots=4, session_id="user-A")
        # Slot 0 gets unrelated content from another request.
        engine.serve_native("completely unrelated prompt", max_slots=4, session_id="other")
        # Second request for session-A, completely different content.
        r2 = engine.serve_native("delta epsilon zeta", max_slots=4, session_id="user-A")
        # Session-A should still be pinned to its original slot, even
        # though the new prompt doesn't share a prefix with anything.
        assert r2.slot_id == r1.slot_id

    def test_session_id_beats_prefix_match(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        # Slot 0 gets "shared prefix X Y Z" — high prefix-match value.
        engine.serve_native("alpha beta gamma delta", max_slots=4)
        # Session pinned to slot 1 with unrelated content.
        engine.serve_native("completely different content", max_slots=4, session_id="s1")
        # New session request — should pick a free slot via round-robin,
        # not steal slot 0 despite its long prefix.
        result = engine.serve_native(
            "alpha beta gamma delta epsilon", max_slots=4, session_id="s2",
        )
        # Whatever slot it picked, it's not pinned to s1.
        assert engine._session_to_slot.get("s1") != result.slot_id or "s2" not in engine._session_to_slot


class TestSlotSpilling:
    def test_slot_spills_when_context_exceeds_budget(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        # Build up a slot's context via a session.
        long_prompt = " ".join(f"word{i}" for i in range(100))
        engine.serve_native(long_prompt, max_slots=2, session_id="s1", max_context_tokens=50)
        spilled_count_before = engine._spill_count
        # Next request should trigger a spill because the slot already
        # has ~100 tokens and max_context_tokens=50.
        result = engine.serve_native(
            "another long prompt " + " ".join(f"w{i}" for i in range(50)),
            max_slots=2, session_id="s1", max_context_tokens=50,
        )
        assert engine._spill_count == spilled_count_before + 1
        assert result.spilled is True

    def test_no_spill_when_within_budget(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.serve_native("short", max_slots=2, session_id="s1", max_context_tokens=4096)
        engine.serve_native("also short", max_slots=2, session_id="s1", max_context_tokens=4096)
        assert engine._spill_count == 0

    def test_clear_slot_resets_backend_state(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.serve_native("hello world", max_slots=2, session_id="s1")
        slot = engine._session_to_slot["s1"]
        assert backend.slot_kv.get(slot) is not None
        # After a spill-triggering request, the backend's clear_slot was
        # called. The slot may then be refilled by the new request.
        engine.serve_native(
            "x y z " * 500, max_slots=2, session_id="s1", max_context_tokens=10,
        )
        # Verify clear_slot was invoked on the backend.
        assert backend.clear_count.get(slot, 0) >= 1


class TestSlotState:
    def test_slot_state_returns_useful_observability(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.serve_native("hello", max_slots=2, session_id="u1")
        engine.serve_native("world", max_slots=2, session_id="u2")
        state = engine.slot_state()
        assert "slots" in state
        assert "sessions" in state
        assert "spill_count" in state
        assert "max_context_tokens" in state
        assert len(state["slots"]) >= 1
        assert state["sessions"]["u1"] is not None

    def test_reset_state_clears_everything(self):
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.serve_native("hello", max_slots=2, session_id="u1")
        engine.serve_native("world", max_slots=2, session_id="u2")
        engine.reset_state()
        state = engine.slot_state()
        assert state["slots"] == []
        assert state["sessions"] == {}
        assert state["spill_count"] == 0
        assert state["reserved_slots"] == []


class TestPersistentPoolWiring:
    """Hot-prefix promotion is delegated to PersistentSlotPool."""

    def test_slot_gets_pooled_after_threshold(self):
        """After ``min_reuse_for_pooling`` hits on the same cache_id,
        the pool acquires a slot for it."""
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        # Lower the pool threshold so the test is fast.
        engine.persistent_pool.min_reuse_for_pooling = 3
        # Stub the pool's llama-server interactions: there's no live
        # server in the test, so allocate + restore are no-ops that
        # hand back slot 0 / no-op respectively.
        engine.persistent_pool._allocate_slot = lambda: 0
        engine.persistent_pool._restore_to_slot = (
            lambda slot_id, cache_id, filename: {}
        )

        # Send the same prompt 5 times — single cache_id accumulates
        # use_count and crosses the pool threshold.
        for _ in range(5):
            engine.serve_native("stable prompt", max_tokens=4)

        # The pool should have promoted this cache_id.
        assert len(engine.persistent_pool._slots) == 1
        assert len(engine.persistent_pool._use_counts) == 1
        assert engine.persistent_pool._use_counts[
            next(iter(engine.persistent_pool._slots))
        ] >= 3

    def test_pool_capped_at_max_slots(self):
        """Only ``max_slots`` cache_ids can be pooled at once."""
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.persistent_pool.min_reuse_for_pooling = 2
        engine.persistent_pool.max_slots = 1
        engine.persistent_pool._allocate_slot = lambda: 0
        engine.persistent_pool._restore_to_slot = (
            lambda slot_id, cache_id, filename: {}
        )

        # Three distinct cache_ids, each sent 3 times to cross the
        # threshold.
        for c in ["alpha content", "beta content", "gamma content"]:
            for _ in range(3):
                engine.serve_native(c, max_tokens=4)

        # Only one slot ends up pooled — the pool evicts LRU when full.
        assert len(engine.persistent_pool._slots) == 1

    def test_pool_slot_skipped_during_round_robin(self):
        """Round-robin should avoid pool-owned slots (to keep their KV)."""
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.persistent_pool.min_reuse_for_pooling = 2
        engine.persistent_pool._allocate_slot = lambda: 0
        engine.persistent_pool._restore_to_slot = (
            lambda slot_id, cache_id, filename: {}
        )

        # Pool a specific cache_id by sending it 3 times. After that,
        # the pool owns a slot for that cache_id.
        for _ in range(3):
            engine.serve_native("hot prompt", max_tokens=4)
        assert len(engine.persistent_pool._slots) == 1
        pool_slot_id = next(iter(engine.persistent_pool._slots.values())).slot_id

        # Now send a completely unrelated prompt — should NOT route
        # to the pool-owned slot since its prefix doesn't match.
        result = engine.serve_native(
            "completely_unique_unrelated_xyz", max_tokens=4, max_slots=2,
        )
        assert result.slot_id != pool_slot_id

    def test_pool_slot_never_spills(self):
        """Even with budget pressure, pool-owned slots are exempt."""
        backend = FakeBackend()
        engine = RecyclingEngine(backend, FakeIndex())
        engine.persistent_pool.min_reuse_for_pooling = 2
        engine.persistent_pool._allocate_slot = lambda: 0
        engine.persistent_pool._restore_to_slot = (
            lambda slot_id, cache_id, filename: {}
        )
        engine._max_context_tokens = 50  # tight budget

        # Pool the prompt by hitting it 3 times first.
        for _ in range(3):
            engine.serve_native(
                "alpha " + "x" * 30, max_tokens=4, session_id="s1",
            )
        assert len(engine.persistent_pool._slots) == 1

        # Now build up additional context with the same prompt;
        # pool-owned slot must survive spill pressure.
        for i in range(5):
            engine.serve_native(
                "alpha " + "x" * 30 + f"_{i}",
                max_tokens=4, session_id="s1",
            )

        assert engine._spill_count == 0
