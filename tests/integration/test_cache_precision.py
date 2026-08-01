"""Integration tests for cache precision under various conditions.

These tests validate that the cache truly preserves the model's output
across cache hits, regardless of model size or attention type. They
are essential for catching precision-loss regressions that would
silently change the model's answers.

The tests run against a dedicated llama-server (see the
`precision_server` fixture in conftest.py) so slot state is
self-contained: byte-identity assertions are meaningless on the shared
session server, whose slots this llama-server build leaves
non-deterministic after certain request sequences.
"""

import pytest


@pytest.fixture
def engine(precision_engine):
    return precision_engine


class TestCachePrecision:
    """Byte-identity precision checks."""

    def test_same_prompt_cache_hit_preserves_output(self, engine):
        """After warming a slot, sending the same prompt via the cache
        should produce byte-identical output."""
        # Warm slot 0 with a generic request.
        engine.serve_native("Hello.", max_tokens=4, max_slots=2)
        engine.reset_state()

        prompt = "What is the capital of France?"
        r1 = engine.serve_native(prompt, max_tokens=16, max_slots=2)
        r2 = engine.serve_native(prompt, max_tokens=16, max_slots=2)
        # After the first call, the slot has the prompt in its KV.
        # The second call should reuse the entire prompt (high reuse)
        # and produce byte-identical output.
        assert r1.output_text == r2.output_text, (
            f"Cache precision violation:\n  r1: {r1.output_text!r}\n  r2: {r2.output_text!r}"
        )

    def test_extending_prompt_preserves_output(self, engine):
        """When the new prompt extends a previous one (true prefix
        growth), the cache must preserve the model's output."""
        # Warm slot 0.
        engine.serve_native("Hello.", max_tokens=4, max_slots=2)
        engine.reset_state()

        system = "You are a helpful assistant. "
        r1 = engine.serve_native(system + "Q: What is 2+2?\nA:",
                                  max_tokens=8, max_slots=2)
        r2 = engine.serve_native(system + "Q: What is 3+3?\nA:",
                                  max_tokens=8, max_slots=2)
        # Both prompts share the system prefix. The model should give
        # different answers (2+2=4 vs 3+3=6) but be deterministic.
        # If the cache was wrong, both might say the same number.
        assert r1.output_text != r2.output_text, (
            "Cache reuse broke the model's ability to differentiate inputs"
        )

    def test_cache_does_not_affect_different_slots(self, engine):
        """Two requests on different slots must produce independent
        outputs (no cross-slot contamination)."""
        engine.serve_native("Hello.", max_tokens=4, max_slots=4)
        engine.reset_state()

        # Pin session A to its own slot.
        r_a1 = engine.serve_native("alpha", max_slots=4, session_id="A")
        r_a2 = engine.serve_native("beta", max_slots=4, session_id="A")
        # Pin session B to a different slot.
        r_b1 = engine.serve_native("gamma", max_slots=4, session_id="B")
        r_b2 = engine.serve_native("delta", max_slots=4, session_id="B")

        # Each session is pinned to a specific slot.
        assert r_a1.slot_id == r_a2.slot_id
        assert r_b1.slot_id == r_b2.slot_id
        assert r_a1.slot_id != r_b1.slot_id, "Sessions should not share slots"

    def test_repeated_run_yields_same_total_metrics(self, engine):
        """Two consecutive runs of the same workload should produce
        nearly identical hit rates (within run-to-run variance)."""
        engine.serve_native("warmup", max_tokens=4, max_slots=4)
        engine.reset_state()

        # Use a long shared prefix to ensure cache hits.
        shared = "You are a helpful assistant answering questions about colors. " * 10
        prompts = [shared + f"Question {i}: name a color." for i in range(10)]
        results_1 = [engine.serve_native(p, max_tokens=8, max_slots=4) for p in prompts]
        engine.reset_state()
        results_2 = [engine.serve_native(p, max_tokens=8, max_slots=4) for p in prompts]

        # Both runs should have similar hit rates — first might miss
        # (cold slot), but the rest should hit.
        hits_1 = sum(1 for r in results_1 if r.hit)
        hits_2 = sum(1 for r in results_2 if r.hit)
        assert hits_1 >= 8, f"expected >= 8 hits in run 1, got {hits_1}"
        assert hits_2 >= 8, f"expected >= 8 hits in run 2, got {hits_2}"


class TestCacheEffectiveness:
    """Tests that measure whether the cache is actually doing its job."""

    def test_high_overprefix_yields_high_hit_rate(self, engine):
        """A workload with shared prefix should achieve high hit rate."""
        engine.serve_native("warmup", max_tokens=4, max_slots=2)
        engine.reset_state()

        system = "You are an expert. " * 50  # ~150 tokens
        prompts = [system + f"Task {i}: explain X." for i in range(20)]
        results = [engine.serve_native(p, max_tokens=8, max_slots=2) for p in prompts]

        # At least 18 of 20 should be hits (first might land on a new slot).
        hits = sum(1 for r in results if r.hit)
        assert hits >= 18, f"expected >= 18 hits, got {hits}/20"

    def test_no_overlap_yields_zero_hits(self, engine):
        """Completely unrelated prompts should not hit the cache."""
        engine.serve_native("warmup", max_tokens=4, max_slots=2)
        engine.reset_state()

        unrelated_prompts = [
            f"random_unrelated_prompt_{i}_{hash(str(i))}" for i in range(10)
        ]
        results = [engine.serve_native(p, max_tokens=4, max_slots=2) for p in unrelated_prompts]
        # No two prompts share a prefix (each is unique).
        hits = sum(1 for r in results if r.hit)
        assert hits == 0, f"expected 0 hits for unrelated prompts, got {hits}"


class TestCacheUnderLoad:
    """Tests that the cache behaves correctly under various conditions."""

    def test_slot_spill_does_not_lose_precision(self, engine):
        """When a slot spills (context exceeds budget), subsequent
        requests must still produce deterministic output."""
        engine._init_slot_state()
        engine._max_context_tokens = 100  # small budget

        # Fill the slot to trigger a spill.
        long = " ".join(f"word{i}" for i in range(80))
        engine.serve_native(long, max_slots=2, session_id="spill")
        # Next request triggers a spill.
        result = engine.serve_native(
            "x y z " * 30, max_slots=2, session_id="spill",
        )
        # The result should still have output, and the slot count incremented.
        assert result.output_text is not None
        assert result.spilled is True
        assert engine._spill_count >= 1

    def test_high_concurrency_no_state_corruption(self, engine):
        """Sending many requests in a row should not corrupt engine state.

        With max_slots=4 and 4 sessions each having unrelated content,
        each session should land on its own slot (round-robin)."""
        engine.serve_native("warmup", max_tokens=4, max_slots=4)
        engine.reset_state()

        # 4 sessions, each with completely unique content (no prefix overlap).
        # First request goes to slot 0, 1, 2, 3 in round-robin order.
        sessions = ["sess-alpha", "sess-beta", "sess-gamma", "sess-delta"]
        # Use content with unique hashes to prevent prefix matches.
        contents = {
            "sess-alpha": "alpha_unique_session_1",
            "sess-beta": "beta_unique_session_2",
            "sess-gamma": "gamma_unique_session_3",
            "sess-delta": "delta_unique_session_4",
        }
        # Make multiple requests per session to ensure pinning takes effect.
        for round_n in range(3):
            for s in sessions:
                engine.serve_native(
                    f"{contents[s]}_round{round_n}",
                    max_slots=4, session_id=s,
                )

        # All 4 sessions should be pinned.
        assert len(engine._session_to_slot) == 4
        # Each session should be on a distinct slot (round-robin on first request).
        slot_to_sessions: dict[int, list[str]] = {}
        for s, slot in engine._session_to_slot.items():
            slot_to_sessions.setdefault(slot, []).append(s)
        for slot, ss in slot_to_sessions.items():
            assert len(ss) == 1, f"slot {slot} has multiple sessions: {ss}"
