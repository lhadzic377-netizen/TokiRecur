"""Integration tests for the RecyclingEngine.serve_native end-to-end."""



class TestEngineBasic:
    def test_first_request_returns_result(self, engine):
        r = engine.serve_native(
            "Hello, how are you?",
            max_tokens=16,
            temperature=0.0,
            max_slots=2,
        )
        assert r.slot_id in (0, 1)
        assert r.reuse_depth == 0
        assert r.hit is False
        assert r.output_text
        assert r.ttft_ms > 0
        assert r.total_ms > 0

    def test_repeated_prompt_yields_higher_reuse(self, engine):
        prefix = (
            "You are a careful assistant. Today's date is 2026-07-28. "
            "Please answer concisely. " * 5
        )
        engine.serve_native(
            prefix + "First question: name a primary color.",
            max_tokens=8, max_slots=2,
        )
        r2 = engine.serve_native(
            prefix + "Second question: name another primary color.",
            max_tokens=8, max_slots=2,
        )
        # Should have meaningful reuse because the prompts share the
        # long system prompt prefix.
        assert r2.reuse_depth > 0
        assert r2.skipped_tokens == r2.reuse_depth


class TestEngineSessionRouting:
    def test_session_id_pins_slot(self, engine):
        engine.serve_native("alpha", max_slots=4, session_id="s1")
        engine.serve_native("beta", max_slots=4, session_id="s2")
        engine.serve_native("gamma", max_slots=4, session_id="s1")
        # Both sessions should have pinned slots.
        assert "s1" in engine._session_to_slot
        assert "s2" in engine._session_to_slot
        # Each session should consistently land on its pinned slot.
        r = engine.serve_native("delta", max_slots=4, session_id="s1")
        assert r.slot_id == engine._session_to_slot["s1"]


class TestEngineSpilling:
    def test_slot_spills_when_context_grows(self, engine):
        engine._init_slot_state()
        # Force a small context budget.
        engine._max_context_tokens = 50
        # Build up via a session.
        long = " ".join(f"word{i}" for i in range(100))
        engine.serve_native(long, max_slots=2, session_id="spill-test")
        spills_before = engine._spill_count
        # Trigger another big request on the same session.
        engine.serve_native(
            " ".join(f"w{i}" for i in range(50)),
            max_slots=2, session_id="spill-test",
        )
        assert engine._spill_count > spills_before


class TestEngineSlotState:
    def test_slot_state_endpoint_data(self, engine):
        engine.serve_native("alpha beta gamma", max_slots=2, session_id="u1")
        state = engine.slot_state()
        assert "slots" in state
        assert "sessions" in state
        assert state["sessions"]["u1"] is not None
        assert len(state["slots"]) >= 1
        slot0 = state["slots"][0]
        assert "slot_id" in slot0
        assert "context_tokens" in slot0
        assert "use_count" in slot0

    def test_reset_state_clears_all(self, engine):
        engine.serve_native("alpha", max_slots=2, session_id="u1")
        engine.serve_native("beta", max_slots=2, session_id="u2")
        engine.reset_state()
        state = engine.slot_state()
        assert state["slots"] == []
        assert state["sessions"] == {}
        assert state["spill_count"] == 0


class TestEnginePrecision:
    def test_same_prompt_twice_yields_identical_output(self, engine):
        """The cache must preserve precision — same prompt twice on the
        same slot must produce byte-identical output.

        Note: the very first request on a slot may trigger Qwen3.5-9B's
        'thinking mode' (different output). We warm the slot first with
        a throwaway request so subsequent runs are deterministic."""
        # Warm the slot with a generic request.
        engine.serve_native(
            "Hello.", max_tokens=4, max_slots=2,
        )
        engine.reset_state()
        # Now run the actual test prompt twice on the warmed slot.
        prompt = "List three primary colors in order."
        r1 = engine.serve_native(prompt, max_tokens=16, max_slots=2)
        r2 = engine.serve_native(prompt, max_tokens=16, max_slots=2)
        assert r1.output_text == r2.output_text, (
            f"Cache precision violation: same prompt produced different "
            f"outputs.\n  r1: {r1.output_text!r}\n  r2: {r2.output_text!r}"
        )
