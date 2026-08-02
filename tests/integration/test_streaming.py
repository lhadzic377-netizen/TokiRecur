"""Integration tests for the engine's streaming chat path.

`serve_chat_stream` must deliver the exact same text as the blocking
`serve_chat` call, just streamed token-by-token. The stream==blocking
assertion requires deterministic slots, so these tests run against the
dedicated `precision_server` (see conftest.py) instead of the shared
session server.
"""

import pytest


@pytest.fixture
def engine(precision_engine):
    return precision_engine


class TestServeChatStream:
    async def _stream_text(self, engine, messages, **kw):
        parts = []
        result = None
        async for item in engine.serve_chat_stream(messages, **kw):
            if item["kind"] == "delta":
                parts.append(item["delta"])
            elif item["kind"] == "result":
                result = item["result"]
        return "".join(parts), result

    @pytest.mark.asyncio
    async def test_stream_equals_blocking_serve_chat(self, engine):
        ctk = {"enable_thinking": False}
        messages = [{"role": "user", "content": "Name a color."}]
        sync = engine.serve_chat(
            messages,
            max_tokens=32,
            temperature=0.0,
            max_slots=2,
            session_id="stream-test",
            chat_template_kwargs=ctk,
        )
        # Route to a different slot to force a fresh prefill, so the
        # streamed generation is conditioned on the same KV as the
        # blocking one (both cache_prompt=False effectively).
        engine.reset_state()
        text, result = await self._stream_text(
            engine,
            messages,
            max_tokens=32,
            temperature=0.0,
            max_slots=2,
            session_id="stream-test-2",
            chat_template_kwargs=ctk,
        )
        assert text == sync.output_text, (
            f"streamed text differs from blocking text:\n"
            f"  stream: {text!r}\n  blocking: {sync.output_text!r}"
        )
        assert result is not None
        assert result.generated_tokens >= 1
        assert result.slot_id is not None

    @pytest.mark.asyncio
    async def test_stream_yields_multiple_deltas(self, engine):
        """A multi-token answer must arrive as multiple deltas, not one
        blob — this is what the TUI renders live."""
        messages = [{"role": "user", "content": "List three animals."}]
        deltas = []
        async for item in engine.serve_chat_stream(
            messages,
            max_tokens=64,
            temperature=0.0,
            max_slots=2,
            chat_template_kwargs={"enable_thinking": False},
        ):
            if item["kind"] == "delta":
                deltas.append(item["delta"])
        text = "".join(deltas)
        assert text.strip(), "expected non-empty streamed text"
        assert len(deltas) >= 2, (
            f"expected >= 2 deltas (token-by-token), got {len(deltas)}: {deltas!r}"
        )

    @pytest.mark.asyncio
    async def test_stream_updates_slot_and_session_state(self, engine):
        """After a streamed request, the engine must have recorded the
        slot/session state exactly like the blocking path."""
        messages = [{"role": "user", "content": "Say hello."}]
        _, result = await self._stream_text(
            engine,
            messages,
            max_tokens=16,
            temperature=0.0,
            max_slots=2,
            session_id="state-test",
            chat_template_kwargs={"enable_thinking": False},
        )
        assert result is not None
        state = engine.slot_state()
        assert "state-test" in state["sessions"]
        slots = {s["slot_id"] for s in state["slots"]}
        assert result.slot_id in slots
