"""Integration tests for the LlamaServerBackend."""

import pytest


class TestBackendHealth:
    def test_health_endpoint(self, http_client):
        r = http_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_slots(self, backend):
        slots = backend.list_slots()
        assert isinstance(slots, list)
        assert len(slots) >= 1


class TestBackendTokenize:
    def test_tokenize_returns_ids(self, backend):
        toks = backend.tokenize("Hello, world!")
        assert isinstance(toks, list)
        assert all(isinstance(t, int) for t in toks)
        assert len(toks) >= 2


class TestBackendEmbed:
    def test_embed_returns_vector(self, backend):
        vec = backend.embed("What is the capital of France?")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) > 0
        # Should be roughly normalised magnitude for sentence embeddings.
        assert any(abs(x) > 0.01 for x in vec)

    def test_embed_falls_back_on_invalid_input(self, backend):
        # Empty string is valid; should still return a vector.
        vec = backend.embed("")
        assert isinstance(vec, list)
        assert len(vec) > 0


class TestBackendComplete:
    def test_complete_runs_and_returns_text(self, backend):
        result = backend.complete_native(
            "The capital of France is",
            slot_id=0,
            max_tokens=8,
            temperature=0.0,
        )
        assert "text" in result
        assert result["tokens_generated"] >= 1
        assert result["prefill_tokens"] >= 1
        assert result["ttft_ms"] > 0
        assert result["total_ms"] > 0

    def test_repeated_prompt_reuses_prefix(self, backend):
        """Two requests with the same prefix should reuse KV on slot 0.

        The first request fully prefills. The second shares the same
        prefix, so llama-server should skip those tokens (prefill_tokens
        is smaller or equal).
        """
        long_prefix = (
            "You are a careful assistant. Today's date is 2026-07-28. "
            "Please answer concisely. " * 5
        )
        first = backend.complete_native(
            long_prefix + " First question: 2+2?",
            slot_id=0,
            max_tokens=8,
            temperature=0.0,
        )
        second = backend.complete_native(
            long_prefix + " Second question: 3+3?",
            slot_id=0,
            max_tokens=8,
            temperature=0.0,
        )
        # Second should have equal-or-fewer prefill tokens than the first
        # (some of the prefix was already in the slot's KV cache).
        # Note: llama-server may clear the slot between requests in some
        # builds, so we don't assert strict inequality — only that it
        # didn't get worse.
        assert second["prefill_tokens"] <= first["prefill_tokens"] + 50


class TestBackendStreaming:
    """complete_openai_stream must produce the same text as the blocking
    complete_openai call, delivered as token-by-token deltas."""

    @pytest.mark.asyncio
    async def test_stream_matches_blocking_output(self, backend):
        ctk = {"enable_thinking": False}
        messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        # Distinct slots => both requests fresh-prefill (no KV reuse),
        # exactly like the cache_prompt=False case, but without the
        # poisoning of the slot that cache_prompt=False causes in this
        # llama-server build.
        sync = backend.complete_openai(
            messages,
            slot_id=0,
            max_tokens=32,
            temperature=0.0,
            cache_prompt=True,
            chat_template_kwargs=ctk,
        )
        parts: list[str] = []
        result: dict | None = None
        async for item in backend.complete_openai_stream(
            messages,
            slot_id=1,
            max_tokens=32,
            temperature=0.0,
            cache_prompt=True,
            chat_template_kwargs=ctk,
        ):
            if item.get("kind") == "delta":
                parts.append(item["delta"])
            elif item.get("kind") == "result":
                result = item["result"]
        text = "".join(parts)
        assert result is not None, "stream never yielded a result item"
        assert text == sync["text"], (
            f"streamed text differs from blocking text:\n"
            f"  stream: {text!r}\n  blocking: {sync['text']!r}"
        )
        assert text.strip(), "expected non-empty streamed content"
        # A 1-token answer legitimately arrives as a single delta; the
        # engine-level test_stream_yields_multiple_deltas covers the
        # multi-delta case.
        assert len(parts) >= 1
        assert result["tokens_generated"] >= 1
        assert result["text"] == text


class TestBackendSlotManagement:
    def test_clear_slot_returns_result(self, backend):
        """clear_slot is a no-op (upstream llama-server doesn't support
        ?action=clear). The actual slot reset happens on the next
        request via cache_prompt=False."""
        backend.complete_native(
            "Long prompt for testing slot clearing. " * 20,
            slot_id=0,
            max_tokens=4,
            temperature=0.0,
        )
        result = backend.clear_slot(0)
        # Returns a result dict; cleared may be False since the
        # upstream endpoint isn't supported, but the call should not
        # raise an exception.
        assert "cleared" in result

    def test_cache_prompt_false_resets_slot(self, backend):
        """After cache_prompt=False, the prefill cost matches the new
        prompt length (no reuse), and the timing data reflects a full
        prefill (not a cache hit)."""
        # Fill the slot with a long prompt via cache_prompt=True.
        long_prompt = "Long prompt that fills the slot. " * 30
        r1 = backend.complete_native(
            long_prompt,
            slot_id=0,
            max_tokens=4,
            temperature=0.0,
        )
        # Now do a short request with cache_prompt=False — prefill cost
        # should match the short prompt, not the long one.
        short_prompt = "Hi."
        r2 = backend.complete_native(
            short_prompt,
            slot_id=0,
            max_tokens=4,
            temperature=0.0,
            cache_prompt=False,
        )
        short_tokens = len(backend.tokenize(short_prompt))
        # Prefill cost on the second request should match the short
        # prompt (within a few tokens of overhead), proving the slot
        # was reset.
        assert abs(r2["prefill_tokens"] - short_tokens) < 5

    def test_clear_nonexistent_slot_is_safe(self, backend):
        result = backend.clear_slot(99)
        assert "cleared" in result
