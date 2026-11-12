"""`toki diff` compares the cache, not two different questions.

The raw side used to be llama-server's *native* `/completion` endpoint
and the cached side the daemon's `/v1/chat/completions`. Those are
different requests in different formats — no chat template on one side, a
reasoning channel on the other — so:

- the "speedup" was measuring two different requests, and
- the outputs were reported as diverging at character zero whenever the
  model thought, because the raw side carried the thinking and the
  cached side did not.

Both sides are the same chat request now: one with the cache, one
without.
"""

from tokirecur.cli.completions import split_thinking


def test_thinking_is_split_out_of_the_answer():
    answer, reasoning = split_thinking(
        "<think>\nThe user wants a short reply.\n</think>\n\nDIFF-OK"
    )
    assert answer == "DIFF-OK"
    assert reasoning == "The user wants a short reply."


def test_a_plain_answer_has_no_reasoning():
    answer, reasoning = split_thinking("DIFF-OK")
    assert answer == "DIFF-OK"
    assert reasoning == ""


def test_thinking_with_no_answer_leaves_nothing_behind():
    answer, reasoning = split_thinking("<think>hmm</think>")
    assert answer == ""
    assert reasoning == "hmm"


def test_several_thinking_blocks_are_all_kept():
    answer, reasoning = split_thinking("<think>one</think>A<think>two</think>B")
    assert answer == "AB"
    assert "one" in reasoning and "two" in reasoning


def test_text_without_a_think_block_is_untouched():
    text = "a normal answer with <think> in it as a word"
    answer, reasoning = split_thinking(text)
    assert answer == text
    assert reasoning == ""


def test_raw_side_reads_token_counts_from_openai_usage(monkeypatch):
    """Regression: the raw side posts /v1/chat/completions but used to
    parse the native /completion fields (tokens_evaluated/tokens_predicted),
    which are absent there — so prefill_tokens was always 0 and the diff's
    'prefill tokens saved' headline was dead."""
    import tokirecur.cli.completions as comp

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 7},
            }

    monkeypatch.setattr(comp, "_auth_headers", lambda: {})
    monkeypatch.setattr(comp.httpx, "post", lambda *a, **k: FakeResp())
    answer, m = comp.completion_once("http://raw", "hi", api="native")
    assert answer == "hello"
    assert m["prefill_tokens"] == 123
    assert m["tokens_generated"] == 7
