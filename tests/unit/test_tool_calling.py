"""Unit tests for tool-calling support in the daemon.

Covers: the OpenAI message/request models (tool_calls / tool_call_id /
tools / tool_choice), the engine's relay of tool-call stream items, and
the daemon's OpenAI wire format in both streaming and non-streaming
chat completions (stubbed backend — no live llama-server).
"""

import json
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur.api.app import ChatCompletionRequest, ChatMessage, create_app
from tokirecur.backend import _sanitize_tools, _strip_patterns
from tokirecur.core.recycling import RecyclingEngine

# ── Fakes ──────────────────────────────────────────────────────────────


@dataclass
class FakeBackend:
    """Minimal LlamaBackend stub: tokenizes by word, embeds to a
    4-vector, and records every chat payload it is asked to send."""

    stream_chunks: list = field(default_factory=list)
    non_stream_result: dict = field(default_factory=dict)
    chat_payloads: list = field(default_factory=list)
    tokenize_texts: list = field(default_factory=list)

    def tokenize(self, text: str) -> list[int]:
        self.tokenize_texts.append(text)
        return [hash(w) % 100000 for w in text.split()]

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def _record(self, **kw) -> None:
        self.chat_payloads.append(
            {k: kw[k] for k in ("messages", "tools", "tool_choice") if k in kw}
        )

    async def complete_openai_stream(self, messages, **kw):
        self._record(messages=messages, **{k: kw[k] for k in ("tools", "tool_choice")})
        for item in self.stream_chunks:
            yield item

    def complete_openai(self, messages, **kw):
        self._record(messages=messages, **{k: kw[k] for k in ("tools", "tool_choice")})
        return dict(self.non_stream_result)


class FakeIndex:
    """No-op index."""

    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def search(self, embedding):
        return None

    def insert(self, entry) -> None:
        self.entries[entry.cache_id] = entry

    def touch(self, cache_id: str) -> None:
        pass

    def stats(self) -> dict:
        return {"entries": len(self.entries), "bytes_on_disk": 0, "embedding_dim": 4}


def tool_sse_chunks(text_chunks, tool_deltas, usage=None):
    """llama-server-style SSE objects: text deltas, tool-call deltas,
    a finish chunk and a usage chunk."""
    chunks = []
    chunks.append({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    for c in text_chunks:
        chunks.append({"choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})
    for tc in tool_deltas:
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": tc}, "finish_reason": None}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    chunks.append({"choices": [], "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3}})
    return chunks


def sse_obj(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def parse_sse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                out.append(json.loads(data))
    return out


# ── Request models ─────────────────────────────────────────────────────


class TestRequestModels:
    def test_tool_messages_round_trip(self):
        msgs = [
            ChatMessage(role="assistant", content=None, tool_calls=[{"id": "c1", "type": "function"}]),
            ChatMessage(role="tool", content='{"ok": true}', tool_call_id="c1", name="bash"),
        ]
        dumped = [m.model_dump(exclude_none=True) for m in msgs]
        assert dumped[0] == {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function"}],
        }
        assert dumped[1] == {
            "role": "tool", "content": '{"ok": true}',
            "tool_call_id": "c1", "name": "bash",
        }

    def test_request_accepts_tools_and_tool_choice(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            tool_choice="auto",
        )
        assert req.tools[0]["function"]["name"] == "bash"
        assert req.tool_choice == "auto"

    def test_request_accepts_kimi_context_sized_output_budget(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=131072,
        )
        assert req.max_tokens == 131072


# ── Engine relay ───────────────────────────────────────────────────────


class TestEngineToolRelay:
    def _engine(self, backend):
        return RecyclingEngine(backend, FakeIndex())

    async def test_stream_relays_tool_calls_and_finish(self):
        backend = FakeBackend(
            stream_chunks=[
                {"kind": "delta", "delta": "thinking"},
                {"kind": "tool_calls", "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "bash"}}]},
                {"kind": "tool_calls", "tool_calls": [{"index": 0, "function": {"arguments": "ls"}}]},
                {"kind": "finish", "finish_reason": "tool_calls"},
                {"kind": "result", "result": {
                    "text": "", "tool_calls": [
                        {"index": 0, "id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": "ls"}},
                    ],
                    "finish_reason": "tool_calls",
                    "tokens_generated": 3, "prefill_tokens": 5,
                    "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
                }},
            ]
        )
        engine = self._engine(backend)
        kinds = []
        final = None
        async for item in engine.serve_chat_stream(
            [{"role": "user", "content": "list files"}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
            tool_choice="auto",
        ):
            kinds.append(item["kind"])
            if item["kind"] == "result":
                final = item["result"]
        assert kinds == ["delta", "tool_calls", "tool_calls", "finish", "result"]
        assert final.tool_calls[0]["function"]["arguments"] == "ls"
        assert final.finish_reason == "tool_calls"
        sent = backend.chat_payloads[0]
        assert sent["tools"][0]["function"]["name"] == "bash"
        assert sent["tool_choice"] == "auto"

    def test_non_stream_carries_tool_calls(self):
        backend = FakeBackend(
            non_stream_result={
                "text": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": "ls"}},
                ],
                "finish_reason": "tool_calls",
                "tokens_generated": 3, "prefill_tokens": 5,
                "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
            }
        )
        engine = self._engine(backend)
        result = engine.serve_chat(
            [{"role": "user", "content": "list files"}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
        )
        assert result.tool_calls[0]["function"]["arguments"] == "ls"
        assert result.finish_reason == "tool_calls"

    def test_multi_part_content_routes_as_string(self):
        """Agentic clients (opencode) send user content as a list of
        parts; routing must normalize it to a string so backend.tokenize
        never sees a non-string (llama-server 500s otherwise)."""
        backend = FakeBackend(
            non_stream_result={
                "text": "", "tokens_generated": 3, "prefill_tokens": 5,
                "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
            }
        )
        engine = self._engine(backend)
        result = engine.serve_chat(
            [{"role": "user", "content": [
                {"type": "text", "text": "list "},
                {"type": "text", "text": "files"},
            ]}]
        )
        assert result.output_text == ""
        assert backend.tokenize_texts and backend.tokenize_texts[0] == "list files"

    def test_last_message_fallback_with_non_text_parts(self):
        backend = FakeBackend(
            non_stream_result={
                "text": "", "tokens_generated": 3, "prefill_tokens": 5,
                "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
            }
        )
        engine = self._engine(backend)
        engine.serve_chat([{"role": "user", "content": [
            {"type": "file", "url": "x"},
            {"type": "text", "text": "summarize"},
        ]}])
        assert backend.tokenize_texts[0] == "summarize"


# ── Daemon wire format ─────────────────────────────────────────────────


def _make_daemon(backend):
    return create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir="/tmp/toki-tests",
        enable_auth=False,
        enable_rate_limit=False,
    )


class TestDaemonToolCompletions:
    def test_stream_relays_tool_calls_deltas_and_finish(self):
        backend = FakeBackend(
            stream_chunks=[
                {"kind": "delta", "delta": ""},
                {"kind": "tool_calls", "tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": ""}},
                ]},
                {"kind": "tool_calls", "tool_calls": [
                    {"index": 0, "function": {"arguments": "ls -la"}},
                ]},
                {"kind": "finish", "finish_reason": "tool_calls"},
                {"kind": "result", "result": {
                    "text": "", "tool_calls": [
                        {"index": 0, "id": "call_1", "type": "function",
                         "function": {"name": "bash", "arguments": "ls -la"}},
                    ],
                    "finish_reason": "tool_calls",
                    "tokens_generated": 3, "prefill_tokens": 5,
                    "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
                }},
            ]
        )
        with TestClient(_make_daemon(backend)) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "list files"}],
                    "tools": [{"type": "function", "function": {"name": "bash"}}],
                    "tool_choice": "auto",
                    "stream": True,
                    "max_tokens": 64,
                },
            )
        assert r.status_code == 200
        chunks = parse_sse(r.text)
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert tool_chunks, f"no tool_calls deltas in: {r.text[:500]}"
        args = "".join(
            tc["function"]["arguments"]
            for c in tool_chunks
            for tc in c["choices"][0]["delta"]["tool_calls"]
        )
        assert args == "ls -la"
        final = [c for c in chunks if c["choices"][0]["finish_reason"]]
        assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_non_stream_returns_tool_calls_in_message(self):
        backend = FakeBackend(
            non_stream_result={
                "text": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": "ls"}},
                ],
                "finish_reason": "tool_calls",
                "tokens_generated": 3, "prefill_tokens": 5,
                "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
            }
        )
        with TestClient(_make_daemon(backend)) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "list files"}],
                    "tools": [{"type": "function", "function": {"name": "bash"}}],
                },
            )
        assert r.status_code == 200
        body = r.json()
        msg = body["choices"][0]["message"]
        assert msg["content"] is None
        assert msg["tool_calls"][0]["function"]["arguments"] == "ls"
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_text_stream_without_tools_unchanged(self):
        backend = FakeBackend(
            stream_chunks=[
                {"kind": "delta", "delta": "Hello"},
                {"kind": "result", "result": {
                    "text": "Hello", "tool_calls": None, "finish_reason": "stop",
                    "tokens_generated": 1, "prefill_tokens": 5,
                    "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
                }},
            ]
        )
        with TestClient(_make_daemon(backend)) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
        assert r.status_code == 200
        chunks = parse_sse(r.text)
        text = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
        )
        assert text == "Hello"
        final = [c for c in chunks if c["choices"][0]["finish_reason"]]
        assert final[-1]["choices"][0]["finish_reason"] == "stop"


# ── Schema pattern stripping ───────────────────────────────────────────


def test_patterns_stripped():
    schema = {"type": "string", "pattern": "[A-Za-z0-9_ -]{1,64}"}
    assert _strip_patterns(schema) == {"type": "string"}


def test_nested_schema_patterns_stripped():
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string", "pattern": "[a-z]+"}},
            "code": {"pattern": "\\S"},
        },
    }
    out = _strip_patterns(schema)
    assert out["properties"]["items"]["items"] == {"type": "string"}
    assert out["properties"]["code"] == {}


def test_property_named_pattern_preserved():
    schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }
    out = _strip_patterns(schema)
    assert out["properties"]["pattern"] == {"type": "string"}
    assert out["required"] == ["pattern"]


def test_sanitize_tools_strips_patterns_and_defaults_parameters():
    tools = [
        {"type": "function", "function": {"name": "no_params"}},
        {
            "type": "function",
            "function": {
                "name": "with_params",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "pattern": "^(?:\\S)$"}},
                },
            },
        },
    ]
    out = _sanitize_tools(tools)
    assert out[0]["function"]["parameters"] == {}
    assert out[1]["function"]["parameters"]["properties"]["x"] == {"type": "string"}


def test_sanitize_tools_preserves_non_tool_entries():
    tools = [{"not": "a tool"}, {"function": "bare"}]
    out = _sanitize_tools(tools)
    assert out[0] == {"not": "a tool"}
    assert out[1]["function"] == "bare"
