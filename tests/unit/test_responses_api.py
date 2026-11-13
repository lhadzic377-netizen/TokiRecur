"""Unit tests for the OpenAI Responses API shim (POST /v1/responses).

Covers: request → chat message mapping (string input, message items,
function_call_output / function_call items, instructions), the
non-streaming response body, the SSE event stream (text + reasoning +
tool calls), and error handling — all with a stubbed backend.
"""

import json
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur.api.app import ResponsesRequest, create_app

# ── Fakes (mirror test_tool_calling.py) ────────────────────────────────


@dataclass
class FakeBackend:
    stream_chunks: list = field(default_factory=list)
    non_stream_result: dict = field(default_factory=dict)
    chat_payloads: list = field(default_factory=list)
    last_tools: list | None = None
    last_tool_choice: object = None

    def tokenize(self, text: str) -> list[int]:
        return [hash(w) % 100000 for w in text.split()]

    def embed(self, text: str) -> list[float]:
        return [0.0] * 4

    def _record(self, messages, **kw) -> None:
        self.chat_payloads.append(messages)
        self.last_tools = kw.get("tools")
        self.last_tool_choice = kw.get("tool_choice")

    async def complete_openai_stream(self, messages, **kw):
        self._record(messages, **kw)
        for item in self.stream_chunks:
            yield item

    def complete_openai(self, messages, **kw):
        self._record(messages, **kw)
        return dict(self.non_stream_result)


class FakeIndex:
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


def _base_result(text="hi there", tool_calls=None, finish_reason="stop",
                 reasoning=""):
    return {
        "text": text, "tool_calls": tool_calls,
        "finish_reason": finish_reason, "reasoning": reasoning,
        "tokens_generated": 3, "prefill_tokens": 5,
        "ttft_ms": 1.0, "total_ms": 2.0, "prefill_tokens_per_sec": 0.0,
    }


def sse_pairs(text: str) -> list[tuple[str, dict]]:
    """Parse `event:`/`data:` SSE lines into (event, payload) pairs."""
    out = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data = line[6:].strip()
        if data and data != "[DONE]":
            out.append((event, json.loads(data)))
    return out


def _make_daemon(backend):
    return create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir="/tmp/toki-tests",
        enable_auth=False,
        enable_rate_limit=False,
    )


# ── Request model / mapping ────────────────────────────────────────────


class TestRequestModel:
    def test_accepts_codex_style_payload(self):
        req = ResponsesRequest.model_validate({
            "model": "qwen3.5-9b",
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}],
            "instructions": "Be terse.",
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "name": "bash",
                       "description": "Run a command",
                       "parameters": {"type": "object"}}],
            "tool_choice": "auto",
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 256,
            "previous_response_id": "resp_123",
        })
        assert req.instructions == "Be terse."
        assert req.stream
        assert req.reasoning.effort == "medium"

    def test_string_input(self):
        req = ResponsesRequest(model="m", input="just text")
        assert req.input == "just text"


class TestMessagesMapping:
    def test_string_input_becomes_user_message(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={"model": "m", "input": "hello"})
        assert backend.chat_payloads[0] == [{"role": "user", "content": "hello"}]

    def test_instructions_prepended_as_system(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m", "input": "hello",
                "instructions": "you are a shell",
            })
        assert backend.chat_payloads[0] == [
            {"role": "system", "content": "you are a shell"},
            {"role": "user", "content": "hello"},
        ]

    def test_developer_message_folds_into_system(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m",
                "input": [
                    {"type": "message", "role": "developer",
                     "content": [{"type": "input_text",
                                  "text": "sandbox rules"}]},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]},
                ],
                "instructions": "You are a coding agent.",
            })
        assert backend.chat_payloads[0] == [
            {"role": "system",
             "content": "You are a coding agent.\n\nsandbox rules"},
            {"role": "user", "content": "hi"},
        ]

    def test_developer_message_first_becomes_system(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m",
                "input": [
                    {"type": "message", "role": "developer",
                     "content": [{"type": "input_text", "text": "rules"}]},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "hi"}]},
                ],
            })
        assert backend.chat_payloads[0] == [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hi"},
        ]

    def test_items_map_with_tool_round_trip(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m",
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "list "},
                                 {"type": "input_text", "text": "files"}]},
                    {"type": "function_call", "call_id": "c1", "name": "bash",
                     "arguments": "ls"},
                    {"type": "function_call_output", "call_id": "c1",
                     "output": '{"ok": true}'},
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "now what"}]},
                ],
            })
        messages = backend.chat_payloads[0]
        assert messages == [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "bash", "arguments": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
            {"role": "user", "content": "now what"},
        ]

    def test_responses_style_tools_rewrapped_for_chat(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m", "input": "list files",
                "tools": [{"type": "function", "name": "exec_command",
                           "description": "Run a command",
                           "parameters": {"type": "object"},
                           "strict": False}],
                "tool_choice": {"type": "function", "name": "exec_command"},
            })
        assert backend.last_tools == [{
            "type": "function", "function": {
                "name": "exec_command", "description": "Run a command",
                "parameters": {"type": "object"}, "strict": False,
            },
        }]
        assert backend.last_tool_choice == {
            "type": "function", "function": {"name": "exec_command"},
        }

    def test_namespace_and_non_function_tools_handled(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            client.post("/v1/responses", json={
                "model": "m", "input": "list files",
                "tools": [
                    {"type": "function", "name": "exec_command",
                     "parameters": {}},
                    {"type": "namespace", "name": "multi_agent_v1",
                     "tools": [{"type": "function", "name": "spawn_agent",
                                "parameters": {}}]},
                    {"type": "web_search", "external_web_access": False},
                ],
            })
        assert backend.last_tools == [
            {"type": "function",
             "function": {"name": "exec_command", "parameters": {}}},
            {"type": "function",
             "function": {"name": "spawn_agent", "parameters": {}}},
        ]


# ── Non-streaming ──────────────────────────────────────────────────────


class TestResponsesNonStream:
    def test_text_response_shape(self):
        backend = FakeBackend(non_stream_result=_base_result(text="hi there"))
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "qwen3.5-9b", "input": "hi",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "response"
        assert body["id"].startswith("resp-")
        assert body["status"] == "completed"
        assert body["model"] == "qwen3.5-9b"
        assert body["error"] is None
        assert body["output_text"] == "hi there"
        (msg,) = body["output"]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"] == [
            {"type": "output_text", "text": "hi there", "annotations": []},
        ]
        assert body["usage"]["input_tokens"] == 5
        assert body["usage"]["output_tokens"] == 3
        assert "cache" in body
        assert "hit" in body["cache"]

    def test_tool_mode_returns_function_call_items(self):
        backend = FakeBackend(non_stream_result=_base_result(
            text="",
            tool_calls=[
                {"id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": "ls -la"}},
            ],
            finish_reason="tool_calls",
        ))
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "list files",
                "tools": [{"type": "function", "name": "bash"}],
            })
        body = r.json()
        (fc,) = body["output"]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "call_1"
        assert fc["name"] == "bash"
        assert fc["arguments"] == "ls -la"
        assert fc["status"] == "completed"
        assert body["output_text"] == ""

    def test_reasoning_becomes_reasoning_item(self):
        backend = FakeBackend(non_stream_result=_base_result(
            text="answer", reasoning="think think",
        ))
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "q",
            })
        body = r.json()
        (reasoning, msg) = body["output"]
        assert reasoning["type"] == "reasoning"
        assert reasoning["summary"] == [
            {"type": "summary_text", "text": "think think"},
        ]
        assert msg["content"][0]["text"] == "answer"


# ── Streaming ──────────────────────────────────────────────────────────


class TestResponsesStream:
    def test_text_stream_event_sequence(self):
        backend = FakeBackend(stream_chunks=[
            {"kind": "delta", "delta": "hi "},
            {"kind": "delta", "delta": "there"},
            {"kind": "finish", "finish_reason": "stop"},
            {"kind": "result", "result": _base_result(text="hi there")},
        ])
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "hi", "stream": True,
            })
        assert r.status_code == 200
        events = sse_pairs(r.text)
        names = [e for e, _ in events]
        assert names == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert events[0][1]["status"] == "in_progress"
        for name, payload in events:
            assert payload.get("type") == name, f"{name}: {payload}"
        deltas = [p["delta"] for e, p in events
                  if e == "response.output_text.delta"]
        assert "".join(deltas) == "hi there"
        done = [p for e, p in events if e == "response.output_text.done"]
        assert done[0]["text"] == "hi there"
        completed = events[-1][1]
        assert completed["response"]["status"] == "completed"
        assert completed["response"]["usage"]["total_tokens"] == 8
        assert completed["response"]["cache"]["hit"] is False
        (msg,) = completed["response"]["output"]
        assert msg["type"] == "message"

    def test_tool_stream_emits_function_call_events(self):
        backend = FakeBackend(stream_chunks=[
            {"kind": "tool_calls", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": ""}},
            ]},
            {"kind": "tool_calls", "tool_calls": [
                {"index": 0, "function": {"arguments": "ls -la"}},
            ]},
            {"kind": "finish", "finish_reason": "tool_calls"},
            {"kind": "result", "result": _base_result(
                text="",
                tool_calls=[
                    {"id": "call_1", "type": "function",
                     "function": {"name": "bash", "arguments": "ls -la"}},
                ],
                finish_reason="tool_calls",
            )},
        ])
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "list files", "stream": True,
                "tools": [{"type": "function", "name": "bash"}],
            })
        events = sse_pairs(r.text)
        names = [e for e, _ in events]
        assert "response.function_call_arguments.delta" in names
        assert "response.function_call_arguments.done" in names
        args = "".join(p["delta"] for e, p in events
                       if e == "response.function_call_arguments.delta")
        assert args == "ls -la"
        (fc_item,) = [p["item"] for e, p in events
                      if e == "response.output_item.done"
                      and p["item"]["type"] == "function_call"]
        assert fc_item["name"] == "bash"
        assert fc_item["arguments"] == "ls -la"
        completed = [p for e, p in events if e == "response.completed"][0]
        (fc,) = completed["response"]["output"]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "call_1"

    def test_reasoning_deltas_relayed(self):
        backend = FakeBackend(stream_chunks=[
            {"kind": "reasoning", "delta": "hmm"},
            {"kind": "delta", "delta": "answer"},
            {"kind": "result", "result": _base_result(
                text="answer", reasoning="hmm"),
            },
        ])
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "q", "stream": True,
            })
        events = sse_pairs(r.text)
        summaries = [p["summary"][0]["text"] for e, p in events
                     if e == "response.reasoning_summary_text.delta"]
        assert summaries == ["hmm"]

    def test_engine_error_yields_failed_event(self):
        class ExplodingBackend(FakeBackend):
            async def complete_openai_stream(self, messages, **kw):
                raise RuntimeError("backend exploded")
                yield  # pragma: no cover

        with TestClient(_make_daemon(ExplodingBackend())) as client:
            r = client.post("/v1/responses", json={
                "model": "m", "input": "q", "stream": True,
            })
        events = sse_pairs(r.text)
        failed = [p for e, p in events if e == "response.failed"]
        assert failed
        assert failed[0]["status"] == "failed"
        assert failed[0]["error"]["code"] == "server_error"
        assert events[-1] == ("response.failed", failed[0])
        assert "[DONE]" not in r.text


class TestInputTokens:
    """The prompt-size probe clients send before a turn.

    MiniMax Code asks for it on every request; without the route the
    client gets a 404, retries, and never shows the number.
    """

    def test_counts_a_plain_prompt(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses/input_tokens", json={
                "model": "m", "input": "one two three",
            })
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 3

    def test_counts_items_without_a_type(self):
        """MiniMax Code sends ``{"role": ..., "content": ...}``.

        The Responses API requires a ``type`` on every item, so a
        client that leaves it out would get a validation error.
        """
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses/input_tokens", json={
                "model": "m",
                "input": [{"role": "developer", "content": "be terse"}],
            })
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 2

    def test_counts_message_items_with_content_parts(self):
        backend = FakeBackend()
        with TestClient(_make_daemon(backend)) as client:
            r = client.post("/v1/responses/input_tokens", json={
                "model": "m",
                "input": [{"type": "message", "role": "user",
                           "content": [{"type": "input_text",
                                        "text": "hello there"}]}],
            })
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 2
