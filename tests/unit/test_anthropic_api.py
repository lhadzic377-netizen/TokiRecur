import asyncio
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur.api.anthropic import (
    AnthropicMessagesRequest,
    stream_message,
    to_openai_messages,
    to_openai_tools,
)
from tokirecur.api.app import create_app


@dataclass
class FakeBackend:
    stream_chunks: list[dict] = field(default_factory=list)
    non_stream_result: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    tools: list | None = None
    tool_choice: object = None

    def tokenize(self, text):
        return text.split()

    def embed(self, text):
        return [0.0, 0.0, 0.0, 0.0]

    def complete_openai(self, messages, **kwargs):
        self.messages = messages
        self.tools = kwargs.get("tools")
        self.tool_choice = kwargs.get("tool_choice")
        return dict(self.non_stream_result)

    async def complete_openai_stream(self, messages, **kwargs):
        self.messages = messages
        self.tools = kwargs.get("tools")
        self.tool_choice = kwargs.get("tool_choice")
        for item in self.stream_chunks:
            yield item


class FakeIndex:
    def search(self, embedding):
        return None

    def insert(self, entry):
        pass


def make_client(backend, tmp_path):
    return TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=False,
        enable_rate_limit=False,
    ))


def test_anthropic_request_converts_system_tools_and_messages():
    request = AnthropicMessagesRequest.model_validate({
        "model": "local",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "system prompt"}],
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }],
        "tools": [{
            "name": "Bash",
            "description": "run a command",
            "input_schema": {"type": "object"},
        }],
    })
    messages = to_openai_messages(request)
    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert to_openai_tools(request.tools) == [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run a command",
            "parameters": {"type": "object"},
        },
    }]


def test_anthropic_non_stream_response_is_messages_shaped(tmp_path):
    backend = FakeBackend(non_stream_result={
        "text": "done",
        "tool_calls": None,
        "finish_reason": "stop",
        "tokens_generated": 2,
        "prefill_tokens": 5,
        "ttft_ms": 1.0,
        "total_ms": 2.0,
        "prefill_tokens_per_sec": 1.0,
    })
    with make_client(backend, tmp_path) as client:
        response = client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        })
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "done"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert backend.messages[0]["content"] == "hello"


def test_anthropic_stream_emits_message_and_tool_events(tmp_path):
    backend = FakeBackend(stream_chunks=[
        {"kind": "reasoning", "delta": "think"},
        {"kind": "delta", "delta": "hello"},
        {"kind": "tool_calls", "tool_calls": [{
            "index": 0,
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": "{}"},
        }]},
        {"kind": "finish", "finish_reason": "tool_calls"},
        {"kind": "result", "result": {
            "text": "hello",
            "reasoning": "think",
            "tool_calls": [{
                "index": 0,
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            }],
            "finish_reason": "tool_calls",
            "tokens_generated": 3,
            "prefill_tokens": 5,
            "ttft_ms": 1.0,
            "total_ms": 2.0,
            "prefill_tokens_per_sec": 1.0,
        }},
    ])
    with make_client(backend, tmp_path) as client:
        response = client.post("/anthropic/v1/messages", json={
            "model": "local",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            "stream": True,
        })
    assert response.status_code == 200
    events = [
        line[len("event: "):]
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert "content_block_start" in events
    assert "content_block_delta" in events
    assert events[-2:] == ["message_delta", "message_stop"]
    assert '"type":"tool_use"' in response.text
    assert '"stop_reason":"tool_use"' in response.text


def test_anthropic_stream_pings_while_the_engine_is_silent():
    async def slow_items():
        await asyncio.sleep(0.12)
        yield {"kind": "delta", "delta": "hi"}
        yield {"kind": "finish", "finish_reason": "stop"}
        yield {"kind": "result", "result": {
            "text": "hi",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "tokens_generated": 1,
            "prefill_tokens": 1,
            "ttft_ms": 1.0,
            "total_ms": 1.0,
            "prefill_tokens_per_sec": 1.0,
        }}

    async def collect():
        return [
            chunk async for chunk in stream_message(
                slow_items(), "local", ping_interval=0.02,
            )
        ]

    chunks = asyncio.run(collect())
    names = [
        line[len("event: "):]
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("event: ")
    ]
    assert names[0] == "message_start"
    assert "ping" in names
    assert names.index("ping") < names.index("content_block_start")
    assert names[-2:] == ["message_delta", "message_stop"]


def test_anthropic_stream_keeps_generation_alive_across_waits():
    async def items():
        yield {"kind": "delta", "delta": "a"}
        await asyncio.sleep(0.1)
        yield {"kind": "delta", "delta": "b"}
        yield {"kind": "finish", "finish_reason": "stop"}
        yield {"kind": "result", "result": {
            "text": "ab",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "tokens_generated": 2,
            "prefill_tokens": 1,
            "ttft_ms": 1.0,
            "total_ms": 1.0,
            "prefill_tokens_per_sec": 1.0,
        }}

    async def collect():
        return [
            chunk async for chunk in stream_message(
                items(), "local", ping_interval=0.02,
            )
        ]

    chunks = asyncio.run(collect())
    body = "".join(chunks)
    assert body.index('"text":"a"') < body.index("event: ping") < body.index('"text":"b"')
    assert body.rstrip().endswith(
        'data: {"type":"message_stop"}',
    )


def test_anthropic_tool_results_are_forwarded_as_tool_messages():
    request = AnthropicMessagesRequest.model_validate({
        "model": "local",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                {"type": "text", "text": "continue"},
            ],
        }],
    })
    assert to_openai_messages(request) == [
        {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
        {"role": "user", "content": "continue"},
    ]


def test_anthropic_accepts_x_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "x-api-secret")
    backend = FakeBackend(non_stream_result={
        "text": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "tokens_generated": 1,
        "prefill_tokens": 1,
        "ttft_ms": 1.0,
        "total_ms": 1.0,
        "prefill_tokens_per_sec": 1.0,
    })
    app = create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=True,
        enable_rate_limit=False,
    )
    with TestClient(app) as client:
        response = client.post(
            "/anthropic/v1/messages",
            headers={"x-api-key": "x-api-secret"},
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200


def test_anthropic_accepts_either_credential_header(monkeypatch, tmp_path):
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "daemon-secret")
    backend = FakeBackend(non_stream_result={
        "text": "ok",
        "tool_calls": None,
        "finish_reason": "stop",
        "tokens_generated": 1,
        "prefill_tokens": 1,
        "ttft_ms": 1.0,
        "total_ms": 1.0,
        "prefill_tokens_per_sec": 1.0,
    })
    app = create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=True,
        enable_rate_limit=False,
    )
    payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(app) as client:
        # A stale ANTHROPIC_AUTH_TOKEN in the user's shell makes Claude Code
        # send a wrong Bearer alongside the correct x-api-key.
        both = client.post(
            "/anthropic/v1/messages",
            headers={
                "authorization": "Bearer stale-shell-token",
                "x-api-key": "daemon-secret",
            },
            json=payload,
        )
        bearer_only = client.post(
            "/anthropic/v1/messages",
            headers={"authorization": "Bearer daemon-secret"},
            json=payload,
        )
        wrong_both = client.post(
            "/anthropic/v1/messages",
            headers={
                "authorization": "Bearer stale-shell-token",
                "x-api-key": "also-stale",
            },
            json=payload,
        )
        no_credential = client.post(
            "/anthropic/v1/messages",
            json=payload,
        )
    assert both.status_code == 200
    assert bearer_only.status_code == 200
    assert wrong_both.status_code == 401
    assert wrong_both.json()["detail"] == "invalid token"
    assert no_credential.status_code == 401


def test_anthropic_hello_probe_is_public(tmp_path):
    backend = FakeBackend()
    app = create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=True,
        enable_rate_limit=False,
    )
    with TestClient(app) as client:
        head = client.head("/anthropic/api/hello")
        get = client.get("/anthropic/api/hello")
    assert head.status_code == 200
    assert get.status_code == 200
    assert get.json() == {"status": "ok"}


def test_anthropic_count_tokens_endpoint(tmp_path):
    backend = FakeBackend()
    with make_client(backend, tmp_path) as client:
        response = client.post("/anthropic/v1/messages/count_tokens", json={
            "model": "local",
            "messages": [{"role": "user", "content": "one two three"}],
        })
    assert response.status_code == 200
    assert response.json()["input_tokens"] == 3
