"""OpenAI message normalization for llama-server compatibility."""

from __future__ import annotations

from tokirecur.backend import LlamaServerBackend, _normalize_messages


def test_consecutive_assistant_text_is_merged():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "next"},
    ]
    out = _normalize_messages(messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[2]["content"] == "first\nsecond"
    assert messages[2]["content"] == "first"


def test_consecutive_assistant_reasoning_and_tool_calls_are_merged():
    messages = [
        {"role": "assistant", "reasoning_content": "think", "content": "a"},
        {
            "role": "assistant",
            "reasoning_content": "more",
            "tool_calls": [{"id": "1"}],
        },
    ]
    out = _normalize_messages(messages)
    assert len(out) == 1
    assert out[0]["content"] == "a"
    assert out[0]["reasoning_content"] == "think\nmore"
    assert out[0]["tool_calls"] == [{"id": "1"}]


def test_consecutive_assistant_list_content_is_merged():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
    ]
    assert _normalize_messages(messages) == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
    ]


def test_empty_assistant_after_assistant_is_dropped():
    messages = [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "q"},
    ]
    out = _normalize_messages(messages)
    assert [m["role"] for m in out] == ["assistant", "user"]
    assert out[0]["content"] == "a"


def test_user_messages_are_not_merged():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    assert _normalize_messages(messages) == messages


def test_complete_openai_sends_normalized_messages():
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ]
            }

    class _Client:
        def __init__(self):
            self.payload = None

        def post(self, url, json, timeout):
            self.payload = json
            return _Response()

    backend = LlamaServerBackend("http://127.0.0.1:1")
    client = _Client()
    backend._client = client
    backend.complete_openai(
        [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
        slot_id=0,
    )
    assert client.payload["messages"] == [
        {"role": "assistant", "content": "a\nb"}
    ]
