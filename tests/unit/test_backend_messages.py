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


def test_a_mid_conversation_system_message_is_demoted():
    """llama-server's chat templates reject a system message that is
    not the first one.

    The clients that trigger it inject context mid-turn: Claude Code
    sends a hook result as a system message, and Codex sends its
    work-state the same way. Both are really user-visible context, so
    they are demoted rather than dropped - and the request 500s
    otherwise, with a Jinja error that reads like a server fault.
    """
    messages = [
        {"role": "system", "content": "instruction preamble"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "SessionStart hook output"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    out = _normalize_messages(messages)
    assert [m["role"] for m in out] == [
        "system", "user", "user", "assistant", "user",
    ]
    # The content is kept - only the role changes.
    assert out[2]["content"] == "SessionStart hook output"
    # The leading instruction preamble stays where it was.
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "instruction preamble"


def test_a_leading_system_message_is_never_demoted():
    messages = [
        {"role": "system", "content": "instruction preamble"},
        {"role": "user", "content": "hi"},
    ]
    assert _normalize_messages(messages) == messages


def test_two_mid_conversation_system_messages_are_both_demoted():
    messages = [
        {"role": "system", "content": "preamble"},
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "second"},
    ]
    out = _normalize_messages(messages)
    assert [m["role"] for m in out] == ["system", "user", "user", "user"]
