"""Antigravity's CLI on the local model.

Antigravity (``agy``) is the odd one out: it does not speak the OpenAI
API, it speaks the Gemini one - ``/v1beta/models/<model>:generateContent``
with ``contents`` and a ``systemInstruction``. So it needs two things
the other clients do not: a custom model in its settings file, and a
daemon route in Gemini's shape.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from tokirecur import ctl
from tokirecur.api.app import create_app
from tokirecur.api.gemini import (
    GeminiGenerateContentRequest,
    response_body,
    sse_chunk,
    to_openai_messages,
    to_openai_tools,
)
from tokirecur.selector import launch_cmd


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "DEFAULT_REGISTRY", tmp_path / "models.yaml")
    monkeypatch.setattr(ctl, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(ctl, "ACTIVE_MARKER", tmp_path / "active-model")
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "test-tok-0123456789abcdef")
    ctl.DEFAULT_REGISTRY.write_text(
        yaml.safe_dump({"models": ctl.DEFAULT_MODELS}, sort_keys=False)
    )
    return tmp_path


@pytest.fixture
def model(env, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    return ctl.get_model("35b")


def test_the_settings_file_names_the_model_and_keeps_the_users_own(env, model):
    """Antigravity's settings file is the user's.

    Their permissions, their colour scheme and their other models stay;
    only the provider, the selected model and our custom model are ours.
    """
    cfg_path = env / "gemini" / "antigravity-cli" / "settings.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({
        "colorScheme": "tokyo night",
        "permissions": {"allow": ["command(ls)"]},
        "customModelsConfig": {"customModels": {"theirs": {
            "modelName": "gemini-api://their-model",
            "label": "Theirs",
        }}},
    }))

    out = ctl.write_antigravity_config(model, path=cfg_path)
    assert out == cfg_path
    cfg = json.loads(out.read_text())
    assert cfg["modelProvider"] == "gemini"
    # The model is selected by the KEY of the custom model: the CLI
    # resolves the key and refuses a label.
    assert cfg["model"] == "tokirecur"
    ours = cfg["customModelsConfig"]["customModels"]["tokirecur"]
    assert ours["modelName"] == f"gemini-api://{model.model_id}"
    assert ours["label"] == "Toki Recur"
    # The CLI's own output budget is 65536 tokens and it refuses to
    # start with a window that is not larger than that.
    assert ours["maxTokens"] >= 131072
    # The user's own things are still there.
    assert cfg["colorScheme"] == "tokyo night"
    assert cfg["permissions"]["allow"] == ["command(ls)"]
    assert "theirs" in cfg["customModelsConfig"]["customModels"]


def test_the_launch_plan_names_the_gemini_endpoint(monkeypatch):
    """The endpoint is the daemon's root, not its /v1.

    The SDK appends ``/v1beta/models/<model>`` to whatever it is given,
    so the base URL is the daemon's root - and the key travels in
    ``GEMINI_API_KEY``, which is where the Google SDKs look for it.
    """
    monkeypatch.setattr("tokirecur.selector.client_mode", lambda _: "managed")
    monkeypatch.setattr("tokirecur.selector._refresh_client_config",
                        lambda *a, **k: None)
    spec = type("S", (), {
        "name": "ornith",
        "model_id": "ornith-1.5-35b-q4_k_m",
        "llama_port": 8112,
        "daemon_port": 8811,
        "llama_url": "http://127.0.0.1:8112",
        "daemon_url": "http://127.0.0.1:8811",
        "ctx_size": 262144,
    })()
    plan = launch_cmd("chat", "antigravity", spec)
    assert plan["argv"] == ["agy"]
    assert plan["env"]["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:8811"
    assert plan["env"]["GEMINI_API_KEY"]


def test_the_gemini_turn_becomes_the_engine_s_messages():
    request = GeminiGenerateContentRequest.model_validate({
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [
                {"text": "thinking", "thought": True},
                {"functionCall": {"name": "read", "args": {"path": "x"}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "read", "response": {"ok": True}}},
                {"text": "and then?"},
            ]},
        ],
        "systemInstruction": {"parts": [{"text": "be brief"}]},
        "tools": [{"functionDeclarations": [{
            "name": "read",
            "description": "read a file",
            "parameters": {"type": "object"},
        }]}],
    })
    messages = to_openai_messages(request)
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1] == {"role": "user", "content": "hello"}
    # The model's turn keeps its tool call.
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "read"
    # The tool's answer comes back as a tool message.
    assert messages[3]["role"] == "tool"
    assert messages[4]["content"] == "and then?"
    assert to_openai_tools(request.tools) == [{
        "type": "function",
        "function": {
            "name": "read",
            "description": "read a file",
            "parameters": {"type": "object"},
        },
    }]


def test_the_answer_is_candidates_shaped():
    class _Result:
        output_text = "done"
        reasoning = ""
        tool_calls = None
        finish_reason = "stop"
        prefilled_tokens = 5
        generated_tokens = 2

    body = response_body(_Result(), "local-model")
    assert body["candidates"][0]["content"]["role"] == "model"
    assert body["candidates"][0]["content"]["parts"] == [{"text": "done"}]
    assert body["candidates"][0]["finishReason"] == "STOP"
    assert body["usageMetadata"]["totalTokenCount"] == 7
    assert body["modelVersion"] == "local-model"


def test_the_stream_has_no_done_sentinel():
    """Gemini's stream ends when the server closes it.

    A client that reads ``[DONE]`` as JSON - the Google SDK does - fails
    on a sentinel it was never promised.
    """
    chunk = sse_chunk({"candidates": []})
    assert chunk.startswith("data: ")
    assert "[DONE]" not in chunk


@dataclass
class FakeBackend:
    stream_chunks: list = field(default_factory=list)
    non_stream_result: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    def tokenize(self, text):
        return text.split()

    def embed(self, text):
        return [0.0, 0.0, 0.0]

    def complete_openai(self, messages, **kwargs):
        self.messages = messages
        return dict(self.non_stream_result)

    async def complete_openai_stream(self, messages, **kwargs):
        self.messages = messages
        for item in self.stream_chunks:
            yield item


class FakeIndex:
    def search(self, embedding):
        return None

    def insert(self, entry):
        pass


def _client(backend, tmp_path):
    return TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=False,
        enable_rate_limit=False,
    ))


def test_the_daemon_answers_in_geminis_shape(tmp_path):
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
    with _client(backend, tmp_path) as client:
        response = client.post("/v1beta/models/local-model:generateContent", json={
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        })
    assert response.status_code == 200
    body = response.json()
    assert body["candidates"][0]["content"]["parts"] == [{"text": "done"}]
    assert body["candidates"][0]["finishReason"] == "STOP"
    assert backend.messages[0]["content"] == "hello"


def test_the_daemon_streams_gemini_chunks(tmp_path):
    backend = FakeBackend(stream_chunks=[
        {"kind": "delta", "delta": "hello"},
        {"kind": "result", "result": {
            "text": "hello",
            "reasoning": "",
            "tool_calls": None,
            "finish_reason": "stop",
            "tokens_generated": 3,
            "prefill_tokens": 5,
            "ttft_ms": 1.0,
            "total_ms": 2.0,
            "prefill_tokens_per_sec": 1.0,
        }},
    ])
    with _client(backend, tmp_path) as client:
        response = client.post(
            "/v1beta/models/local-model:streamGenerateContent?alt=sse",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
    assert response.status_code == 200
    body = response.text
    assert body.startswith("data: ")
    assert "[DONE]" not in body
    assert "hello" in body


def test_the_gemini_route_accepts_the_goog_api_key(tmp_path, monkeypatch):
    """The Google SDKs carry the key in their own header.

    Antigravity sends ``x-goog-api-key`` and nothing else, so a daemon
    that only reads ``Authorization`` would answer every turn with a
    401.
    """
    monkeypatch.setenv("TOKI_AUTH_TOKEN", "the-token")
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
    client = TestClient(create_app(
        backend=backend,
        index=FakeIndex(),
        cache_dir=tmp_path,
        enable_auth=True,
        enable_rate_limit=False,
    ))
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with client:
        good = client.post(
            "/v1beta/models/local-model:generateContent",
            headers={"x-goog-api-key": "the-token"},
            json=payload,
        )
        bad = client.post(
            "/v1beta/models/local-model:generateContent",
            headers={"x-goog-api-key": "the-wrong-token"},
            json=payload,
        )
    assert good.status_code == 200
    assert bad.status_code == 401


def test_antigravity_is_synced_with_the_rest(env, model, monkeypatch):
    monkeypatch.setattr(ctl, "daemon_reachable", lambda spec: True)
    monkeypatch.setenv("TOKI_ANTIGRAVITY_CONFIG", str(
        env / "gemini" / "antigravity-cli" / "settings.json"
    ))
    written = ctl.sync_cli_configs(model)
    assert "antigravity" in written
    assert Path(written["antigravity"]).exists()
    cfg = json.loads(Path(written["antigravity"]).read_text())
    assert cfg["modelProvider"] == "gemini"
    assert os.path.isabs(written["antigravity"])
