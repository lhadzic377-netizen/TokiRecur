"""The MemT memory fields, end to end through the daemon.

The ``memt-native-memory`` branch of llama.cpp keeps memory states in
RAM by id: a request names the state it reads (``memory_id``), may
write its own turns into it (``memory_commit``) and may write documents
into it first (``memory_documents``). The daemon has to carry those
fields through untouched, and only when the client used them - a build
without the branch rejects an unknown field.
"""

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from tokirecur.api.app import create_app


@dataclass
class FakeBackend:
    messages: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    memories: list = field(default_factory=lambda: ["alice", "bob"])
    memory_calls: list = field(default_factory=list)
    memory_supported: bool = True

    def tokenize(self, text):
        return text.split()

    def embed(self, text):
        return [0.0, 0.0, 0.0, 0.0]

    def complete_openai(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {
            "text": "ok",
            "tokens_generated": 1,
            "prefill_tokens": 1,
            "ttft_ms": 1.0,
            "total_ms": 1.0,
            "prefill_tokens_per_sec": 1.0,
        }

    def memory_list(self):
        if not self.memory_supported:
            return {"memories": None, "supported": False, "error": "404"}
        return {"memories": self.memories, "supported": True}

    def memory_action(self, memory_id, action):
        self.memory_calls.append((memory_id, action))
        return {"ok": True, "body": {"memory_id": memory_id, "action": action}}


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


def test_the_chat_route_forwards_the_memory_fields(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "I have a kitten named memmy"}],
        "max_tokens": 16,
        "memory_id": "alice",
        "memory_commit": True,
        "memory_documents": ["memmy likes shrimp"],
    })
    assert response.status_code == 200, response.text
    assert backend.kwargs["memory_id"] == "alice"
    assert backend.kwargs["memory_commit"] is True
    assert backend.kwargs["memory_documents"] == ["memmy likes shrimp"]


def test_a_request_without_memory_uses_no_memory(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
    })
    assert response.status_code == 200, response.text
    # Not passed at all: a backend that has never heard of memory must
    # keep working, and a server without the branch rejects the field.
    assert "memory_id" not in backend.kwargs
    assert "memory_commit" not in backend.kwargs
    assert "memory_documents" not in backend.kwargs


def test_the_anthropic_route_forwards_the_memory_fields(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.post("/v1/messages", json={
        "model": "local",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
        "memory_id": "alice",
        "memory_commit": True,
    })
    assert response.status_code == 200, response.text
    assert backend.kwargs["memory_id"] == "alice"
    assert backend.kwargs["memory_commit"] is True


def test_the_memory_list_is_proxied(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.get("/memory")
    assert response.status_code == 200
    assert response.json()["supported"] is True
    assert response.json()["memories"] == ["alice", "bob"]


def test_a_build_without_memt_says_so(tmp_path):
    backend = FakeBackend(memory_supported=False)
    client = make_client(backend, tmp_path)
    response = client.get("/memory")
    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is False
    assert "memt-native-memory" in body["hint"]


def test_a_memory_action_is_proxied(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.post("/memory/alice?action=save")
    assert response.status_code == 200
    assert backend.memory_calls == [("alice", "save")]


def test_an_unknown_memory_action_is_refused(tmp_path):
    backend = FakeBackend()
    client = make_client(backend, tmp_path)
    response = client.post("/memory/alice?action=explode")
    assert response.status_code == 400
    assert "unknown action" in response.json()["detail"]
    assert backend.memory_calls == []


def test_the_payload_carries_the_memory_fields_only_when_used():
    """A build without the branch rejects an unknown field.

    So the fields go out when the client used them and stay away when
    it did not - that is the whole contract with an ordinary build.
    """
    import httpx

    from tokirecur.backend import LlamaServerBackend

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
            "timings": {},
        })

    backend = LlamaServerBackend(base_url="http://127.0.0.1:1")
    backend._client = httpx.Client(transport=httpx.MockTransport(handler))

    backend.complete_openai(
        [{"role": "user", "content": "hi"}], slot_id=0,
        memory_id="alice", memory_commit=True, memory_documents=["doc"],
    )
    assert seen[-1]["memory_id"] == "alice"
    assert seen[-1]["memory_commit"] is True
    assert seen[-1]["memory_documents"] == ["doc"]

    backend.complete_openai([{"role": "user", "content": "hi"}], slot_id=0)
    assert "memory_id" not in seen[-1]
    assert "memory_commit" not in seen[-1]
    assert "memory_documents" not in seen[-1]
