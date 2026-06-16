import time

import pytest

from llmcache import (
    AnthropicBackend,
    CacheExpiredError,
    CacheNotFoundError,
    ExplicitCache,
    GeminiBackend,
    available_backends,
    get_backend,
)


def _docs():
    return [{"role": "user", "content": "Here is a long reference document. " * 50}]


def test_registry_lists_all_platforms():
    assert set(available_backends()) >= {
        "memory",
        "vllm",
        "sglang",
        "openai",
        "anthropic",
        "gemini",
    }


def test_create_get_list_delete_roundtrip():
    cache = ExplicitCache("memory")
    handle = cache.create(model="m", content=_docs(), system="You are helpful.", ttl=60)

    assert handle.id
    entry = cache.get(handle)
    assert entry.spec.system == "You are helpful."
    assert entry.usage.tokens_cached > 0
    assert [e.id for e in cache.list()] == [handle.id]

    cache.delete(handle)
    with pytest.raises(CacheNotFoundError):
        cache.get(handle)
    assert cache.list() == []


def test_expiry_raises_and_drops_from_list():
    cache = ExplicitCache("memory")
    handle = cache.create(model="m", content=_docs(), ttl=0.05)
    time.sleep(0.06)
    with pytest.raises(CacheExpiredError):
        cache.get(handle)
    assert cache.list() == []


def test_extend_revives_ttl():
    cache = ExplicitCache("memory")
    handle = cache.create(model="m", content=_docs(), ttl=0.05)
    cache.extend(handle, ttl=60)
    time.sleep(0.06)
    assert cache.get(handle).id == handle.id


def test_vllm_request_is_openai_compatible_and_prepends_prefix():
    cache = ExplicitCache("vllm")  # no client -> offline, no warming
    handle = cache.create(model="meta-llama/Llama-3.1-8B", system="sys", content=_docs())
    req = cache.build_request(handle, [{"role": "user", "content": "Question?"}], max_tokens=16)

    assert req["model"] == "meta-llama/Llama-3.1-8B"
    assert req["max_tokens"] == 16
    roles = [m["role"] for m in req["messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert req["messages"][-1]["content"] == "Question?"
    assert cache.get(handle).usage.hits == 1


def test_anthropic_places_single_cache_breakpoint_at_end_of_prefix():
    cache = ExplicitCache("anthropic")
    handle = cache.create(model="claude-x", system="sys", content=_docs(), ttl=3600)
    req = cache.build_request(handle, [{"role": "user", "content": "hi"}])

    # system present but breakpoint should be on the last cached message, not system
    assert "cache_control" not in req["system"][0]
    breakpoints = [
        block
        for msg in req["messages"]
        for block in msg["content"]
        if "cache_control" in block
    ]
    assert len(breakpoints) == 1
    assert breakpoints[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_anthropic_breakpoint_on_system_when_no_content():
    cache = ExplicitCache(AnthropicBackend())
    handle = cache.create(model="claude-x", system="just system")
    req = cache.build_request(handle, [{"role": "user", "content": "hi"}])
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_gemini_native_cache_and_request_reference():
    backend = GeminiBackend()  # offline -> synthetic native id
    cache = ExplicitCache(backend)
    assert cache.native_explicit_cache is True

    handle = cache.create(model="gemini-2.0-flash", content=_docs(), system="sys")
    entry = cache.get(handle)
    assert entry.native_id == f"cachedContents/{handle.id}"

    req = cache.build_request(handle, [{"role": "user", "content": "ask"}])
    assert req["config"]["cached_content"] == entry.native_id
    assert req["contents"][0]["role"] == "user"
    assert req["contents"][0]["parts"][0]["text"] == "ask"


def test_gemini_maps_assistant_role_to_model():
    cache = ExplicitCache("gemini")
    handle = cache.create(model="gemini-2.0-flash", content=_docs())
    req = cache.build_request(handle, [{"role": "assistant", "content": "prior"}])
    assert req["contents"][0]["role"] == "model"


class _FakeChat:
    def __init__(self, sink):
        self.sink = sink
        self.completions = self

    def create(self, **kwargs):
        self.sink.append(kwargs)
        return {"ok": True}


class _FakeOpenAIClient:
    def __init__(self):
        self.calls = []
        self.chat = _FakeChat(self.calls)


def test_vllm_warms_on_create_and_completes_via_client():
    client = _FakeOpenAIClient()
    backend = get_backend("vllm", client=client)
    cache = ExplicitCache(backend)

    handle = cache.create(model="m", content=_docs())
    # warm-up request fired on create with max_tokens=1
    assert client.calls and client.calls[0]["max_tokens"] == 1

    cache.complete(handle, [{"role": "user", "content": "go"}], temperature=0.0)
    assert client.calls[-1]["temperature"] == 0.0
    assert client.calls[-1]["messages"][-1]["content"] == "go"
