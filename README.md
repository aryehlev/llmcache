# llmcache

A unified **explicit-caching** control plane for LLM serving platforms.

Declare reusable prompt content once — a system instruction, a large document, a
few-shot block, a tool catalogue — get a portable handle back, and reference it
across many requests. The same API drives platforms with a native cache object
(Google Gemini) and platforms that only have prefix / prompt caching (vLLM,
SGLang, OpenAI, Anthropic).

## Why "explicit"?

LLM serving engines reuse work in two ways:

- **Implicit / automatic prefix caching** — the engine notices that two requests
  share a prefix and reuses the computed KV blocks (vLLM `--enable-prefix-caching`,
  SGLang RadixAttention, OpenAI automatic prompt caching). You don't control *what*
  is cached or *when* it warms.
- **Explicit caching** — you declare the content to cache, get a handle/TTL, and
  reference it deliberately (Gemini `cachedContents`, Anthropic `cache_control`
  breakpoints).

`llmcache` gives you the **explicit** model everywhere. For native platforms it
calls the real cache API. For prefix-cache platforms it tracks the prefix + TTL
for you and re-materialises it into each request (optionally warming the engine
first), so you get the same create/reference/expire workflow regardless of where
you serve.

## Install

```bash
pip install -e .            # core, zero dependencies
pip install -e ".[openai]"  # + openai SDK (also used for vLLM / SGLang servers)
pip install -e ".[anthropic]"
pip install -e ".[gemini]"
```

## Quickstart

```python
from llmcache import ExplicitCache

cache = ExplicitCache("vllm")  # or "sglang", "openai", "anthropic", "gemini", "memory"

handle = cache.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system="You are an HR assistant. Answer only from the handbook.",
    content=[{"role": "user", "content": LONG_HANDBOOK_TEXT}],
    ttl=3600,
    name="handbook-v1",
)

request = cache.build_request(
    handle,
    [{"role": "user", "content": "How many vacation days do I get?"}],
    max_tokens=256,
)
# `request` is the platform-native payload, with the cached prefix attached.
```

### Going live

Pass the platform's SDK client and let `llmcache` warm, reference, and (for native
caches) create/delete the server-side object:

```python
from openai import OpenAI
from llmcache import ExplicitCache

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
cache = ExplicitCache("vllm", client=client)        # warms the prefix on create()

handle = cache.create(model="meta-llama/Llama-3.1-8B-Instruct", content=docs, ttl=600)
resp = cache.complete(handle, [{"role": "user", "content": "Q?"}])
```

```python
from google import genai
from llmcache import ExplicitCache

cache = ExplicitCache("gemini", client=genai.Client())  # uses native cachedContents
handle = cache.create(model="gemini-2.0-flash", content=docs, ttl=600)
resp = cache.complete(handle, [{"role": "user", "content": "Q?"}])
```

## Supported platforms

| Platform   | Backend name | Mechanism                                  | Native cache API |
|------------|--------------|--------------------------------------------|------------------|
| vLLM       | `vllm`       | automatic prefix caching + warm-up         | no               |
| SGLang     | `sglang`     | RadixAttention prefix caching + warm-up    | no               |
| OpenAI     | `openai`     | implicit prompt caching                    | no               |
| Anthropic  | `anthropic`  | `cache_control` breakpoints                | no (request-level)|
| Gemini     | `gemini`     | `cachedContents`                           | yes              |
| in-memory  | `memory`     | reference backend for tests / local dev    | no               |

## API

- `ExplicitCache(backend, **client_kwargs)` — `backend` is a platform name or a
  `CacheBackend` instance.
- `.create(model=..., content=..., system=..., tools=..., ttl=..., name=...) -> CacheHandle`
- `.get(ref)`, `.list()`, `.delete(ref)`, `.extend(ref, ttl)`
- `.build_request(ref, messages, **params) -> dict` — platform-native payload.
- `.complete(ref, messages, **params)` — runs it via the configured client.

`ref` may be a `CacheHandle`, a `CacheEntry`, or a raw cache id string.

### Adding a platform

Subclass `LocalPrefixCacheBackend` (prefix-cache style) or `CacheBackend`
(native), then register it:

```python
from llmcache import LocalPrefixCacheBackend, register_backend

class MyEngineBackend(LocalPrefixCacheBackend):
    name = "myengine"

register_backend("myengine", MyEngineBackend)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
