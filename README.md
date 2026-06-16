# llmcache

**Explicit KV caching for LLM serving engines.**

An [LMCache](https://github.com/LMCache/LMCache)-style KV offload and reuse layer
— but **explicit**. Instead of automatically hashing token content and scanning
for the longest shared prefix, your application registers KV under a **cache id**
(a handle it owns) and references that handle deliberately. The trade is intent
for determinism.

## Explicit vs. automatic (why)

LMCache reuses KV implicitly: it chunks every request, hashes the tokens, and
looks for the longest matching prefix already in its store (CPU, disk, remote). It
is powerful and zero-touch, but reuse is best-effort and opaque — you don't decide
what is cached, when it warms, or whether it survives memory pressure.

`llmcache` makes caching a first-class, explicit operation:

| | LMCache (automatic) | llmcache (explicit) |
|---|---|---|
| Lookup key | rolling hash of token content | application-owned cache id |
| Hit semantics | best-effort longest-prefix match | deterministic, O(1) by id |
| Eviction | LRU under memory pressure | **pinning** — guaranteed-resident caches |
| Warming | on first matching request | register/precompute **offline**, ahead of traffic |
| False sharing | possible across tenants/prompts | impossible (ids are explicit) |
| Correctness guard | token match | token-fingerprint prefix check on reuse |

You still get tiered storage (CPU → disk → remote) and prefix-level partial reuse;
you just drive it on purpose. Great for: shared system prompts, RAG document packs,
long few-shot blocks, agent tool catalogues, and multi-tenant isolation.

## Architecture

```
            ┌─────────────────────────────────────────────┐
            │            ExplicitKVCache (engine)          │  control plane:
            │   register · lookup · load · pin · ttl       │  ids → chunks, fingerprints
            └───────────────┬─────────────────────────────┘
                            │ opaque KVPayload chunks
            ┌───────────────▼─────────────────────────────┐
            │  StorageBackend: CPU → Disk → (remote)       │  tiered, LRU + pinning
            └───────────────┬─────────────────────────────┘
                            │ KVTransfer (tensor copy)
   ┌────────────────────────▼──────────────┐   ┌───────────────────────────┐
   │ ExplicitLMCacheConnector (vLLM V1)     │   │ SGLangExplicitCache (hook) │
   └────────────────────────────────────────┘   └───────────────────────────┘
```

The engine and storage layers are **tensor- and framework-agnostic** — they index
opaque payloads — so the whole control plane runs and is tested without torch or a
GPU. Real KV crosses the boundary only inside a `KVTransfer` adapter at the
connector.

## Install

```bash
pip install -e .            # core engine + storage + connectors (zero deps)
pip install -e ".[torch]"   # + torch/numpy for real KV (de)serialization
pip install -e ".[dev]"     # + pytest
```

## Quickstart (engine, offline)

```python
from llmcache import ExplicitKVCache, CPUBackend, DiskBackend, TieredBackend

store = TieredBackend(CPUBackend(capacity_bytes=8 << 30), DiskBackend("/var/kvcache"))
engine = ExplicitKVCache(store, chunk_size=256)

# Register KV for a prompt under an explicit, pinned handle.
engine.register(prompt_token_ids, kv_payloads, cache_id="handbook-v1", ttl=3600, pin=True)

# Later requests reference it; partial prefix reuse is verified by fingerprint.
result = engine.lookup("handbook-v1", request_token_ids)
print(result.num_tokens, "tokens reusable")
payloads = engine.load("handbook-v1", num_chunks=result.num_chunks)
```

## vLLM integration

`ExplicitLMCacheConnector` implements vLLM's V1 `KVConnectorBase_V1`. Requests opt
in via `kv_transfer_params` on `SamplingParams`:

```python
# reuse a registered cache
SamplingParams(extra_args={"kv_transfer_params": {"cache_id": "handbook-v1"}})

# create/refresh a cache from this request's computed KV
SamplingParams(extra_args={"kv_transfer_params": {
    "cache_id": "handbook-v1", "save": True, "ttl": 3600, "pin": True, "name": "handbook",
}})
```

Scheduler side: `get_num_new_matched_tokens` reports how much KV is reusable for
that id (verified against the prompt prefix); `build_connector_meta` ships per-request
load/save ops to the workers. Worker side: `start_load_kv` copies cached KV into the
paged buffer via a `KVTransfer`; `wait_for_save` captures and registers new KV.

> **Tensor copy maturity.** The engine, storage, scheduling, and metadata logic are
> covered by the test suite. The torch paged-buffer copy (`TorchKVTransfer`) is
> framework- and vLLM-version-sensitive and is **not** exercised here — validate it
> against your target vLLM build before production. `DictKVTransfer` lets you dry-run
> the entire connector lifecycle without torch (see `examples/quickstart.py`).

## SGLang integration

`SGLangExplicitCache` exposes the same engine through a `store`/`retrieve` surface
suitable for SGLang's host KV cache hook, so a cache registered for vLLM is reusable
from SGLang and vice-versa.

## API surface

- `ExplicitKVCache(store=None, *, chunk_size=256, default_ttl=None)`
  - `.register(token_ids, payloads, *, cache_id=None, ttl=None, pin=False, name=None)`
  - `.lookup(cache_id, request_tokens=None) -> LookupResult`
  - `.load(cache_id, *, num_chunks=None, worker_id=0) -> list[KVPayload]`
  - `.get / .exists / .list / .delete / .extend_ttl / .pin / .unpin / .sweep_expired / .stats`
  - low-level: `.put_chunk(...)`, `.commit(...)` (used by connectors)
- Storage: `CPUBackend`, `DiskBackend`, `TieredBackend` (implement `StorageBackend` to add Redis/S3/…)
- Payload: `KVPayload`, `RawSerializer`, `TorchSerializer`
- Connectors: `ExplicitLMCacheConnector`, `SGLangExplicitCache`, `KVTransfer` (`DictKVTransfer`, `TorchKVTransfer`)

## Tests

```bash
pip install -e ".[dev]" && pytest
```
