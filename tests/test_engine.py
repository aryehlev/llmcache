import time

import pytest

from llmcache import (
    CacheExpiredError,
    CacheNotFoundError,
    ExplicitKVCache,
    KVPayload,
    chunk_token_ids,
    matched_chunk_count,
)


def payloads(n, size=32, layers=4):
    return [KVPayload(data=b"x" * size, num_tokens=0, num_layers=layers) for _ in range(n)]


def test_chunking_and_fingerprint_prefix_match():
    eng = ExplicitKVCache(chunk_size=4)
    tokens = list(range(10))  # 3 chunks: [0-3],[4-7],[8-9]
    desc = eng.register(tokens, payloads(3), cache_id="c")
    assert desc.num_chunks == 3
    assert desc.num_tokens == 10

    # exact same prompt -> full reuse
    full = eng.lookup("c", tokens)
    assert full.full and full.num_tokens == 10

    # shares first two chunks then diverges -> partial reuse (2 chunks = 8 tokens)
    diverged = list(range(8)) + [99, 99]
    part = eng.lookup("c", diverged)
    assert not part.full and part.num_tokens == 8 and part.num_chunks == 2

    # no shared prefix
    assert eng.lookup("c", [42, 42, 42, 42]).num_tokens == 0


def test_lookup_without_tokens_trusts_full_coverage():
    eng = ExplicitKVCache(chunk_size=4)
    eng.register(list(range(8)), payloads(2), cache_id="c")
    res = eng.lookup("c")
    assert res.full and res.num_tokens == 8


def test_register_validates_chunk_count():
    eng = ExplicitKVCache(chunk_size=4)
    with pytest.raises(ValueError):
        eng.register(list(range(8)), payloads(1))  # needs 2 chunks


def test_load_returns_ordered_payloads_and_multiworker():
    eng = ExplicitKVCache(chunk_size=4)
    eng.register(
        list(range(8)),
        {0: payloads(2, size=10), 1: payloads(2, size=20)},
        cache_id="c",
    )
    w0 = eng.load("c", worker_id=0)
    w1 = eng.load("c", worker_id=1)
    assert [p.nbytes for p in w0] == [10, 10]
    assert [p.nbytes for p in w1] == [20, 20]


def test_ttl_expiry_and_extend():
    eng = ExplicitKVCache(chunk_size=4)
    eng.register(list(range(4)), payloads(1), cache_id="c", ttl=0.05)
    assert eng.exists("c")
    time.sleep(0.06)
    with pytest.raises(CacheExpiredError):
        eng.get("c")
    assert not eng.exists("c")  # swept on access

    eng.register(list(range(4)), payloads(1), cache_id="c2", ttl=0.05)
    eng.extend_ttl("c2", 60)
    time.sleep(0.06)
    assert eng.exists("c2")


def test_delete_removes_chunks_from_store():
    eng = ExplicitKVCache(chunk_size=4)
    eng.register(list(range(8)), payloads(2), cache_id="c")
    assert eng.stats().num_objects == 2
    eng.delete("c")
    assert eng.stats().num_objects == 0
    with pytest.raises(CacheNotFoundError):
        eng.get("c")


def test_pin_protects_from_eviction():
    from llmcache import CPUBackend

    # capacity for ~2 objects of 32 bytes
    eng = ExplicitKVCache(CPUBackend(capacity_bytes=80), chunk_size=4)
    eng.register(list(range(4)), payloads(1, size=32), cache_id="pinned", pin=True)
    # add more unpinned caches to force eviction pressure
    for i in range(5):
        eng.register([100 + i] * 4, payloads(1, size=32), cache_id=f"u{i}")
    # pinned chunk survives
    assert eng.load("pinned")[0].nbytes == 32


def test_matched_chunk_count_helper():
    eng = ExplicitKVCache(chunk_size=4)
    desc = eng.register(list(range(8)), payloads(2), cache_id="c")
    assert matched_chunk_count(list(range(8)), desc) == 2
    assert matched_chunk_count(list(range(4)) + [0, 0, 0, 0], desc) == 1
    assert len(chunk_token_ids(list(range(10)), 4)) == 3
