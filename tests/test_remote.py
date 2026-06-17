import fnmatch

from llmcache import CPUBackend, KVPayload, RedisBackend, TieredBackend
from llmcache.types import ChunkKey


class FakeRedis:
    """Minimal redis-py-compatible in-memory client for tests."""

    def __init__(self):
        self.kv: dict[str, bytes] = {}
        self.persistent: set[str] = set()
        self.expiries: dict[str, int] = {}

    def set(self, name, value, ex=None):
        self.kv[name] = bytes(value)
        if ex is None:
            self.persistent.add(name)
            self.expiries.pop(name, None)
        else:
            self.expiries[name] = ex
            self.persistent.discard(name)

    def get(self, name):
        return self.kv.get(name)

    def delete(self, *names):
        n = 0
        for name in names:
            if name in self.kv:
                del self.kv[name]
                self.persistent.discard(name)
                self.expiries.pop(name, None)
                n += 1
        return n

    def exists(self, name):
        return 1 if name in self.kv else 0

    def scan_iter(self, match=None):
        for name in list(self.kv):
            if match is None or fnmatch.fnmatch(name, match):
                yield name

    def persist(self, name):
        if name in self.kv:
            self.persistent.add(name)
            self.expiries.pop(name, None)

    def expire(self, name, ttl):
        if name in self.kv:
            self.expiries[name] = ttl
            self.persistent.discard(name)


def obj(size, layers=4):
    return KVPayload(data=b"r" * size, num_tokens=8, num_layers=layers)


def k(i, cache="c"):
    return ChunkKey(cache, i)


def test_redis_put_get_roundtrip_preserves_metadata():
    r = FakeRedis()
    b = RedisBackend(r, namespace="ns")
    b.put(k(0), obj(10, layers=9))
    got = b.get(k(0))
    assert got.nbytes == 10 and got.num_layers == 9 and got.num_tokens == 8
    assert b.contains(k(0))


def test_redis_key_encoding_roundtrips_through_keys():
    r = FakeRedis()
    b = RedisBackend(r)
    key = ChunkKey("cache:with:colons", 3, 1)
    b.put(key, obj(4))
    assert set(b.keys()) == {key}


def test_redis_ttl_and_pin_persist():
    r = FakeRedis()
    b = RedisBackend(r, key_ttl=30)
    b.put(k(0), obj(4))                 # ttl applied
    assert r.expiries[b._name(k(0))] == 30
    b.put(k(1), obj(4), pin=True)       # pinned -> persistent, no expiry
    assert b._name(k(1)) in r.persistent
    b.unpin(k(1))
    assert r.expiries[b._name(k(1))] == 30


def test_redis_remove_and_stats():
    r = FakeRedis()
    b = RedisBackend(r)
    b.put(k(0), obj(10))
    b.put(k(1), obj(20))
    s = b.stats()
    assert s.num_objects == 2 and s.used_bytes > 30  # includes framing header
    assert b.remove(k(0)) and not b.contains(k(0))


def test_tiered_with_redis_cold_demotes_and_promotes():
    hot = CPUBackend(capacity_bytes=25)
    cold = RedisBackend(FakeRedis())
    t = TieredBackend(hot, cold)
    t.put(k(0), obj(10))
    t.put(k(1), obj(10))
    t.put(k(2), obj(10))            # evict k(0) -> demote to redis
    assert not hot.contains(k(0)) and cold.contains(k(0))
    assert t.get(k(0)).nbytes == 10  # promoted back
    assert hot.contains(k(0))
