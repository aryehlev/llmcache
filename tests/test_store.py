import pytest

from llmcache import CPUBackend, DiskBackend, KVPayload, StorageFullError, TieredBackend
from llmcache.types import ChunkKey


def obj(size, tok=8, layers=4):
    return KVPayload(data=b"a" * size, num_tokens=tok, num_layers=layers)


def k(i):
    return ChunkKey("cache", i)


def test_cpu_put_get_remove():
    s = CPUBackend()
    s.put(k(0), obj(10))
    assert s.get(k(0)).nbytes == 10
    assert s.contains(k(0))
    assert s.remove(k(0)) and not s.contains(k(0))
    assert s.remove(k(0)) is False


def test_cpu_lru_eviction():
    s = CPUBackend(capacity_bytes=25)
    s.put(k(0), obj(10))
    s.put(k(1), obj(10))
    s.get(k(0))               # touch 0 so 1 is now LRU
    s.put(k(2), obj(10))      # evicts k(1)
    assert s.contains(k(0)) and s.contains(k(2))
    assert not s.contains(k(1))


def test_cpu_pinned_not_evicted():
    s = CPUBackend(capacity_bytes=25)
    s.put(k(0), obj(10), pin=True)
    s.put(k(1), obj(10))
    s.put(k(2), obj(10))      # must evict k(1), never k(0)
    assert s.contains(k(0))
    assert not s.contains(k(1))


def test_cpu_all_pinned_raises_when_full():
    s = CPUBackend(capacity_bytes=15)
    s.put(k(0), obj(10), pin=True)
    with pytest.raises(StorageFullError):
        s.put(k(1), obj(10))


def test_disk_roundtrip_and_persistence(tmp_path):
    s = DiskBackend(str(tmp_path))
    s.put(k(0), obj(12, layers=7))
    got = s.get(k(0))
    assert got.nbytes == 12 and got.num_layers == 7
    # new backend over same dir sees persisted data
    s2 = DiskBackend(str(tmp_path))
    assert s2.contains(k(0))
    assert set(s2.keys()) == {k(0)}
    assert s2.remove(k(0)) and not s2.contains(k(0))


def test_tiered_demote_on_eviction_and_promote_on_read(tmp_path):
    hot = CPUBackend(capacity_bytes=25)
    cold = DiskBackend(str(tmp_path))
    t = TieredBackend(hot, cold)

    t.put(k(0), obj(10))
    t.put(k(1), obj(10))
    t.put(k(2), obj(10))      # evicts k(0) from hot -> demoted to cold

    assert not hot.contains(k(0))
    assert cold.contains(k(0))

    got = t.get(k(0))         # cold hit -> promote back to hot
    assert got.nbytes == 10
    assert hot.contains(k(0))


def test_tiered_pin_persists_to_cold(tmp_path):
    hot = CPUBackend(capacity_bytes=100)
    cold = DiskBackend(str(tmp_path))
    t = TieredBackend(hot, cold)
    t.put(k(0), obj(10), pin=True)
    assert cold.contains(k(0))  # pinned objects mirrored to cold tier
