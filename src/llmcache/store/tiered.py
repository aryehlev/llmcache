"""Two-tier store: fast hot tier backed by a slower cold tier.

Reads fall through hot -> cold and promote cold hits back to hot. Writes land in
hot; when hot evicts an unpinned object it is demoted to cold rather than dropped.
This mirrors LMCache's CPU + disk (or CPU + remote) hierarchy, but composition is
generic: any two :class:`StorageBackend` instances work, e.g. CPU + Redis.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..payload import KVPayload
from ..types import ChunkKey
from .base import StorageBackend, StoreStats
from .cpu import CPUBackend


class TieredBackend(StorageBackend):
    name = "tiered"

    def __init__(self, hot: StorageBackend, cold: StorageBackend) -> None:
        self.hot = hot
        self.cold = cold
        # Wire hot-tier eviction to demote into the cold tier.
        if isinstance(hot, CPUBackend):
            hot.on_evict = self._demote

    def _demote(self, key: ChunkKey, obj: KVPayload) -> None:
        self.cold.put(key, obj)

    def put(self, key: ChunkKey, obj: KVPayload, *, pin: bool = False) -> None:
        self.hot.put(key, obj, pin=pin)
        if pin:
            # Pinned objects are also persisted to the cold tier so a hot-tier
            # restart does not lose a guaranteed cache.
            self.cold.put(key, obj, pin=True)

    def get(self, key: ChunkKey) -> Optional[KVPayload]:
        obj = self.hot.get(key)
        if obj is not None:
            return obj
        obj = self.cold.get(key)
        if obj is not None:
            self.hot.put(key, obj)  # promote
        return obj

    def contains(self, key: ChunkKey) -> bool:
        return self.hot.contains(key) or self.cold.contains(key)

    def remove(self, key: ChunkKey) -> bool:
        removed_hot = self.hot.remove(key)
        removed_cold = self.cold.remove(key)
        return removed_hot or removed_cold

    def pin(self, key: ChunkKey) -> None:
        self.hot.pin(key)
        self.cold.pin(key)

    def unpin(self, key: ChunkKey) -> None:
        self.hot.unpin(key)
        self.cold.unpin(key)

    def keys(self) -> Iterable[ChunkKey]:
        seen = set(self.hot.keys())
        for key in self.cold.keys():
            seen.add(key)
        return list(seen)

    def stats(self) -> StoreStats:
        h, c = self.hot.stats(), self.cold.stats()
        cap = None
        if h.capacity_bytes is not None and c.capacity_bytes is not None:
            cap = h.capacity_bytes + c.capacity_bytes
        return StoreStats(
            num_objects=len(list(self.keys())),
            used_bytes=h.used_bytes + c.used_bytes,
            capacity_bytes=cap,
            pinned_objects=h.pinned_objects + c.pinned_objects,
            pinned_bytes=h.pinned_bytes + c.pinned_bytes,
        )
