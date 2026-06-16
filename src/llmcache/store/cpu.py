"""In-memory (host RAM) KV store with LRU eviction and pinning.

This is the hot tier — the analogue of LMCache's CPU offload buffer. Capacity is a
byte budget; on overflow the least-recently-used *unpinned* object is evicted. An
optional ``on_evict`` callback lets a tiered store demote evicted objects to a
slower tier instead of dropping them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Iterable, Optional

from ..errors import StorageFullError
from ..payload import KVPayload
from ..types import ChunkKey
from .base import StorageBackend, StoreStats

EvictCallback = Callable[[ChunkKey, KVPayload], None]


class CPUBackend(StorageBackend):
    name = "cpu"

    def __init__(
        self,
        capacity_bytes: Optional[int] = None,
        on_evict: Optional[EvictCallback] = None,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.on_evict = on_evict
        self._data: "OrderedDict[ChunkKey, KVPayload]" = OrderedDict()
        self._pinned: set[ChunkKey] = set()
        self._used = 0
        self._lock = threading.RLock()

    def put(self, key: ChunkKey, obj: KVPayload, *, pin: bool = False) -> None:
        with self._lock:
            if key in self._data:
                self._used -= self._data[key].nbytes
                del self._data[key]
            self._ensure_room(obj.nbytes, protect=key)
            self._data[key] = obj
            self._data.move_to_end(key)
            self._used += obj.nbytes
            if pin:
                self._pinned.add(key)

    def get(self, key: ChunkKey) -> Optional[KVPayload]:
        with self._lock:
            obj = self._data.get(key)
            if obj is not None:
                self._data.move_to_end(key)  # mark as recently used
            return obj

    def contains(self, key: ChunkKey) -> bool:
        with self._lock:
            return key in self._data

    def remove(self, key: ChunkKey) -> bool:
        with self._lock:
            obj = self._data.pop(key, None)
            if obj is None:
                return False
            self._used -= obj.nbytes
            self._pinned.discard(key)
            return True

    def pin(self, key: ChunkKey) -> None:
        with self._lock:
            if key in self._data:
                self._pinned.add(key)

    def unpin(self, key: ChunkKey) -> None:
        with self._lock:
            self._pinned.discard(key)

    def keys(self) -> Iterable[ChunkKey]:
        with self._lock:
            return list(self._data.keys())

    def stats(self) -> StoreStats:
        with self._lock:
            pinned_bytes = sum(self._data[k].nbytes for k in self._pinned if k in self._data)
            return StoreStats(
                num_objects=len(self._data),
                used_bytes=self._used,
                capacity_bytes=self.capacity_bytes,
                pinned_objects=len(self._pinned),
                pinned_bytes=pinned_bytes,
            )

    # -- internals ---------------------------------------------------------
    def _ensure_room(self, incoming: int, protect: ChunkKey) -> None:
        if self.capacity_bytes is None:
            return
        if incoming > self.capacity_bytes and not self._pinned:
            raise StorageFullError(
                f"object of {incoming} bytes exceeds capacity {self.capacity_bytes}"
            )
        for key in list(self._data.keys()):
            if self._used + incoming <= self.capacity_bytes:
                return
            if key == protect or key in self._pinned:
                continue
            victim = self._data.pop(key)
            self._used -= victim.nbytes
            if self.on_evict is not None:
                self.on_evict(key, victim)
        if self._used + incoming > self.capacity_bytes:
            raise StorageFullError(
                "cannot free enough space: remaining objects are pinned"
            )
