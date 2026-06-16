"""Storage backend abstraction for KV payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

from ..payload import KVPayload
from ..types import ChunkKey


@dataclass
class StoreStats:
    num_objects: int
    used_bytes: int
    capacity_bytes: Optional[int]
    pinned_objects: int
    pinned_bytes: int


class StorageBackend(ABC):
    """A keyed store of :class:`KVPayload` objects with optional pinning.

    Implementations decide their own eviction policy and capacity. ``put`` may
    evict unpinned objects to make room; if it cannot, it raises
    :class:`~llmcache.errors.StorageFullError`.
    """

    name: str = "base"

    @abstractmethod
    def put(self, key: ChunkKey, obj: KVPayload, *, pin: bool = False) -> None: ...

    @abstractmethod
    def get(self, key: ChunkKey) -> Optional[KVPayload]: ...

    @abstractmethod
    def contains(self, key: ChunkKey) -> bool: ...

    @abstractmethod
    def remove(self, key: ChunkKey) -> bool:
        """Remove a key (even if pinned). Returns True if it existed."""

    @abstractmethod
    def pin(self, key: ChunkKey) -> None: ...

    @abstractmethod
    def unpin(self, key: ChunkKey) -> None: ...

    @abstractmethod
    def keys(self) -> Iterable[ChunkKey]: ...

    @abstractmethod
    def stats(self) -> StoreStats: ...

    def clear(self) -> None:
        for key in list(self.keys()):
            self.remove(key)
