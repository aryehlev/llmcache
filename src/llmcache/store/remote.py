"""Remote KV store backed by Redis (or any redis-py-compatible client).

Use as the cold/shared tier so explicit caches outlive a single node and can be
reused across a fleet: ``TieredBackend(CPUBackend(...), RedisBackend(client))``.
Payloads are stored with the shared framing from :mod:`llmcache.payload`, so a
cache written by one process is readable by any other pointed at the same Redis.

The client only needs a small redis-py subset: ``set``/``get``/``delete``/
``exists``/``scan_iter``/``persist``/``expire``. ``redis`` is imported lazily, so
importing llmcache never requires it; you may also inject any compatible client
(e.g. a cluster client, or a fake in tests).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ..payload import KVPayload, pack_payload, unpack_payload
from ..types import ChunkKey
from .base import StorageBackend, StoreStats


class RedisBackend(StorageBackend):
    name = "redis"

    def __init__(
        self,
        client: Any = None,
        *,
        url: Optional[str] = None,
        namespace: str = "llmcache",
        key_ttl: Optional[int] = None,
    ) -> None:
        """``client`` takes precedence; else connect via ``url`` (needs redis-py).

        ``key_ttl`` (seconds) sets a default expiry on every stored key so a remote
        tier self-prunes; pinned keys are made persistent (no expiry).
        """
        if client is None:
            if url is None:
                raise ValueError("RedisBackend needs a client or a url")
            try:
                import redis  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("redis-py is required for url-based RedisBackend") from exc
            client = redis.Redis.from_url(url)
        self.client = client
        self.namespace = namespace
        self.key_ttl = key_ttl

    # -- key encoding ------------------------------------------------------
    def _name(self, key: ChunkKey) -> str:
        return f"{self.namespace}:{key.cache_id}:{key.worker_id}:{key.chunk_index}"

    def _parse(self, name: str) -> Optional[ChunkKey]:
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if not name.startswith(self.namespace + ":"):
            return None
        rest = name[len(self.namespace) + 1 :]
        try:
            cache_id, worker_s, chunk_s = rest.rsplit(":", 2)
            return ChunkKey(cache_id, int(chunk_s), int(worker_s))
        except ValueError:
            return None

    # -- StorageBackend ----------------------------------------------------
    def put(self, key: ChunkKey, obj: KVPayload, *, pin: bool = False) -> None:
        name = self._name(key)
        ex = None if pin else self.key_ttl
        self.client.set(name, pack_payload(obj), ex=ex)
        if pin:
            self.client.persist(name)

    def get(self, key: ChunkKey) -> Optional[KVPayload]:
        blob = self.client.get(self._name(key))
        if blob is None:
            return None
        return unpack_payload(blob)

    def contains(self, key: ChunkKey) -> bool:
        return bool(self.client.exists(self._name(key)))

    def remove(self, key: ChunkKey) -> bool:
        return bool(self.client.delete(self._name(key)))

    def pin(self, key: ChunkKey) -> None:
        self.client.persist(self._name(key))

    def unpin(self, key: ChunkKey) -> None:
        if self.key_ttl is not None:
            self.client.expire(self._name(key), self.key_ttl)

    def keys(self) -> Iterable[ChunkKey]:
        out: list[ChunkKey] = []
        for name in self.client.scan_iter(match=f"{self.namespace}:*"):
            parsed = self._parse(name)
            if parsed is not None:
                out.append(parsed)
        return out

    def stats(self) -> StoreStats:
        used = 0
        n = 0
        for key in self.keys():
            blob = self.client.get(self._name(key))
            if blob is not None:
                used += len(blob)
                n += 1
        return StoreStats(
            num_objects=n,
            used_bytes=used,
            capacity_bytes=None,
            pinned_objects=0,
            pinned_bytes=0,
        )
