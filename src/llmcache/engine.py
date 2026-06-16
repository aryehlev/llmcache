"""ExplicitKVCache — the explicit-caching control plane (LMCache's role).

Where LMCache reuses KV automatically by hashing token content and scanning for
the longest shared prefix, this engine reuses KV only for content the application
has **explicitly registered under a cache id**. The trade is intent for
determinism: registered caches can be pinned (never evicted out from under you),
warmed offline, shared across requests without false matches, and looked up in
O(1) by id. Optional token fingerprints still validate that a request's prompt
really is a prefix of the registered tokens before any KV is reused, so an
explicit hit can never serve stale/garbage KV.

The engine is storage- and tensor-agnostic: it indexes opaque
:class:`~llmcache.payload.KVPayload` chunks held by a :class:`StorageBackend`.
That makes the whole control plane testable without torch or a GPU; the connector
layer supplies real KV at the engine boundary.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

from .errors import CacheExpiredError, CacheMismatchError, CacheNotFoundError
from .payload import KVPayload
from .store import CPUBackend, StorageBackend
from .types import (
    CacheDescriptor,
    ChunkKey,
    chunk_token_ids,
    matched_chunk_count,
    token_fingerprint,
)

PayloadsByWorker = Union[Sequence[KVPayload], Mapping[int, Sequence[KVPayload]]]


@dataclass
class LookupResult:
    cache_id: str
    num_tokens: int          # reusable token count given the request
    num_chunks: int          # reusable chunk count
    full: bool               # whether the entire registered cache is reusable


class ExplicitKVCache:
    def __init__(
        self,
        store: Optional[StorageBackend] = None,
        *,
        chunk_size: int = 256,
        default_ttl: Optional[float] = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.store = store if store is not None else CPUBackend()
        self.chunk_size = chunk_size
        self.default_ttl = default_ttl
        self._descriptors: dict[str, CacheDescriptor] = {}
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------
    @staticmethod
    def _normalize_payloads(payloads: PayloadsByWorker) -> dict[int, list[KVPayload]]:
        if isinstance(payloads, Mapping):
            return {int(w): list(p) for w, p in payloads.items()}
        return {0: list(payloads)}

    def register(
        self,
        token_ids: Sequence[int],
        payloads: PayloadsByWorker,
        *,
        cache_id: Optional[str] = None,
        ttl: Optional[float] = None,
        pin: bool = False,
        name: Optional[str] = None,
    ) -> CacheDescriptor:
        """Register KV for ``token_ids`` under an explicit cache id.

        ``payloads`` is one :class:`KVPayload` per chunk (single worker), or a
        ``{worker_id: [payload, ...]}`` mapping for tensor-parallel shards. The
        number of payload chunks must equal ``ceil(len(token_ids)/chunk_size)``.
        """
        token_chunks = chunk_token_ids(token_ids, self.chunk_size)
        by_worker = self._normalize_payloads(payloads)
        for worker_id, chunks in by_worker.items():
            if len(chunks) != len(token_chunks):
                raise ValueError(
                    f"worker {worker_id}: got {len(chunks)} payload chunks, "
                    f"expected {len(token_chunks)}"
                )

        cid = cache_id or CacheDescriptor.new_id()
        num_layers = 0
        for worker_id, chunks in by_worker.items():
            for idx, payload in enumerate(chunks):
                num_layers = max(num_layers, payload.num_layers)
                self.store.put(ChunkKey(cid, idx, worker_id), payload, pin=pin)

        descriptor = self._build_descriptor(
            cid, token_chunks, num_layers, tuple(sorted(by_worker)), ttl, pin, name
        )
        with self._lock:
            self._descriptors[cid] = descriptor
        return descriptor

    def _build_descriptor(
        self,
        cache_id: str,
        token_chunks: list[list[int]],
        num_layers: int,
        worker_ids: tuple[int, ...],
        ttl: Optional[float],
        pin: bool,
        name: Optional[str],
    ) -> CacheDescriptor:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        now = time.time()
        num_tokens = sum(len(c) for c in token_chunks)
        return CacheDescriptor(
            cache_id=cache_id,
            num_tokens=num_tokens,
            chunk_size=self.chunk_size,
            num_layers=num_layers,
            worker_ids=worker_ids,
            created_at=now,
            expires_at=(now + effective_ttl) if effective_ttl is not None else None,
            pinned=pin,
            chunk_fingerprints=[token_fingerprint(c) for c in token_chunks],
            name=name,
        )

    # -- low-level building blocks (used by connectors) --------------------
    def put_chunk(
        self,
        cache_id: str,
        chunk_index: int,
        payload: KVPayload,
        *,
        worker_id: int = 0,
        pin: bool = False,
    ) -> None:
        self.store.put(ChunkKey(cache_id, chunk_index, worker_id), payload, pin=pin)

    def commit(
        self,
        cache_id: str,
        token_ids: Sequence[int],
        num_layers: int,
        *,
        worker_ids: Sequence[int] = (0,),
        ttl: Optional[float] = None,
        pin: bool = False,
        name: Optional[str] = None,
    ) -> CacheDescriptor:
        """Publish a descriptor for chunks already written via :meth:`put_chunk`."""
        token_chunks = chunk_token_ids(token_ids, self.chunk_size)
        descriptor = self._build_descriptor(
            cache_id, token_chunks, num_layers, tuple(sorted(worker_ids)), ttl, pin, name
        )
        with self._lock:
            self._descriptors[cache_id] = descriptor
        return descriptor

    # -- lookup / load -----------------------------------------------------
    def _live_descriptor(self, cache_id: str) -> CacheDescriptor:
        with self._lock:
            descriptor = self._descriptors.get(cache_id)
        if descriptor is None:
            raise CacheNotFoundError(cache_id)
        if descriptor.is_expired():
            self.delete(cache_id)
            raise CacheExpiredError(cache_id)
        return descriptor

    def get(self, cache_id: str) -> CacheDescriptor:
        return self._live_descriptor(cache_id)

    def exists(self, cache_id: str) -> bool:
        try:
            self._live_descriptor(cache_id)
            return True
        except (CacheNotFoundError, CacheExpiredError):
            return False

    def lookup(
        self,
        cache_id: str,
        request_tokens: Optional[Sequence[int]] = None,
    ) -> LookupResult:
        """Return how much of ``cache_id`` is reusable for a request.

        With ``request_tokens`` the engine verifies the prompt is a real prefix of
        the registered tokens (chunk fingerprints) and truncates reuse at the first
        divergence. Without them it trusts the caller and reports full coverage.
        """
        descriptor = self._live_descriptor(cache_id)
        if request_tokens is None:
            matched = descriptor.num_chunks
        else:
            matched = matched_chunk_count(request_tokens, descriptor)

        if matched == descriptor.num_chunks:
            reusable = descriptor.num_tokens
            full = True
        else:
            reusable = matched * descriptor.chunk_size
            full = False

        descriptor.hits += 1
        descriptor.last_used_at = time.time()
        return LookupResult(cache_id, reusable, matched, full)

    def load(
        self,
        cache_id: str,
        *,
        num_chunks: Optional[int] = None,
        worker_id: int = 0,
    ) -> list[KVPayload]:
        """Fetch the ordered KV chunk payloads for a (prefix of a) cache."""
        descriptor = self._live_descriptor(cache_id)
        count = descriptor.num_chunks if num_chunks is None else min(num_chunks, descriptor.num_chunks)
        out: list[KVPayload] = []
        for idx in range(count):
            payload = self.store.get(ChunkKey(cache_id, idx, worker_id))
            if payload is None:
                raise CacheMismatchError(
                    f"cache {cache_id!r} chunk {idx} (worker {worker_id}) missing from store"
                )
            out.append(payload)
        return out

    # -- lifecycle ---------------------------------------------------------
    def delete(self, cache_id: str) -> None:
        with self._lock:
            descriptor = self._descriptors.pop(cache_id, None)
        if descriptor is None:
            return
        for worker_id in descriptor.worker_ids:
            for idx in range(descriptor.num_chunks):
                self.store.remove(ChunkKey(cache_id, idx, worker_id))

    def extend_ttl(self, cache_id: str, ttl: float) -> CacheDescriptor:
        descriptor = self._live_descriptor(cache_id)
        descriptor.expires_at = time.time() + ttl
        return descriptor

    def pin(self, cache_id: str) -> None:
        descriptor = self._live_descriptor(cache_id)
        descriptor.pinned = True
        for worker_id in descriptor.worker_ids:
            for idx in range(descriptor.num_chunks):
                self.store.pin(ChunkKey(cache_id, idx, worker_id))

    def unpin(self, cache_id: str) -> None:
        descriptor = self._live_descriptor(cache_id)
        descriptor.pinned = False
        for worker_id in descriptor.worker_ids:
            for idx in range(descriptor.num_chunks):
                self.store.unpin(ChunkKey(cache_id, idx, worker_id))

    def list(self) -> list[CacheDescriptor]:
        self.sweep_expired()
        with self._lock:
            return list(self._descriptors.values())

    def sweep_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [c for c in self._descriptors.values() if c.is_expired(now)]
        for descriptor in expired:
            self.delete(descriptor.cache_id)
        return len(expired)

    def stats(self):
        return self.store.stats()
