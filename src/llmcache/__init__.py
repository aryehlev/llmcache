"""llmcache — explicit KV caching for LLM serving engines.

An LMCache-style KV offload/reuse layer, but **explicit**: applications register
KV under a cache id and reference it deliberately, instead of relying on automatic
token-hash prefix matching. Deterministic hits, pinning, offline warming, and
cross-request sharing with no false matches.

Layers:

* :class:`ExplicitKVCache` — the control plane (register / lookup / load / pin /
  TTL), tensor- and storage-agnostic.
* ``store`` — :class:`CPUBackend`, :class:`DiskBackend`, :class:`TieredBackend`.
* ``connectors`` — :class:`ExplicitLMCacheConnector` (vLLM V1) and
  :class:`SGLangExplicitCache`, plus :class:`KVTransfer` adapters.
"""

from __future__ import annotations

from .engine import ExplicitKVCache, LookupResult
from .errors import (
    CacheError,
    CacheExpiredError,
    CacheMismatchError,
    CacheNotFoundError,
    ConnectorError,
    SerializationError,
    StorageFullError,
)
from .payload import KVPayload, KVSerializer, RawSerializer, TorchSerializer
from .store import CPUBackend, DiskBackend, StorageBackend, StoreStats, TieredBackend
from .types import (
    CacheDescriptor,
    ChunkKey,
    chunk_token_ids,
    matched_chunk_count,
    token_fingerprint,
)

__version__ = "0.2.0"

__all__ = [
    "ExplicitKVCache",
    "LookupResult",
    "CacheDescriptor",
    "ChunkKey",
    "chunk_token_ids",
    "matched_chunk_count",
    "token_fingerprint",
    "KVPayload",
    "KVSerializer",
    "RawSerializer",
    "TorchSerializer",
    "StorageBackend",
    "StoreStats",
    "CPUBackend",
    "DiskBackend",
    "TieredBackend",
    "CacheError",
    "CacheNotFoundError",
    "CacheExpiredError",
    "CacheMismatchError",
    "StorageFullError",
    "SerializationError",
    "ConnectorError",
    "__version__",
]
