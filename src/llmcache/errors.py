"""Exception hierarchy for llmcache."""

from __future__ import annotations


class CacheError(Exception):
    """Base class for all llmcache errors."""


class CacheNotFoundError(CacheError):
    """Raised when a cache id / handle is not known to the engine."""

    def __init__(self, cache_id: str):
        self.cache_id = cache_id
        super().__init__(f"No cache registered for id {cache_id!r}")


class CacheExpiredError(CacheError):
    """Raised when an operation references a cache whose TTL has elapsed."""

    def __init__(self, cache_id: str):
        self.cache_id = cache_id
        super().__init__(f"Cache {cache_id!r} has expired")


class CacheMismatchError(CacheError):
    """Raised when request tokens do not match the registered cache prefix."""


class StorageFullError(CacheError):
    """Raised when a backend cannot make room for an object (all pinned)."""


class SerializationError(CacheError):
    """Raised when KV (de)serialization fails."""


class ConnectorError(CacheError):
    """Raised for serving-engine connector integration failures."""
