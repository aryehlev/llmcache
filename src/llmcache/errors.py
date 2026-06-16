"""Exception hierarchy for llmcache."""

from __future__ import annotations


class CacheError(Exception):
    """Base class for all llmcache errors."""


class CacheNotFoundError(CacheError):
    """Raised when a cache id is not known to the backend."""

    def __init__(self, cache_id: str):
        self.cache_id = cache_id
        super().__init__(f"No cache entry found for id {cache_id!r}")


class CacheExpiredError(CacheError):
    """Raised when an operation references a cache whose TTL has elapsed."""

    def __init__(self, cache_id: str):
        self.cache_id = cache_id
        super().__init__(f"Cache entry {cache_id!r} has expired")


class UnsupportedOperationError(CacheError):
    """Raised when a backend cannot perform the requested operation."""


class BackendError(CacheError):
    """Raised when an underlying serving platform / SDK call fails."""
