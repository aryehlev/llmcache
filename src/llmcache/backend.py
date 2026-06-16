"""Backend abstraction.

A *backend* maps llmcache's platform-neutral explicit-cache operations onto a
specific serving platform. Platforms fall into two families:

* **Native explicit cache** (e.g. Google Gemini ``cachedContents``): the platform
  has a first-class API to create a cache object and reference it by id later.
  ``supports_native_explicit_cache`` returns ``True``.

* **Prefix / prompt cache** (e.g. vLLM, SGLang, TGI automatic prefix caching;
  OpenAI implicit prompt caching; Anthropic ``cache_control`` breakpoints): there
  is no "create cache" call. Instead the cached prefix must be prepended to each
  request and the engine reuses previously computed KV blocks. For these,
  llmcache keeps the prefix and TTL bookkeeping locally and re-materialises the
  prefix into every request via :meth:`build_request`. ``LocalPrefixCacheBackend``
  implements this shared behaviour.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from .errors import CacheExpiredError, CacheNotFoundError
from .types import (
    CacheEntry,
    CacheHandle,
    CacheSpec,
    CacheStatus,
    CacheUsage,
    MessageLike,
    coerce_messages,
)


class CacheBackend(ABC):
    """Interface every serving-platform adapter implements."""

    #: Stable identifier used by the registry, e.g. ``"vllm"``.
    name: str = "base"

    @abstractmethod
    def supports_native_explicit_cache(self) -> bool:
        """Whether the platform exposes a first-class cache-object API."""

    @abstractmethod
    def create(self, spec: CacheSpec) -> CacheEntry:
        """Create a cache entry from ``spec`` and return the stored record."""

    @abstractmethod
    def get(self, cache_id: str) -> CacheEntry:
        """Return the entry for ``cache_id`` or raise :class:`CacheNotFoundError`."""

    @abstractmethod
    def list(self) -> list[CacheEntry]:
        """Return all live entries."""

    @abstractmethod
    def delete(self, cache_id: str) -> None:
        """Delete an entry. Idempotent."""

    @abstractmethod
    def extend(self, cache_id: str, ttl: float) -> CacheEntry:
        """Reset the entry's TTL to ``ttl`` seconds from now."""

    @abstractmethod
    def build_request(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> dict[str, Any]:
        """Return a platform-native request payload that references the cache.

        ``messages`` are the *new* turn(s) appended after the cached prefix.
        """


class LocalPrefixCacheBackend(CacheBackend):
    """Shared implementation for prefix/prompt-cache platforms.

    Lifecycle (create/get/list/delete/extend) is tracked in-process. The cached
    prefix is rebuilt into each request by :meth:`build_request`, which subclasses
    customise by overriding :meth:`render_messages` (and optionally the system /
    tools handling) to match their wire format.
    """

    name = "local-prefix"

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self.default_ttl = default_ttl
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def supports_native_explicit_cache(self) -> bool:
        return False

    # -- lifecycle ---------------------------------------------------------
    def create(self, spec: CacheSpec) -> CacheEntry:
        ttl = spec.ttl if spec.ttl is not None else self.default_ttl
        handle = CacheHandle.new(self.name, spec.model, ttl, spec.name)
        entry = CacheEntry(
            handle=handle,
            spec=spec,
            usage=CacheUsage(tokens_cached=spec.approx_tokens()),
        )
        self.on_create(entry)
        with self._lock:
            self._entries[entry.id] = entry
        return entry

    def get(self, cache_id: str) -> CacheEntry:
        with self._lock:
            entry = self._entries.get(cache_id)
        if entry is None:
            raise CacheNotFoundError(cache_id)
        if entry.is_expired():
            entry.status = CacheStatus.EXPIRED
            raise CacheExpiredError(cache_id)
        return entry

    def list(self) -> list[CacheEntry]:
        now = time.time()
        with self._lock:
            live = []
            for entry in self._entries.values():
                if entry.is_expired(now):
                    entry.status = CacheStatus.EXPIRED
                    continue
                live.append(entry)
        return live

    def delete(self, cache_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(cache_id, None)
        if entry is not None:
            entry.status = CacheStatus.DELETED
            self.on_delete(entry)

    def extend(self, cache_id: str, ttl: float) -> CacheEntry:
        entry = self.get(cache_id)
        entry.handle.expires_at = time.time() + ttl
        entry.status = CacheStatus.ACTIVE
        return entry

    # -- hooks for subclasses ---------------------------------------------
    def on_create(self, entry: CacheEntry) -> None:
        """Override to warm the engine (e.g. issue a priming request)."""

    def on_delete(self, entry: CacheEntry) -> None:
        """Override to release platform resources."""

    def render_messages(self, entry: CacheEntry, messages: list) -> list[dict[str, Any]]:
        """Default OpenAI-compatible message rendering: prefix then new turns."""
        out: list[dict[str, Any]] = []
        if entry.spec.system:
            out.append({"role": "system", "content": entry.spec.system})
        out.extend(m.to_dict() for m in entry.spec.content)
        out.extend(m.to_dict() for m in messages)
        return out

    # -- request construction ---------------------------------------------
    def build_request(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> dict[str, Any]:
        entry = self.get(cache_id)
        new_messages = coerce_messages(messages)
        request: dict[str, Any] = {
            "model": entry.spec.model,
            "messages": self.render_messages(entry, new_messages),
        }
        if entry.spec.tools:
            request["tools"] = entry.spec.tools
        request.update(params)
        self._record_hit(entry)
        return request

    def _record_hit(self, entry: CacheEntry) -> None:
        entry.usage.hits += 1
        entry.usage.last_used_at = time.time()
