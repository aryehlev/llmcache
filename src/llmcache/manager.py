"""The public, platform-neutral explicit-cache control plane."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .backend import CacheBackend
from .errors import BackendError
from .registry import get_backend
from .types import (
    CacheEntry,
    CacheHandle,
    CacheSpec,
    MessageLike,
    coerce_messages,
)


class ExplicitCache:
    """Manage explicit LLM caches against any supported serving platform.

    Construct it with either a backend instance or a platform name::

        cache = ExplicitCache("vllm", client=openai_client)
        cache = ExplicitCache(MemoryBackend())

    Then declare reusable content once and reference it many times::

        handle = cache.create(model="meta-llama/Llama-3.1-8B", content=docs, ttl=600)
        request = cache.build_request(handle, [{"role": "user", "content": "Q?"}])
    """

    def __init__(self, backend: Union[str, CacheBackend], **backend_kwargs: Any) -> None:
        self.backend: CacheBackend = (
            get_backend(backend, **backend_kwargs)
            if isinstance(backend, str)
            else backend
        )

    @property
    def platform(self) -> str:
        return self.backend.name

    @property
    def native_explicit_cache(self) -> bool:
        return self.backend.supports_native_explicit_cache()

    def create(
        self,
        *,
        model: str,
        content: Sequence[MessageLike] = (),
        system: Optional[str] = None,
        tools: Optional[list[Any]] = None,
        ttl: Optional[float] = None,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CacheHandle:
        """Create a cache entry and return its portable handle."""
        spec = CacheSpec(
            model=model,
            content=coerce_messages(content),
            system=system,
            tools=tools,
            ttl=ttl,
            name=name,
            metadata=metadata or {},
        )
        entry = self.backend.create(spec)
        return entry.handle

    @staticmethod
    def _id(ref: Union[str, CacheHandle, CacheEntry]) -> str:
        if isinstance(ref, str):
            return ref
        if isinstance(ref, CacheHandle):
            return ref.id
        if isinstance(ref, CacheEntry):
            return ref.id
        raise TypeError(f"expected cache id/handle/entry, got {type(ref)!r}")

    def get(self, ref: Union[str, CacheHandle, CacheEntry]) -> CacheEntry:
        return self.backend.get(self._id(ref))

    def list(self) -> list[CacheEntry]:
        return self.backend.list()

    def delete(self, ref: Union[str, CacheHandle, CacheEntry]) -> None:
        self.backend.delete(self._id(ref))

    def extend(self, ref: Union[str, CacheHandle, CacheEntry], ttl: float) -> CacheEntry:
        return self.backend.extend(self._id(ref), ttl)

    def build_request(
        self,
        ref: Union[str, CacheHandle, CacheEntry],
        messages: Sequence[MessageLike],
        **params: Any,
    ) -> dict[str, Any]:
        """Return the platform-native request payload referencing the cache."""
        return self.backend.build_request(self._id(ref), messages, **params)

    def complete(
        self,
        ref: Union[str, CacheHandle, CacheEntry],
        messages: Sequence[MessageLike],
        **params: Any,
    ) -> Any:
        """Run a completion through the backend if it supports one."""
        complete = getattr(self.backend, "complete", None)
        if complete is None:
            raise BackendError(f"backend {self.platform!r} has no complete()")
        return complete(self._id(ref), messages, **params)
