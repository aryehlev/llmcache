"""OpenAI backend.

OpenAI prompt caching is *implicit*: there is no cache-create API and no warming
knob. Caching kicks in automatically for prompts over ~1024 tokens that share a
common prefix, and routing is best-effort. llmcache models this by tracking the
prefix locally and re-attaching it (longest-static-content-first) so the implicit
cache has the best chance of hitting. The emitted payload is a standard
``chat.completions`` request.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..backend import LocalPrefixCacheBackend
from ..errors import BackendError
from ..types import MessageLike


class OpenAIBackend(LocalPrefixCacheBackend):
    name = "openai"

    def __init__(self, client: Any = None, *, default_ttl: Optional[float] = None) -> None:
        # OpenAI evicts cache entries on its own schedule (minutes of inactivity);
        # default_ttl is advisory bookkeeping only.
        super().__init__(default_ttl=default_ttl)
        self.client = client

    def supports_native_explicit_cache(self) -> bool:
        return False

    def complete(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> Any:
        if self.client is None:
            raise BackendError("OpenAIBackend has no client configured")
        request = self.build_request(cache_id, messages, **params)
        try:
            return self.client.chat.completions.create(**request)
        except Exception as exc:  # pragma: no cover - depends on live service
            raise BackendError(f"OpenAI completion failed: {exc}") from exc
