"""Google Gemini backend — native explicit caching.

Gemini is the canonical *native* explicit cache: ``client.caches.create(...)``
returns a ``cachedContents/<id>`` object that you later reference via
``generate_content(config={"cached_content": name})``. The cached content is
stored server-side and billed at a discount; you never resend it.

This backend reuses llmcache's local bookkeeping (handles, TTL, listing) and, when
a ``genai`` client is supplied, mirrors each operation to the live API. Without a
client it runs offline with a synthetic ``cachedContents/`` id so request shaping
can still be exercised.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..backend import LocalPrefixCacheBackend
from ..errors import BackendError
from ..types import CacheEntry, MessageLike, coerce_messages


def _to_gemini_role(role: str) -> str:
    return "model" if role in ("assistant", "model") else "user"


def _to_contents(messages: Sequence) -> list[dict[str, Any]]:
    return [
        {"role": _to_gemini_role(m.role), "parts": [{"text": m.content}]}
        for m in messages
    ]


class GeminiBackend(LocalPrefixCacheBackend):
    name = "gemini"

    def __init__(self, client: Any = None, *, default_ttl: Optional[float] = None) -> None:
        super().__init__(default_ttl=default_ttl)
        self.client = client

    def supports_native_explicit_cache(self) -> bool:
        return True

    def on_create(self, entry: CacheEntry) -> None:
        if self.client is None:
            entry.native_id = f"cachedContents/{entry.id}"
            return
        config: dict[str, Any] = {"contents": _to_contents(entry.spec.content)}
        if entry.spec.system:
            config["system_instruction"] = entry.spec.system
        if entry.spec.tools:
            config["tools"] = entry.spec.tools
        if entry.handle.expires_at is not None and entry.spec.ttl is not None:
            config["ttl"] = f"{int(entry.spec.ttl)}s"
        if entry.spec.name:
            config["display_name"] = entry.spec.name
        try:
            cache = self.client.caches.create(model=entry.spec.model, config=config)
        except Exception as exc:  # pragma: no cover - depends on live service
            raise BackendError(f"Gemini cache create failed: {exc}") from exc
        entry.native_id = getattr(cache, "name", None) or str(cache)

    def on_delete(self, entry: CacheEntry) -> None:
        if self.client is not None and entry.native_id:
            try:
                self.client.caches.delete(name=entry.native_id)
            except Exception as exc:  # pragma: no cover - depends on live service
                raise BackendError(f"Gemini cache delete failed: {exc}") from exc

    def build_request(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> dict[str, Any]:
        entry = self.get(cache_id)
        new_messages = coerce_messages(messages)
        config: dict[str, Any] = {"cached_content": entry.native_id}
        config.update(params.pop("config", {}))
        request: dict[str, Any] = {
            "model": entry.spec.model,
            "contents": _to_contents(new_messages),
            "config": config,
        }
        request.update(params)
        self._record_hit(entry)
        return request

    def complete(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> Any:
        if self.client is None:
            raise BackendError("GeminiBackend has no client configured")
        request = self.build_request(cache_id, messages, **params)
        try:
            return self.client.models.generate_content(**request)
        except Exception as exc:  # pragma: no cover - depends on live service
            raise BackendError(f"Gemini completion failed: {exc}") from exc
