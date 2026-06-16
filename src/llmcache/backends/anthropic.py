"""Anthropic backend.

Anthropic prompt caching is *explicit at the request level*: you place a
``cache_control`` breakpoint at the end of the static prefix, and everything
before it (tools, then system, then leading messages) is cached. There is no
separate cache object — the breakpoint plus the resent prefix is the mechanism.

llmcache stores the prefix and renders the Anthropic Messages API shape, placing
a single breakpoint at the end of the cached content. TTL maps to Anthropic's two
supported durations: ``<= 5 min`` -> ``"5m"``, longer -> ``"1h"`` (the latter
requires the extended-cache beta header on the client).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..backend import LocalPrefixCacheBackend
from ..errors import BackendError
from ..types import CacheEntry, MessageLike, coerce_messages


def _ttl_label(ttl: Optional[float]) -> str:
    return "1h" if (ttl is not None and ttl > 300) else "5m"


class AnthropicBackend(LocalPrefixCacheBackend):
    name = "anthropic"

    def __init__(self, client: Any = None, *, default_ttl: Optional[float] = None) -> None:
        super().__init__(default_ttl=default_ttl)
        self.client = client

    def _cache_control(self, entry: CacheEntry) -> dict[str, Any]:
        ttl = entry.handle.expires_at
        label = "5m"
        if entry.spec.ttl is not None:
            label = _ttl_label(entry.spec.ttl)
        elif self.default_ttl is not None:
            label = _ttl_label(self.default_ttl)
        cc: dict[str, Any] = {"type": "ephemeral"}
        if label != "5m":
            cc["ttl"] = label
        return cc

    def build_request(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> dict[str, Any]:
        entry = self.get(cache_id)
        new_messages = coerce_messages(messages)
        cache_control = self._cache_control(entry)

        request: dict[str, Any] = {"model": entry.spec.model}

        # System block(s). The breakpoint lands here only if there are no cached
        # messages after it.
        system_gets_breakpoint = not entry.spec.content
        if entry.spec.system:
            sys_block: dict[str, Any] = {"type": "text", "text": entry.spec.system}
            if system_gets_breakpoint:
                sys_block["cache_control"] = cache_control
            request["system"] = [sys_block]

        if entry.spec.tools:
            request["tools"] = entry.spec.tools

        msgs: list[dict[str, Any]] = []
        cached = entry.spec.content
        for i, m in enumerate(cached):
            block: dict[str, Any] = {"type": "text", "text": m.content}
            if i == len(cached) - 1:  # breakpoint at end of cached prefix
                block["cache_control"] = cache_control
            msgs.append({"role": m.role, "content": [block]})
        msgs.extend(
            {"role": m.role, "content": [{"type": "text", "text": m.content}]}
            for m in new_messages
        )
        request["messages"] = msgs
        request.update(params)
        self._record_hit(entry)
        return request

    def complete(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> Any:
        if self.client is None:
            raise BackendError("AnthropicBackend has no client configured")
        request = self.build_request(cache_id, messages, **params)
        try:
            return self.client.messages.create(**request)
        except Exception as exc:  # pragma: no cover - depends on live service
            raise BackendError(f"Anthropic completion failed: {exc}") from exc
