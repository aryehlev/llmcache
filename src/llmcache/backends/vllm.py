"""vLLM backend.

vLLM has no "create cache" endpoint; it has **automatic prefix caching** (start
the server with ``--enable-prefix-caching``). The first request that contains a
given prefix computes and stores its KV blocks; later requests sharing that
prefix reuse them. llmcache turns that into an *explicit* workflow:

* :meth:`create` optionally fires a tiny "priming" request so the prefix's KV
  blocks are resident before real traffic arrives (cache warming).
* :meth:`build_request` re-attaches the prefix to every request so the engine can
  hit the cached blocks. The output is an OpenAI-compatible chat payload, which is
  exactly what vLLM's ``/v1/chat/completions`` server consumes.

Pass ``client`` (any object exposing ``client.chat.completions.create(**kwargs)``,
such as the ``openai`` SDK pointed at the vLLM server) to enable warming and the
:meth:`complete` convenience. Everything works offline without a client.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..backend import LocalPrefixCacheBackend
from ..errors import BackendError
from ..types import CacheEntry, MessageLike


class VLLMBackend(LocalPrefixCacheBackend):
    name = "vllm"

    def __init__(
        self,
        client: Any = None,
        *,
        default_ttl: Optional[float] = None,
        warm_on_create: bool = True,
    ) -> None:
        super().__init__(default_ttl=default_ttl)
        self.client = client
        self.warm_on_create = warm_on_create

    def on_create(self, entry: CacheEntry) -> None:
        if not (self.warm_on_create and self.client is not None):
            return
        # Prime the engine so the prefix's KV blocks are resident. max_tokens=1
        # keeps the warm-up cheap; the completion itself is discarded.
        request = {
            "model": entry.spec.model,
            "messages": self.render_messages(entry, []),
            "max_tokens": 1,
        }
        if entry.spec.tools:
            request["tools"] = entry.spec.tools
        try:
            self.client.chat.completions.create(**request)
        except Exception as exc:  # pragma: no cover - depends on live server
            raise BackendError(f"vLLM cache warm-up failed: {exc}") from exc

    def complete(
        self, cache_id: str, messages: Sequence[MessageLike], **params: Any
    ) -> Any:
        """Build the request and run it against the configured vLLM server."""
        if self.client is None:
            raise BackendError("VLLMBackend has no client configured")
        request = self.build_request(cache_id, messages, **params)
        try:
            return self.client.chat.completions.create(**request)
        except Exception as exc:  # pragma: no cover - depends on live server
            raise BackendError(f"vLLM completion failed: {exc}") from exc
