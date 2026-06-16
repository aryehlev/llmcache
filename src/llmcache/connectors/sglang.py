"""SGLang adapter for the explicit KV cache.

SGLang's hierarchical cache (HiCache) can offload KV to a host-side store. Rather
than reimplement RadixAttention, this adapter exposes the explicit engine through
the small ``store`` / ``retrieve`` surface a host KV backend needs, keyed by
explicit cache id + chunk. It shares the same :class:`ExplicitKVCache` and
:class:`KVTransfer` machinery as the vLLM connector, so a registered cache is
reusable from either engine.

This is a thin, framework-light integration point; wire it into SGLang's host
cache hook for your deployment.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..engine import ExplicitKVCache
from ..payload import KVPayload
from .transfer import KVTransfer


class SGLangExplicitCache:
    def __init__(
        self,
        engine: Optional[ExplicitKVCache] = None,
        transfer: Optional[KVTransfer] = None,
    ) -> None:
        self.engine = engine if engine is not None else ExplicitKVCache()
        self.transfer = transfer

    def store(
        self,
        cache_id: str,
        token_ids: Sequence[int],
        block_ids: Sequence[int],
        *,
        ttl: Optional[float] = None,
        pin: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if self.transfer is None:
            raise RuntimeError("SGLangExplicitCache needs a KVTransfer to store KV")
        num_chunks = -(-len(token_ids) // self.engine.chunk_size)
        payloads = self.transfer.save_from(block_ids, num_chunks)
        for idx, payload in enumerate(payloads):
            self.engine.put_chunk(cache_id, idx, payload, pin=pin)
        self.engine.commit(
            cache_id,
            token_ids,
            num_layers=self.transfer.num_layers,
            ttl=ttl,
            pin=pin,
            name=name,
        )

    def retrieve(
        self,
        cache_id: str,
        block_ids: Sequence[int],
        request_tokens: Optional[Sequence[int]] = None,
    ) -> int:
        """Load cached KV into ``block_ids``; return the reused token count."""
        if self.transfer is None:
            raise RuntimeError("SGLangExplicitCache needs a KVTransfer to retrieve KV")
        if not self.engine.exists(cache_id):
            return 0
        result = self.engine.lookup(cache_id, request_tokens)
        payloads: list[KVPayload] = self.engine.load(cache_id, num_chunks=result.num_chunks)
        self.transfer.load_into(block_ids, payloads)
        return result.num_tokens
