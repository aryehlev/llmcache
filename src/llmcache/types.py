"""Core data types for the explicit KV cache.

Unlike LMCache, which keys stored KV by a rolling hash of the *token content* and
matches new requests by scanning for the longest shared prefix, llmcache keys KV
by an **explicit cache id** (a handle the application owns). A handle covers an
ordered, contiguous run of tokens split into fixed-size chunks; each chunk is one
stored object. Lookups are O(1) by id, hits are deterministic, and entries can be
pinned so they are never silently evicted.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class ChunkKey:
    """Storage key for a single KV chunk belonging to a cache handle."""

    cache_id: str
    chunk_index: int
    # worker_id distinguishes tensor-parallel shards that hold different KV.
    worker_id: int = 0

    def __str__(self) -> str:
        return f"{self.cache_id}@{self.worker_id}#{self.chunk_index}"


def token_fingerprint(tokens: Sequence[int]) -> str:
    """Stable short fingerprint of a token run, used to validate prefix match."""
    h = hashlib.blake2b(digest_size=16)
    for t in tokens:
        h.update(int(t).to_bytes(4, "little", signed=True))
    return h.hexdigest()


@dataclass
class CacheDescriptor:
    """Metadata the engine keeps for one registered explicit cache."""

    cache_id: str
    num_tokens: int
    chunk_size: int
    num_layers: int
    worker_ids: tuple[int, ...]
    created_at: float
    expires_at: Optional[float]
    pinned: bool
    # Per-chunk fingerprint of the covered tokens, so a request can be verified
    # to actually share this prefix before its KV is reused.
    chunk_fingerprints: list[str] = field(default_factory=list)
    name: Optional[str] = None
    hits: int = 0
    last_used_at: Optional[float] = None

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_fingerprints)

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time.time()) >= self.expires_at

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


def chunk_token_ids(tokens: Sequence[int], chunk_size: int) -> list[list[int]]:
    """Split a token sequence into fixed-size chunks (last may be short)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [list(tokens[i : i + chunk_size]) for i in range(0, len(tokens), chunk_size)]


def matched_chunk_count(
    request_tokens: Sequence[int],
    descriptor: "CacheDescriptor",
) -> int:
    """How many cached chunks are a valid prefix of ``request_tokens``.

    Returns the count of leading chunks whose token fingerprints match. Stops at
    the first divergence (explicit caches are prefix caches: a partial mismatch
    truncates reuse rather than failing).
    """
    chunks = chunk_token_ids(request_tokens, descriptor.chunk_size)
    matched = 0
    for i, fp in enumerate(descriptor.chunk_fingerprints):
        if i >= len(chunks):
            break
        # A short final cached chunk can only match if the request has at least
        # that many tokens in the corresponding position.
        if token_fingerprint(chunks[i]) != fp:
            break
        matched += 1
    return matched
