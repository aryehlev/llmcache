"""Core data types shared across backends.

The model here is deliberately small. Explicit caching is about declaring a
*prefix* of a request — typically a system instruction plus some large, reused
context (documents, few-shot examples, a tool catalogue) — once, getting a
handle back, and then referencing that handle on many subsequent requests so the
prefix does not have to be re-sent or re-computed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Union


@dataclass
class Message:
    """A single chat message used to describe cached content."""

    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


MessageLike = Union[Message, Mapping[str, Any]]


def coerce_message(m: MessageLike) -> Message:
    if isinstance(m, Message):
        return m
    if isinstance(m, Mapping):
        try:
            return Message(role=str(m["role"]), content=str(m["content"]))
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"message mapping missing key: {exc}") from exc
    raise TypeError(f"cannot coerce {type(m)!r} into a Message")


def coerce_messages(messages: Sequence[MessageLike]) -> list[Message]:
    return [coerce_message(m) for m in messages]


class CacheStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass
class CacheSpec:
    """Describes content that should be cached.

    ``content`` is the reusable prefix. ``system`` is broken out because several
    platforms treat the system instruction specially (Anthropic ``system`` block,
    Gemini ``systemInstruction``). ``ttl`` is in seconds; ``None`` means "use the
    backend default".
    """

    model: str
    content: list[Message] = field(default_factory=list)
    system: Optional[str] = None
    tools: Optional[list[Any]] = None
    ttl: Optional[float] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = coerce_messages(self.content)

    def approx_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) used for usage metrics only."""
        chars = len(self.system or "")
        for m in self.content:
            chars += len(m.content)
        return chars // 4


@dataclass
class CacheUsage:
    hits: int = 0
    tokens_cached: int = 0
    last_used_at: Optional[float] = None


@dataclass
class CacheHandle:
    """A portable reference to a cache entry returned to callers."""

    id: str
    backend: str
    model: str
    created_at: float
    expires_at: Optional[float]
    name: Optional[str] = None

    @staticmethod
    def new(backend: str, model: str, ttl: Optional[float], name: Optional[str]) -> "CacheHandle":
        now = time.time()
        return CacheHandle(
            id=uuid.uuid4().hex,
            backend=backend,
            model=model,
            created_at=now,
            expires_at=(now + ttl) if ttl is not None else None,
            name=name,
        )


@dataclass
class CacheEntry:
    """Full server-side record for a cache entry."""

    handle: CacheHandle
    spec: CacheSpec
    status: CacheStatus = CacheStatus.ACTIVE
    usage: CacheUsage = field(default_factory=CacheUsage)
    # Some platforms (e.g. Gemini) mint their own id; we keep it for native calls.
    native_id: Optional[str] = None

    @property
    def id(self) -> str:
        return self.handle.id

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.handle.expires_at is None:
            return False
        return (now or time.time()) >= self.handle.expires_at
