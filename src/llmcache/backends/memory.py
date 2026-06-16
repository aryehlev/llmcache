"""In-memory reference backend.

Useful for tests and local development. It is exactly ``LocalPrefixCacheBackend``
with no engine integration: prefixes are tracked in process and rendered in the
OpenAI-compatible chat format.
"""

from __future__ import annotations

from ..backend import LocalPrefixCacheBackend


class MemoryBackend(LocalPrefixCacheBackend):
    name = "memory"
