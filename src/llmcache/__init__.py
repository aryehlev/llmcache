"""llmcache — a unified explicit-caching control plane for LLM serving platforms.

Declare reusable prompt content (system instructions, documents, few-shot
examples, tool catalogues) once, get a portable handle, and reference it across
many requests. The same API drives native explicit caches (Gemini) and
prefix/prompt caches (vLLM, SGLang, OpenAI, Anthropic).
"""

from __future__ import annotations

from .backend import CacheBackend, LocalPrefixCacheBackend
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.memory import MemoryBackend
from .backends.openai import OpenAIBackend
from .backends.sglang import SGLangBackend
from .backends.vllm import VLLMBackend
from .errors import (
    BackendError,
    CacheError,
    CacheExpiredError,
    CacheNotFoundError,
    UnsupportedOperationError,
)
from .manager import ExplicitCache
from .registry import available_backends, get_backend, register_backend
from .types import (
    CacheEntry,
    CacheHandle,
    CacheSpec,
    CacheStatus,
    CacheUsage,
    Message,
)

__version__ = "0.1.0"

__all__ = [
    "ExplicitCache",
    "CacheBackend",
    "LocalPrefixCacheBackend",
    "MemoryBackend",
    "VLLMBackend",
    "SGLangBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "GeminiBackend",
    "get_backend",
    "register_backend",
    "available_backends",
    "CacheSpec",
    "CacheHandle",
    "CacheEntry",
    "CacheStatus",
    "CacheUsage",
    "Message",
    "CacheError",
    "CacheNotFoundError",
    "CacheExpiredError",
    "UnsupportedOperationError",
    "BackendError",
    "__version__",
]
