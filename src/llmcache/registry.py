"""Backend registry — look up adapters by platform name."""

from __future__ import annotations

from typing import Any, Callable

from .backend import CacheBackend
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.memory import MemoryBackend
from .backends.openai import OpenAIBackend
from .backends.sglang import SGLangBackend
from .backends.vllm import VLLMBackend

_REGISTRY: dict[str, Callable[..., CacheBackend]] = {
    "memory": MemoryBackend,
    "vllm": VLLMBackend,
    "sglang": SGLangBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
}


def register_backend(name: str, factory: Callable[..., CacheBackend]) -> None:
    """Register a custom backend factory under ``name``."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, **kwargs: Any) -> CacheBackend:
    """Instantiate a backend by platform name (e.g. ``"vllm"``)."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; available: {', '.join(available_backends())}"
        ) from None
    return factory(**kwargs)
