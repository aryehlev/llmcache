"""SGLang backend.

SGLang caches prefixes with RadixAttention and serves an OpenAI-compatible API.
Behaviour mirrors the vLLM backend: optional warm-up on create, prefix re-attached
on every request.
"""

from __future__ import annotations

from .vllm import VLLMBackend


class SGLangBackend(VLLMBackend):
    name = "sglang"
