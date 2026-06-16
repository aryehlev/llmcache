"""Explicit caching across platforms with one API.

Run: python examples/quickstart.py
"""

from llmcache import ExplicitCache

# A large, reusable context we don't want to resend on every question.
KNOWLEDGE_BASE = [
    {"role": "user", "content": "Company handbook:\n" + ("policy text ... " * 200)},
    {"role": "assistant", "content": "Understood. I'll answer using the handbook."},
]


def demo(platform: str) -> None:
    cache = ExplicitCache(platform)  # offline; pass client=... to go live
    handle = cache.create(
        model="example-model",
        system="You are an HR assistant. Answer only from the handbook.",
        content=KNOWLEDGE_BASE,
        ttl=3600,
        name="handbook-v1",
    )

    request = cache.build_request(
        handle,
        [{"role": "user", "content": "How many vacation days do I get?"}],
        max_tokens=256,
    )

    print(f"\n=== {platform} (native explicit cache: {cache.native_explicit_cache}) ===")
    print("cache id:", handle.id)
    print("request keys:", sorted(request))
    print("hits:", cache.get(handle).usage.hits)


if __name__ == "__main__":
    for platform in ("vllm", "sglang", "openai", "anthropic", "gemini"):
        demo(platform)
