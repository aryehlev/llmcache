"""Explicit KV caching — offline walkthrough (no torch / GPU needed).

Run: python examples/quickstart.py

Shows the engine + storage tiers directly, then dry-runs the vLLM connector's
save/load lifecycle with the in-memory transfer adapter.
"""

import tempfile

from llmcache import CPUBackend, DiskBackend, ExplicitKVCache, KVPayload, TieredBackend
from llmcache.connectors import (
    DictKVTransfer,
    ExplicitLMCacheConnector,
    KVConnectorRole,
)


def kv(n_chunks, size=64, layers=4):
    # Stand-in for serialized KV; real deployments produce these from tensors.
    return [KVPayload(b"\x00" * size, num_tokens=0, num_layers=layers) for _ in range(n_chunks)]


def engine_demo():
    print("=== engine: register / lookup / load ===")
    with tempfile.TemporaryDirectory() as d:
        store = TieredBackend(CPUBackend(capacity_bytes=1 << 20), DiskBackend(d))
        engine = ExplicitKVCache(store, chunk_size=8)

        prompt = list(range(24))  # 3 chunks
        engine.register(prompt, kv(3), cache_id="handbook-v1", ttl=3600, pin=True, name="handbook")

        # exact reuse
        print("full reuse:", engine.lookup("handbook-v1", prompt))
        # request that shares only the first 2 chunks
        diverged = list(range(16)) + [999] * 8
        print("partial reuse:", engine.lookup("handbook-v1", diverged))
        print("loaded chunks:", len(engine.load("handbook-v1")))
        print("store stats:", engine.stats())


def connector_demo():
    print("\n=== vLLM connector: save then reuse (dry run) ===")
    engine = ExplicitKVCache(chunk_size=8)
    transfer = DictKVTransfer(block_size=4, chunk_size=8, num_layers=4)
    conn = ExplicitLMCacheConnector(role=KVConnectorRole.SCHEDULER, engine=engine, transfer=transfer)

    class Req:
        def __init__(self, rid, toks, params):
            self.request_id, self.prompt_token_ids, self.kv_transfer_params = rid, toks, params

    class Blocks:
        def __init__(self, ids):
            self.ids = ids

        def get_block_ids(self):
            return [self.ids]

    tokens = list(range(16))
    transfer.blocks.update({0: b"AAAA", 1: b"BBBB", 2: b"CCCC", 3: b"DDDD"})

    save = Req("r1", tokens, {"cache_id": "kb", "save": True, "pin": True})
    conn.update_state_after_alloc(save, Blocks([0, 1, 2, 3]), 0)
    conn.bind_connector_metadata(conn.build_connector_meta())
    conn.wait_for_save()
    conn.clear_connector_metadata()
    print("saved cache:", engine.get("kb").cache_id, "chunks:", engine.get("kb").num_chunks)

    load = Req("r2", tokens, {"cache_id": "kb"})
    external, _ = conn.get_num_new_matched_tokens(load, 0)
    print("reusable tokens for next request:", external)
    conn.update_state_after_alloc(load, Blocks([8, 9, 10, 11]), external)
    conn.bind_connector_metadata(conn.build_connector_meta())
    conn.start_load_kv()
    print("KV loaded into fresh blocks:", {b: transfer.blocks[b] for b in (8, 9, 10, 11)})


if __name__ == "__main__":
    engine_demo()
    connector_demo()
