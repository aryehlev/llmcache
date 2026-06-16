from llmcache import ExplicitKVCache
from llmcache.connectors import (
    DictKVTransfer,
    ExplicitLMCacheConnector,
    KVConnectorRole,
    SGLangExplicitCache,
)


class FakeBlocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self):
        return [list(self.ids)]  # single kv-cache group


class FakeRequest:
    def __init__(self, request_id, prompt_token_ids, kv_transfer_params):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.kv_transfer_params = kv_transfer_params


def make_connector():
    engine = ExplicitKVCache(chunk_size=8)
    transfer = DictKVTransfer(block_size=4, chunk_size=8, num_layers=4)
    conn = ExplicitLMCacheConnector(
        role=KVConnectorRole.SCHEDULER, engine=engine, transfer=transfer
    )
    return conn, engine, transfer


TOKENS = list(range(16))  # 2 chunks of 8 with chunk_size=8


def test_no_cache_id_means_no_reuse():
    conn, _, _ = make_connector()
    req = FakeRequest("r0", TOKENS, kv_transfer_params={})
    assert conn.get_num_new_matched_tokens(req, 0) == (0, False)


def test_save_then_load_roundtrip_through_connector():
    conn, engine, transfer = make_connector()

    # --- save path: a request creates cache "kb" from its computed blocks ---
    transfer.blocks.update({10: b"AAAA", 11: b"BBBB", 12: b"CCCC", 13: b"DDDD"})
    save_req = FakeRequest(
        "r1", TOKENS, kv_transfer_params={"cache_id": "kb", "save": True, "pin": True}
    )
    # cache doesn't exist yet -> nothing to match
    assert conn.get_num_new_matched_tokens(save_req, 0) == (0, False)
    conn.update_state_after_alloc(save_req, FakeBlocks([10, 11, 12, 13]), 0)
    meta = conn.build_connector_meta()
    conn.bind_connector_metadata(meta)
    conn.wait_for_save()
    conn.clear_connector_metadata()

    assert engine.exists("kb")
    desc = engine.get("kb")
    assert desc.num_chunks == 2 and desc.pinned

    # --- load path: a later request reuses "kb" into fresh blocks ---
    load_req = FakeRequest("r2", TOKENS, kv_transfer_params={"cache_id": "kb"})
    external, is_async = conn.get_num_new_matched_tokens(load_req, 0)
    assert external == 16 and is_async is False
    conn.update_state_after_alloc(load_req, FakeBlocks([20, 21, 22, 23]), external)
    meta = conn.build_connector_meta()
    conn.bind_connector_metadata(meta)
    conn.start_load_kv()
    conn.clear_connector_metadata()

    # KV landed in the new blocks, preserving block->content mapping
    assert transfer.blocks[20] == b"AAAA"
    assert transfer.blocks[21] == b"BBBB"
    assert transfer.blocks[22] == b"CCCC"
    assert transfer.blocks[23] == b"DDDD"


def test_external_tokens_subtract_already_computed():
    conn, engine, _ = make_connector()
    from llmcache import KVPayload

    engine.register(
        TOKENS,
        [KVPayload(b"x" * 8, 8, 4), KVPayload(b"y" * 8, 8, 4)],
        cache_id="kb",
    )
    req = FakeRequest("r3", TOKENS, kv_transfer_params={"cache_id": "kb"})
    external, _ = conn.get_num_new_matched_tokens(req, num_computed_tokens=8)
    assert external == 8  # 16 reusable minus 8 already computed


def test_sglang_adapter_store_and_retrieve():
    engine = ExplicitKVCache(chunk_size=8)
    transfer = DictKVTransfer(block_size=4, chunk_size=8, num_layers=4)
    sg = SGLangExplicitCache(engine=engine, transfer=transfer)

    transfer.blocks.update({1: b"WWWW", 2: b"XXXX", 3: b"YYYY", 4: b"ZZZZ"})
    sg.store("doc", TOKENS, [1, 2, 3, 4], ttl=60, name="doc")
    assert engine.exists("doc")

    reused = sg.retrieve("doc", [5, 6, 7, 8], request_tokens=TOKENS)
    assert reused == 16
    assert transfer.blocks[5] == b"WWWW" and transfer.blocks[8] == b"ZZZZ"

    assert sg.retrieve("missing", [9, 10]) == 0
