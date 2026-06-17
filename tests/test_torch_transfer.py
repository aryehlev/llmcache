"""Validation for the torch KV (de)serialization and paged-buffer copy.

These run only when torch is importable. They do not require vLLM: we stand up a
buffer shaped like a vLLM paged KV cache (``[2, num_blocks, block_size, heads,
head_dim]`` per layer) and check that save -> load reproduces the KV exactly. The
shape mirrors vLLM's layout; validate against your specific build before relying on
it in production (layouts shift across releases).
"""

import pytest

torch = pytest.importorskip("torch")

from llmcache.connectors.transfer import TorchKVTransfer  # noqa: E402
from llmcache.payload import TorchSerializer  # noqa: E402


def test_torch_serializer_roundtrip_exact():
    s = TorchSerializer()
    x = torch.randn(3, 2, 4, 5)  # [layers, kv, tokens, hidden]
    payload = s.serialize(x, num_tokens=4, num_layers=3)
    assert payload.fmt == "torch" and payload.num_layers == 3
    y = s.deserialize(payload)
    assert torch.allclose(x, y)


def test_torch_kv_transfer_paged_roundtrip():
    num_layers, num_blocks, block_size, heads, head_dim = 3, 8, 4, 2, 5
    kv = {
        f"layer{i}": torch.randn(2, num_blocks, block_size, heads, head_dim)
        for i in range(num_layers)
    }
    transfer = TorchKVTransfer(kv, block_size=block_size, chunk_size=8)  # 2 blocks/chunk
    assert transfer.blocks_per_chunk() == 2

    # save the KV occupying blocks [0,1,2,3] (two chunks)
    payloads = transfer.save_from([0, 1, 2, 3], num_chunks=2)
    assert len(payloads) == 2

    # load it into a different block range [4,5,6,7]
    transfer.load_into([4, 5, 6, 7], payloads)

    # each destination block must now equal its source block, every layer
    for i in range(num_layers):
        src = kv[f"layer{i}"][:, 0:4]
        dst = kv[f"layer{i}"][:, 4:8]
        assert torch.allclose(src, dst)
