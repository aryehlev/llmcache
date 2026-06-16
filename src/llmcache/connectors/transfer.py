"""Adapters that move KV between an engine's paged buffer and KV payloads.

The connector logic (deciding *what* to load/save, mapping cache ids to chunks and
blocks, talking to the engine) is platform-neutral and fully tested. The actual
tensor copy is the only part that needs the live framework, so it lives behind
:class:`KVTransfer`:

* :class:`DictKVTransfer` — a pure-Python paged buffer (block id -> bytes) used in
  tests and for dry-running the connector without torch/vLLM.
* :class:`TorchKVTransfer` — copies between vLLM's paged KV tensors and payloads.
  It is framework- and version-sensitive; validate it against your target vLLM
  build before relying on it (see the docstring).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..payload import KVPayload


class KVTransfer(ABC):
    """Moves KV for a contiguous token run in/out of a paged buffer."""

    #: Tokens per KV block in the underlying buffer.
    block_size: int
    #: Tokens per cache chunk (must be a multiple of ``block_size``).
    chunk_size: int
    num_layers: int

    def blocks_per_chunk(self) -> int:
        if self.chunk_size % self.block_size != 0:
            raise ValueError("chunk_size must be a multiple of block_size")
        return self.chunk_size // self.block_size

    @abstractmethod
    def load_into(self, block_ids: Sequence[int], payloads: Sequence[KVPayload]) -> None:
        """Write cached ``payloads`` (one per chunk) into the given blocks."""

    @abstractmethod
    def save_from(self, block_ids: Sequence[int], num_chunks: int) -> list[KVPayload]:
        """Read ``num_chunks`` chunk payloads out of the given blocks."""


class DictKVTransfer(KVTransfer):
    """In-memory paged buffer for tests / offline dry runs."""

    def __init__(self, block_size: int = 16, chunk_size: int = 256, num_layers: int = 4):
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        # block_id -> opaque bytes representing that block's KV across all layers.
        self.blocks: dict[int, bytes] = {}

    def load_into(self, block_ids: Sequence[int], payloads: Sequence[KVPayload]) -> None:
        bpc = self.blocks_per_chunk()
        for chunk_idx, payload in enumerate(payloads):
            chunk_blocks = block_ids[chunk_idx * bpc : (chunk_idx + 1) * bpc]
            # Split the chunk payload evenly across its blocks (fake layout).
            parts = _split(payload.data, len(chunk_blocks))
            for blk, part in zip(chunk_blocks, parts):
                self.blocks[blk] = part

    def save_from(self, block_ids: Sequence[int], num_chunks: int) -> list[KVPayload]:
        bpc = self.blocks_per_chunk()
        out: list[KVPayload] = []
        for chunk_idx in range(num_chunks):
            chunk_blocks = block_ids[chunk_idx * bpc : (chunk_idx + 1) * bpc]
            data = b"".join(self.blocks.get(blk, b"") for blk in chunk_blocks)
            out.append(
                KVPayload(
                    data=data,
                    num_tokens=self.chunk_size,
                    num_layers=self.num_layers,
                    fmt="raw",
                )
            )
        return out


def _split(data: bytes, n: int) -> list[bytes]:
    if n <= 0:
        return []
    size = max(1, (len(data) + n - 1) // n)
    parts = [data[i * size : (i + 1) * size] for i in range(n)]
    while len(parts) < n:
        parts.append(b"")
    return parts[:n]


class TorchKVTransfer(KVTransfer):
    """Copy between vLLM paged KV tensors and payloads.

    vLLM stores KV per layer as a tensor whose leading dims are roughly
    ``[2, num_blocks, block_size, num_kv_heads, head_dim]`` (exact layout varies by
    attention backend and version). This adapter gathers the blocks for a chunk
    across all layers, stacks them, and serializes; load reverses that. Treat it as
    a starting point and validate against your build — paged layouts change between
    vLLM releases.
    """

    def __init__(
        self,
        kv_caches: dict,  # layer_name -> torch.Tensor (paged buffer)
        *,
        block_size: int,
        chunk_size: int,
    ):  # pragma: no cover - requires torch + vLLM paged buffers
        self.kv_caches = kv_caches
        self.layer_names = list(kv_caches.keys())
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.num_layers = len(self.layer_names)
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("TorchKVTransfer requires torch") from exc

    def load_into(self, block_ids, payloads):  # pragma: no cover
        import torch

        bpc = self.blocks_per_chunk()
        for chunk_idx, payload in enumerate(payloads):
            chunk_blocks = list(block_ids[chunk_idx * bpc : (chunk_idx + 1) * bpc])
            dtype, shape = payload.meta["dtype"], payload.meta["shape"]
            buf = torch.frombuffer(bytearray(payload.data), dtype=getattr(torch, dtype.replace("torch.", "")))
            buf = buf.reshape(shape)  # [num_layers, 2, len(chunk_blocks), block_size, ...]
            for li, name in enumerate(self.layer_names):
                layer = self.kv_caches[name]
                for bi, blk in enumerate(chunk_blocks):
                    layer[:, blk].copy_(buf[li, :, bi])

    def save_from(self, block_ids, num_chunks):  # pragma: no cover
        import torch

        bpc = self.blocks_per_chunk()
        out: list[KVPayload] = []
        for chunk_idx in range(num_chunks):
            chunk_blocks = list(block_ids[chunk_idx * bpc : (chunk_idx + 1) * bpc])
            layers = []
            for name in self.layer_names:
                layer = self.kv_caches[name]
                blocks = torch.stack([layer[:, blk] for blk in chunk_blocks], dim=1)
                layers.append(blocks)
            stacked = torch.stack(layers, dim=0).contiguous().cpu()
            out.append(
                KVPayload(
                    data=stacked.numpy().tobytes(),
                    num_tokens=self.chunk_size,
                    num_layers=self.num_layers,
                    fmt="torch",
                    meta={"dtype": str(stacked.dtype), "shape": list(stacked.shape)},
                )
            )
        return out
