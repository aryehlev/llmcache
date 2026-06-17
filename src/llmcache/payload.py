"""KV payloads and (de)serialization.

A :class:`KVPayload` is an opaque, storable blob holding the key/value tensors for
one chunk of one layer-set, plus enough metadata to place it back into a paged KV
buffer. The bytes are produced by a :class:`KVSerializer`. Keeping the payload
opaque is deliberate: the storage and engine layers never need torch, so they are
fully testable offline, while a torch-backed serializer handles real tensors at
the connector boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from .errors import SerializationError

_MAGIC = b"LKVP1\n"


@dataclass
class KVPayload:
    """Serialized KV for one chunk."""

    data: bytes
    num_tokens: int
    num_layers: int
    fmt: str = "raw"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return len(self.data)


def pack_payload(obj: "KVPayload") -> bytes:
    """Serialize a payload (metadata header + data) for byte-oriented stores."""
    header = json.dumps(
        {
            "num_tokens": obj.num_tokens,
            "num_layers": obj.num_layers,
            "fmt": obj.fmt,
            "meta": obj.meta,
        }
    ).encode("utf-8")
    return b"".join((_MAGIC, len(header).to_bytes(4, "little"), header, obj.data))


def unpack_payload(blob: bytes) -> "KVPayload":
    if blob[: len(_MAGIC)] != _MAGIC:
        raise SerializationError("bad payload framing")
    off = len(_MAGIC)
    hlen = int.from_bytes(blob[off : off + 4], "little")
    off += 4
    header = json.loads(blob[off : off + hlen].decode("utf-8"))
    off += hlen
    return KVPayload(
        data=blob[off:],
        num_tokens=header["num_tokens"],
        num_layers=header["num_layers"],
        fmt=header["fmt"],
        meta=header.get("meta", {}),
    )


@runtime_checkable
class KVSerializer(Protocol):
    """Converts between engine-native KV tensors and :class:`KVPayload`."""

    fmt: str

    def serialize(self, kv: Any, *, num_tokens: int, num_layers: int) -> KVPayload: ...

    def deserialize(self, payload: KVPayload) -> Any: ...


class RawSerializer:
    """Identity serializer used for tests and non-tensor payloads.

    ``kv`` must be ``bytes`` (or buffer-like). No tensor framework required.
    """

    fmt = "raw"

    def serialize(self, kv: Any, *, num_tokens: int, num_layers: int) -> KVPayload:
        if isinstance(kv, (bytes, bytearray, memoryview)):
            data = bytes(kv)
        else:
            raise SerializationError(
                "RawSerializer expects bytes-like KV; "
                "use TorchSerializer for tensors"
            )
        return KVPayload(data=data, num_tokens=num_tokens, num_layers=num_layers, fmt=self.fmt)

    def deserialize(self, payload: KVPayload) -> bytes:
        if payload.fmt != self.fmt:
            raise SerializationError(f"cannot raw-deserialize fmt={payload.fmt!r}")
        return payload.data


class TorchSerializer:
    """Serializer for a stacked KV tensor of shape ``[num_layers, 2, tokens, ...]``.

    torch is imported lazily so importing llmcache never requires it. The tensor
    is moved to CPU, made contiguous, and dumped to bytes; metadata records dtype
    and shape for exact reconstruction.
    """

    fmt = "torch"

    def __init__(self, device: str = "cpu", pin_memory: bool = False) -> None:
        self.device = device
        self.pin_memory = pin_memory

    def _torch(self):  # pragma: no cover - exercised only with torch present
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SerializationError("torch is required for TorchSerializer") from exc
        return torch

    def serialize(self, kv: Any, *, num_tokens: int, num_layers: int) -> KVPayload:  # pragma: no cover
        torch = self._torch()
        t = kv.detach().to("cpu").contiguous()
        data = t.numpy().tobytes()
        return KVPayload(
            data=data,
            num_tokens=num_tokens,
            num_layers=num_layers,
            fmt=self.fmt,
            meta={"dtype": str(t.dtype), "shape": list(t.shape)},
        )

    def deserialize(self, payload: KVPayload) -> Any:  # pragma: no cover
        torch = self._torch()
        import numpy as np  # noqa: F401

        if payload.fmt != self.fmt:
            raise SerializationError(f"cannot torch-deserialize fmt={payload.fmt!r}")
        dtype = payload.meta["dtype"].replace("torch.", "")
        shape = tuple(payload.meta["shape"])
        np_dtype = _TORCH_TO_NUMPY.get(dtype, dtype)
        arr = np.frombuffer(payload.data, dtype=np_dtype).reshape(shape)
        out = torch.from_numpy(arr.copy())
        if self.device != "cpu":
            out = out.to(self.device)
        return out


_TORCH_TO_NUMPY = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "float16",  # numpy lacks bf16; caller should bitcast if needed
    "float64": "float64",
    "int8": "int8",
    "uint8": "uint8",
    "int32": "int32",
    "int64": "int64",
}
