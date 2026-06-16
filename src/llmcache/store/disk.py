"""Local-filesystem KV store — the spill/cold tier.

Each chunk is one file: ``<root>/<cache_id>/<worker>_<chunk>.kvp``, with a tiny
header carrying the payload metadata. This is the analogue of LMCache's local-disk
backend and lets explicit caches persist across process restarts and exceed RAM.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Optional

from ..payload import KVPayload
from ..types import ChunkKey
from .base import StorageBackend, StoreStats

_MAGIC = b"LKVP1\n"


class DiskBackend(StorageBackend):
    name = "disk"

    def __init__(self, root: str, capacity_bytes: Optional[int] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = capacity_bytes
        self._lock = threading.RLock()
        self._pinned: set[ChunkKey] = set()

    def _path(self, key: ChunkKey) -> Path:
        return self.root / key.cache_id / f"{key.worker_id}_{key.chunk_index}.kvp"

    def put(self, key: ChunkKey, obj: KVPayload, *, pin: bool = False) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "num_tokens": obj.num_tokens,
                "num_layers": obj.num_layers,
                "fmt": obj.fmt,
                "meta": obj.meta,
                "size": obj.nbytes,
            }
        ).encode("utf-8")
        tmp = path.with_suffix(".tmp")
        with self._lock:
            with open(tmp, "wb") as f:
                f.write(_MAGIC)
                f.write(len(header).to_bytes(4, "little"))
                f.write(header)
                f.write(obj.data)
            os.replace(tmp, path)
            if pin:
                self._pinned.add(key)

    def get(self, key: ChunkKey) -> Optional[KVPayload]:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            if f.read(len(_MAGIC)) != _MAGIC:
                return None
            hlen = int.from_bytes(f.read(4), "little")
            header = json.loads(f.read(hlen).decode("utf-8"))
            data = f.read()
        return KVPayload(
            data=data,
            num_tokens=header["num_tokens"],
            num_layers=header["num_layers"],
            fmt=header["fmt"],
            meta=header.get("meta", {}),
        )

    def contains(self, key: ChunkKey) -> bool:
        return self._path(key).exists()

    def remove(self, key: ChunkKey) -> bool:
        path = self._path(key)
        with self._lock:
            self._pinned.discard(key)
            if path.exists():
                path.unlink()
                try:
                    path.parent.rmdir()  # remove cache dir if now empty
                except OSError:
                    pass
                return True
        return False

    def pin(self, key: ChunkKey) -> None:
        self._pinned.add(key)

    def unpin(self, key: ChunkKey) -> None:
        self._pinned.discard(key)

    def keys(self) -> Iterable[ChunkKey]:
        out: list[ChunkKey] = []
        if not self.root.exists():
            return out
        for cache_dir in self.root.iterdir():
            if not cache_dir.is_dir():
                continue
            for f in cache_dir.glob("*.kvp"):
                worker_s, _, chunk_s = f.stem.partition("_")
                try:
                    out.append(ChunkKey(cache_dir.name, int(chunk_s), int(worker_s)))
                except ValueError:
                    continue
        return out

    def stats(self) -> StoreStats:
        used = 0
        n = 0
        for key in self.keys():
            p = self._path(key)
            if p.exists():
                used += p.stat().st_size
                n += 1
        return StoreStats(
            num_objects=n,
            used_bytes=used,
            capacity_bytes=self.capacity_bytes,
            pinned_objects=len(self._pinned),
            pinned_bytes=0,
        )
