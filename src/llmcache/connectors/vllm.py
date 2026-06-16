"""vLLM V1 KV-connector with explicit-cache semantics.

This is the drop-in alternative to ``LMCacheConnectorV1``: register it as vLLM's
``kv_transfer_config`` connector and it will, on the scheduler side, decide how
much KV to reuse — but only for requests that name a cache id — and on the worker
side load that KV into the paged buffer (and optionally save new KV under a cache
id). Unlike LMCache it never matches by token-hash; reuse is opt-in per request.

A request opts in through ``SamplingParams.extra_args`` / ``kv_transfer_params``::

    {"cache_id": "handbook-v1"}                 # reuse a registered cache
    {"cache_id": "handbook-v1", "save": True,   # create/refresh it from this req
     "ttl": 3600, "pin": True, "name": "hb"}

The class subclasses vLLM's ``KVConnectorBase_V1`` when vLLM is importable; when it
is not (e.g. in unit tests), it falls back to local stand-ins with the same
surface so the scheduling/IO logic can be exercised without the engine installed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from ..engine import ExplicitKVCache
from .transfer import KVTransfer

try:  # pragma: no cover - exercised only with vLLM installed
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1 as _VLLMBase,
        KVConnectorMetadata as _VLLMMeta,
        KVConnectorRole as _VLLMRole,
    )

    KVConnectorRole = _VLLMRole
    _BASE = _VLLMBase
    _META_BASE = _VLLMMeta
    HAS_VLLM = True
except Exception:  # noqa: BLE001
    HAS_VLLM = False

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class _META_BASE:  # minimal stand-in
        pass

    class _BASE:  # minimal stand-in mirroring the V1 surface
        def __init__(self, vllm_config: Any = None, role: Any = None) -> None:
            self._vllm_config = vllm_config
            self.role = role
            self._connector_metadata: Any = None

        def bind_connector_metadata(self, metadata: Any) -> None:
            self._connector_metadata = metadata

        def clear_connector_metadata(self) -> None:
            self._connector_metadata = None

        def _get_connector_metadata(self) -> Any:
            return self._connector_metadata


# --- connector metadata (scheduler -> worker) ----------------------------
@dataclass
class LoadOp:
    request_id: str
    cache_id: str
    block_ids: list[int]
    num_chunks: int


@dataclass
class SaveOp:
    request_id: str
    cache_id: str
    token_ids: list[int]
    block_ids: list[int]
    ttl: Optional[float]
    pin: bool
    name: Optional[str]


@dataclass
class ExplicitConnectorMetadata(_META_BASE):
    loads: list[LoadOp] = field(default_factory=list)
    saves: list[SaveOp] = field(default_factory=list)


def _xfer_params(request: Any) -> dict[str, Any]:
    params = getattr(request, "kv_transfer_params", None)
    if isinstance(params, dict):
        return params
    return {}


def _prompt_tokens(request: Any) -> list[int]:
    toks = getattr(request, "prompt_token_ids", None)
    if toks is None:
        toks = getattr(request, "all_token_ids", None)
    return list(toks) if toks is not None else []


def _extract_block_ids(blocks: Any) -> list[int]:
    # vLLM's KVCacheBlocks groups block ids per kv-cache group; explicit caching
    # assumes a single group, so take the first.
    if blocks is None:
        return []
    getter = getattr(blocks, "get_block_ids", None)
    if getter is not None:
        groups = getter()
        if groups and isinstance(groups[0], (list, tuple)):
            return list(groups[0])
        return list(groups)
    if isinstance(blocks, (list, tuple)):
        return list(blocks)
    return []


class ExplicitLMCacheConnector(_BASE):
    """Explicit-cache vLLM connector. See module docstring."""

    def __init__(
        self,
        vllm_config: Any = None,
        role: Any = KVConnectorRole.SCHEDULER,
        *,
        engine: Optional[ExplicitKVCache] = None,
        transfer: Optional[KVTransfer] = None,
    ) -> None:
        super().__init__(vllm_config, role)
        self.engine = engine if engine is not None else ExplicitKVCache()
        self.transfer = transfer
        # scheduler-side staging: request_id -> pending load/save plan
        self._pending_loads: dict[str, LoadOp] = {}
        self._pending_saves: dict[str, SaveOp] = {}

    # ===================== SCHEDULER SIDE =============================
    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        params = _xfer_params(request)
        cache_id = params.get("cache_id")
        if not cache_id or not self.engine.exists(cache_id):
            return 0, False
        result = self.engine.lookup(cache_id, _prompt_tokens(request))
        external = max(0, result.num_tokens - num_computed_tokens)
        # Loading is synchronous in start_load_kv, so async flag is False.
        return external, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        params = _xfer_params(request)
        cache_id = params.get("cache_id")
        request_id = getattr(request, "request_id", None)
        block_ids = _extract_block_ids(blocks)

        if cache_id and num_external_tokens > 0 and self.engine.exists(cache_id):
            num_chunks = -(-num_external_tokens // self.engine.chunk_size)  # ceil
            self._pending_loads[request_id] = LoadOp(
                request_id=request_id,
                cache_id=cache_id,
                block_ids=block_ids,
                num_chunks=num_chunks,
            )

        if cache_id and params.get("save"):
            self._pending_saves[request_id] = SaveOp(
                request_id=request_id,
                cache_id=cache_id,
                token_ids=_prompt_tokens(request),
                block_ids=block_ids,
                ttl=params.get("ttl"),
                pin=bool(params.get("pin", False)),
                name=params.get("name"),
            )

    def build_connector_meta(self, scheduler_output: Any = None) -> ExplicitConnectorMetadata:
        meta = ExplicitConnectorMetadata(
            loads=list(self._pending_loads.values()),
            saves=list(self._pending_saves.values()),
        )
        self._pending_loads.clear()
        self._pending_saves.clear()
        return meta

    def request_finished(
        self, request: Any, block_ids: list[int]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Saving is performed synchronously in wait_for_save; nothing async to
        # hand back to the scheduler.
        return False, None

    # ===================== WORKER SIDE ===============================
    def register_kv_caches(self, kv_caches: dict) -> None:
        self._kv_caches = kv_caches
        # A real deployment supplies a TorchKVTransfer built from these buffers.

    def start_load_kv(self, forward_context: Any = None, **kwargs: Any) -> None:
        meta = self._current_meta()
        if meta is None or self.transfer is None:
            return
        for op in meta.loads:
            payloads = self.engine.load(op.cache_id, num_chunks=op.num_chunks)
            self.transfer.load_into(op.block_ids, payloads)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous load already completed in start_load_kv.
        return None

    def save_kv_layer(
        self, layer_name: str, kv_layer: Any = None, attn_metadata: Any = None, **kwargs: Any
    ) -> None:
        # This reference connector saves whole chunks (all layers) in wait_for_save
        # via the transfer adapter, so per-layer hooks are no-ops.
        return None

    def wait_for_save(self) -> None:
        meta = self._current_meta()
        if meta is None or self.transfer is None:
            return
        for op in meta.saves:
            num_chunks = -(-len(op.token_ids) // self.engine.chunk_size)
            payloads = self.transfer.save_from(op.block_ids, num_chunks)
            for idx, payload in enumerate(payloads):
                self.engine.put_chunk(op.cache_id, idx, payload, pin=op.pin)
            self.engine.commit(
                op.cache_id,
                op.token_ids,
                num_layers=self.transfer.num_layers,
                ttl=op.ttl,
                pin=op.pin,
                name=op.name,
            )

    def get_finished(
        self, finished_req_ids: set
    ) -> tuple[Optional[set], Optional[set]]:
        return None, None

    # ----- helpers -----
    def _current_meta(self) -> Optional[ExplicitConnectorMetadata]:
        getter = getattr(self, "_get_connector_metadata", None)
        if getter is not None:
            meta = getter()
            if meta is not None:
                return meta
        return getattr(self, "_connector_metadata", None)
