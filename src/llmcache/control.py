"""HTTP control plane for an :class:`ExplicitKVCache`.

In a real deployment the engine lives inside the serving process (the vLLM
connector holds it). Embedding a :class:`ControlServer` next to it exposes the
cache's lifecycle over a small JSON/REST API so operators — and the bundled CLI —
can list, inspect, pin, re-TTL, and evict caches in a running server without
touching the inference hot path.

Built on the standard library only (``http.server``); no web framework needed.

Routes (all JSON):

    GET    /healthz
    GET    /v1/caches
    POST   /v1/caches                 {cache_id?, token_ids, payloads[b64], num_layers, ttl?, pin?, name?}
    GET    /v1/caches/{id}
    DELETE /v1/caches/{id}
    POST   /v1/caches/{id}/pin
    POST   /v1/caches/{id}/unpin
    POST   /v1/caches/{id}/ttl        {ttl}
    GET    /v1/stats
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse

from .engine import ExplicitKVCache
from .errors import CacheExpiredError, CacheNotFoundError
from .payload import KVPayload
from .types import CacheDescriptor

DEFAULT_PORT = 8975


def descriptor_to_dict(d: CacheDescriptor) -> dict[str, Any]:
    return {
        "cache_id": d.cache_id,
        "name": d.name,
        "num_tokens": d.num_tokens,
        "num_chunks": d.num_chunks,
        "chunk_size": d.chunk_size,
        "num_layers": d.num_layers,
        "worker_ids": list(d.worker_ids),
        "pinned": d.pinned,
        "created_at": d.created_at,
        "expires_at": d.expires_at,
        "hits": d.hits,
        "last_used_at": d.last_used_at,
    }


def _make_handler(engine: ExplicitKVCache):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # -- helpers --
        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def log_message(self, *args: Any) -> None:  # silence default logging
            return

        # -- routing --
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/healthz":
                return self._send(200, {"status": "ok"})
            if path == "/v1/stats":
                s = engine.stats()
                return self._send(200, {
                    "num_objects": s.num_objects,
                    "used_bytes": s.used_bytes,
                    "capacity_bytes": s.capacity_bytes,
                    "pinned_objects": s.pinned_objects,
                    "pinned_bytes": s.pinned_bytes,
                    "num_caches": len(engine.list()),
                })
            if path == "/v1/caches":
                return self._send(200, {"caches": [descriptor_to_dict(d) for d in engine.list()]})
            cid = self._cache_id(path)
            if cid is not None:
                return self._with_cache(cid, lambda d: self._send(200, descriptor_to_dict(d)))
            return self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/v1/caches":
                return self._register()
            for suffix, action in (("/pin", "pin"), ("/unpin", "unpin"), ("/ttl", "ttl")):
                if path.endswith(suffix) and path.startswith("/v1/caches/"):
                    cid = path[len("/v1/caches/") : -len(suffix)]
                    return self._mutate(cid, action)
            return self._send(404, {"error": "not found"})

        def do_DELETE(self) -> None:
            cid = self._cache_id(urlparse(self.path).path)
            if cid is None:
                return self._send(404, {"error": "not found"})
            engine.delete(cid)
            return self._send(200, {"deleted": cid})

        # -- actions --
        def _cache_id(self, path: str) -> Optional[str]:
            prefix = "/v1/caches/"
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                return path[len(prefix):] or None
            return None

        def _with_cache(self, cid: str, fn):
            try:
                return fn(engine.get(cid))
            except CacheNotFoundError:
                return self._send(404, {"error": f"cache {cid!r} not found"})
            except CacheExpiredError:
                return self._send(410, {"error": f"cache {cid!r} expired"})

        def _mutate(self, cid: str, action: str) -> None:
            try:
                if action == "pin":
                    engine.pin(cid)
                elif action == "unpin":
                    engine.unpin(cid)
                elif action == "ttl":
                    ttl = float(self._read_json()["ttl"])
                    engine.extend_ttl(cid, ttl)
            except CacheNotFoundError:
                return self._send(404, {"error": f"cache {cid!r} not found"})
            except CacheExpiredError:
                return self._send(410, {"error": f"cache {cid!r} expired"})
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(200, descriptor_to_dict(engine.get(cid)))

        def _register(self) -> None:
            try:
                body = self._read_json()
                token_ids = list(body["token_ids"])
                payloads = [
                    KVPayload(
                        data=base64.b64decode(b),
                        num_tokens=0,
                        num_layers=int(body.get("num_layers", 1)),
                    )
                    for b in body["payloads"]
                ]
                d = engine.register(
                    token_ids,
                    payloads,
                    cache_id=body.get("cache_id"),
                    ttl=body.get("ttl"),
                    pin=bool(body.get("pin", False)),
                    name=body.get("name"),
                )
            except (KeyError, ValueError) as exc:
                return self._send(400, {"error": str(exc)})
            return self._send(201, descriptor_to_dict(d))

    return Handler


class ControlServer:
    def __init__(self, engine: ExplicitKVCache, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.engine = engine
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(engine))
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> tuple[str, int]:
        return self._httpd.server_address[0], self._httpd.server_address[1]

    def start_background(self) -> "ControlServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
