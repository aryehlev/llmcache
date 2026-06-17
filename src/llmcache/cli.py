"""``llmcache`` command-line interface.

Two roles:

* ``llmcache serve`` starts a :class:`ControlServer` over a configured engine /
  storage tier (cpu, disk, redis, or tiered).
* the other subcommands are a thin REST client that manages caches in a running
  server: ``list``, ``get``, ``pin``, ``unpin``, ``ttl``, ``delete``, ``stats``,
  and ``register`` (for precomputed/offline caches).

Example::

    llmcache serve --store tiered --disk-path /var/kvcache --redis-url redis://localhost:6379
    llmcache list
    llmcache pin handbook-v1
    llmcache ttl handbook-v1 7200
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

from .control import DEFAULT_PORT, ControlServer
from .engine import ExplicitKVCache
from .store import CPUBackend, DiskBackend, RedisBackend, TieredBackend


# ----------------------------- serve -----------------------------------
def _build_store(args: argparse.Namespace):
    cap = args.capacity_bytes
    if args.store == "cpu":
        return CPUBackend(capacity_bytes=cap)
    if args.store == "disk":
        return DiskBackend(args.disk_path)
    if args.store == "redis":
        return RedisBackend(url=args.redis_url, key_ttl=args.redis_ttl)
    if args.store == "tiered":
        cold = (
            RedisBackend(url=args.redis_url, key_ttl=args.redis_ttl)
            if args.redis_url
            else DiskBackend(args.disk_path)
        )
        return TieredBackend(CPUBackend(capacity_bytes=cap), cold)
    raise SystemExit(f"unknown store {args.store!r}")


def _cmd_serve(args: argparse.Namespace) -> int:
    engine = ExplicitKVCache(_build_store(args), chunk_size=args.chunk_size, default_ttl=args.ttl)
    server = ControlServer(engine, host=args.host, port=args.port)
    host, port = server.address
    print(f"llmcache control server on http://{host}:{port} (store={args.store}, chunk_size={args.chunk_size})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.stop()
    return 0


# ----------------------------- client ----------------------------------
def _request(server: str, method: str, path: str, body: Optional[dict] = None) -> Any:
    url = server.rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"server error {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _cmd_list(a):
    _print(_request(a.server, "GET", "/v1/caches"))
    return 0


def _cmd_get(a):
    _print(_request(a.server, "GET", f"/v1/caches/{a.cache_id}"))
    return 0


def _cmd_pin(a):
    _print(_request(a.server, "POST", f"/v1/caches/{a.cache_id}/pin"))
    return 0


def _cmd_unpin(a):
    _print(_request(a.server, "POST", f"/v1/caches/{a.cache_id}/unpin"))
    return 0


def _cmd_ttl(a):
    _print(_request(a.server, "POST", f"/v1/caches/{a.cache_id}/ttl", {"ttl": a.seconds}))
    return 0


def _cmd_delete(a):
    _print(_request(a.server, "DELETE", f"/v1/caches/{a.cache_id}"))
    return 0


def _cmd_stats(a):
    _print(_request(a.server, "GET", "/v1/stats"))
    return 0


def _cmd_register(a):
    with open(a.file, "r") as f:
        spec = json.load(f)
    # Accept raw bytes payloads given as base64 already, or hex/utf8 for convenience.
    if "payloads" not in spec and "payload_bytes" in spec:
        spec["payloads"] = [base64.b64encode(p.encode()).decode() for p in spec["payload_bytes"]]
    _print(_request(a.server, "POST", "/v1/caches", spec))
    return 0


# ----------------------------- parser ----------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmcache", description="Explicit KV cache control")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the control server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--store", choices=["cpu", "disk", "redis", "tiered"], default="cpu")
    s.add_argument("--disk-path", default="/tmp/llmcache")
    s.add_argument("--redis-url", default=None)
    s.add_argument("--redis-ttl", type=int, default=None)
    s.add_argument("--capacity-bytes", type=int, default=None)
    s.add_argument("--chunk-size", type=int, default=256)
    s.add_argument("--ttl", type=float, default=None)
    s.set_defaults(func=_cmd_serve)

    def add_client(name, fn, help_):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--server", default=f"http://127.0.0.1:{DEFAULT_PORT}")
        c.set_defaults(func=fn)
        return c

    add_client("list", _cmd_list, "list caches")
    add_client("stats", _cmd_stats, "show store stats")
    add_client("get", _cmd_get, "show one cache").add_argument("cache_id")
    add_client("pin", _cmd_pin, "pin a cache").add_argument("cache_id")
    add_client("unpin", _cmd_unpin, "unpin a cache").add_argument("cache_id")
    add_client("delete", _cmd_delete, "evict a cache").add_argument("cache_id")
    ttl = add_client("ttl", _cmd_ttl, "reset a cache TTL")
    ttl.add_argument("cache_id")
    ttl.add_argument("seconds", type=float)
    reg = add_client("register", _cmd_register, "register a cache from a JSON spec file")
    reg.add_argument("file")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
