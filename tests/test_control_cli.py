import base64
import json

import pytest

from llmcache import ExplicitKVCache
from llmcache import cli
from llmcache.control import ControlServer


@pytest.fixture()
def server():
    engine = ExplicitKVCache(chunk_size=8)
    srv = ControlServer(engine, host="127.0.0.1", port=0).start_background()
    host, port = srv.address
    yield engine, f"http://{host}:{port}"
    srv.stop()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_register_list_get_pin_ttl_delete_via_http(server):
    engine, url = server

    # register a 16-token, 2-chunk cache through the REST API
    spec = {
        "cache_id": "doc-1",
        "token_ids": list(range(16)),
        "num_layers": 4,
        "payloads": [_b64(b"A" * 16), _b64(b"B" * 16)],
        "ttl": 600,
        "name": "doc",
    }
    created = cli._request(url, "POST", "/v1/caches", spec)
    assert created["cache_id"] == "doc-1" and created["num_chunks"] == 2

    listing = cli._request(url, "GET", "/v1/caches")
    assert [c["cache_id"] for c in listing["caches"]] == ["doc-1"]

    got = cli._request(url, "GET", "/v1/caches/doc-1")
    assert got["name"] == "doc" and got["pinned"] is False

    cli._request(url, "POST", "/v1/caches/doc-1/pin")
    assert engine.get("doc-1").pinned is True

    cli._request(url, "POST", "/v1/caches/doc-1/ttl", {"ttl": 30})
    assert engine.get("doc-1").expires_at is not None

    cli._request(url, "DELETE", "/v1/caches/doc-1")
    assert not engine.exists("doc-1")


def test_stats_and_healthz(server):
    engine, url = server
    assert cli._request(url, "GET", "/healthz")["status"] == "ok"
    stats = cli._request(url, "GET", "/v1/stats")
    assert stats["num_caches"] == 0 and stats["num_objects"] == 0


def test_missing_cache_returns_404(server):
    _, url = server
    with pytest.raises(SystemExit) as exc:
        cli._request(url, "GET", "/v1/caches/nope")
    assert "404" in str(exc.value)


def test_cli_main_drives_server(server, tmp_path, capsys):
    _, url = server
    spec = {
        "cache_id": "c1",
        "token_ids": list(range(8)),
        "num_layers": 2,
        "payloads": [_b64(b"X" * 8)],
    }
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))

    assert cli.main(["register", "--server", url, str(spec_file)]) == 0
    capsys.readouterr()  # discard register output

    assert cli.main(["list", "--server", url]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["caches"][0]["cache_id"] == "c1"

    assert cli.main(["pin", "--server", url, "c1"]) == 0
    capsys.readouterr()  # discard pin output

    assert cli.main(["stats", "--server", url]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["pinned_objects"] >= 1
