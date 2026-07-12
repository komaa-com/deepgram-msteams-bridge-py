import time

from deepgram_msteams_bridge.hmac_auth import (
    LEGACY_SIGNATURE_HEADER,
    LEGACY_TIMESTAMP_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    sign,
)
from deepgram_msteams_bridge.server import ReplayGuard, authorize_upgrade, call_id_from_path

from conftest import make_config


def _headers(secret: str, call_id: str, ts: int | None = None) -> dict:
    ts = ts if ts is not None else int(time.time() * 1000)
    return {TIMESTAMP_HEADER: str(ts), SIGNATURE_HEADER: sign(secret, ts, call_id)}


def test_call_id_from_path():
    assert call_id_from_path("/voice/msteams/stream/abc123") == "abc123"
    assert call_id_from_path("/abc%2F123") == "abc/123"
    assert call_id_from_path("/x/y?z=1") == "y"
    assert call_id_from_path("/") is None
    assert call_id_from_path("") is None
    assert call_id_from_path("/%zz") is None or call_id_from_path("/%zz") == "%zz"  # never raises


def test_authorize_ok():
    cfg = make_config()
    auth = authorize_upgrade(cfg, "/stream/call-1", _headers("test-secret", "call-1"))
    assert auth == {"callId": "call-1"}


def test_authorize_accepts_legacy_headers():
    """Pre-rename workers send X-OpenClawTeamsBridge-*; they must keep working."""
    cfg = make_config()
    ts = int(time.time() * 1000)
    headers = {LEGACY_TIMESTAMP_HEADER: str(ts), LEGACY_SIGNATURE_HEADER: sign("test-secret", ts, "call-1")}
    assert authorize_upgrade(cfg, "/stream/call-1", headers) == {"callId": "call-1"}


def test_authorize_bad_signature():
    cfg = make_config()
    headers = _headers("wrong-secret", "call-1")
    assert "error" in authorize_upgrade(cfg, "/stream/call-1", headers)


def test_authorize_stale_timestamp():
    cfg = make_config()
    old = int(time.time() * 1000) - 120_000
    headers = _headers("test-secret", "call-1", old)
    auth = authorize_upgrade(cfg, "/stream/call-1", headers)
    assert auth.get("error") == "stale or missing timestamp"


def test_authorize_missing_headers():
    cfg = make_config()
    assert "error" in authorize_upgrade(cfg, "/stream/call-1", {})


def test_authorize_no_call_id():
    cfg = make_config()
    assert authorize_upgrade(cfg, "/", _headers("test-secret", "x")).get("error") == "no callId in path"


def test_authorize_empty_secret_fails_closed():
    cfg = make_config(worker_shared_secret="")
    headers = _headers("", "call-1")
    assert "error" in authorize_upgrade(cfg, "/stream/call-1", headers)


def test_replay_guard_blocks_second_use():
    cfg = make_config()
    replay = ReplayGuard(cfg.hmac_freshness_ms)
    headers = _headers("test-secret", "call-1")
    assert authorize_upgrade(cfg, "/s/call-1", headers, replay) == {"callId": "call-1"}
    assert authorize_upgrade(cfg, "/s/call-1", headers, replay).get("error") == "replayed handshake"


def test_replay_guard_expiry():
    guard = ReplayGuard(1000)
    now = 1_000_000
    assert guard.claim("c", now, "sig", now)
    assert not guard.claim("c", now, "sig", now + 10)
    # after ts + window the record expires (is_fresh would reject the tuple anyway)
    assert guard.claim("c", now, "sig", now + 1001)


def test_unauthenticated_probe_consumes_no_replay_slot():
    cfg = make_config()
    replay = ReplayGuard(cfg.hmac_freshness_ms)
    ts = int(time.time() * 1000)
    bad = {TIMESTAMP_HEADER: str(ts), SIGNATURE_HEADER: "0" * 64}
    assert "error" in authorize_upgrade(cfg, "/s/call-1", bad, replay)
    assert replay.size == 0
