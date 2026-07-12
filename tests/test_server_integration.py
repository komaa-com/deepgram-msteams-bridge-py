"""Integration tests: a real BridgeServer on a loopback port, real WebSocket
upgrades, real caps - the transport layer the unit tests can't see."""

import asyncio
import json
import socket
import time

import aiohttp
import pytest

import deepgram_msteams_bridge.deepgram as dg_mod
import deepgram_msteams_bridge.server as server_mod
from deepgram_msteams_bridge.deepgram import DeepgramAgentSocket, DgSessionHandlers
from deepgram_msteams_bridge.hmac_auth import SIGNATURE_HEADER, TIMESTAMP_HEADER, sign
from deepgram_msteams_bridge.log import logger
from deepgram_msteams_bridge.server import start_server

from conftest import FakeAgentPort, make_config


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def signed_headers(call_id: str, secret: str = "test-secret") -> dict:
    ts = int(time.time() * 1000)
    return {TIMESTAMP_HEADER: str(ts), SIGNATURE_HEADER: sign(secret, ts, call_id)}


async def fake_connector(cfg, log, handlers):
    return FakeAgentPort()


@pytest.fixture
async def running_server():
    cfg = make_config(port=free_port(), max_connections=2, pre_start_timeout_ms=300)
    server = await start_server(cfg, connect_dg=fake_connector, vision=None)
    try:
        yield cfg, server
    finally:
        await server.close()


def url(cfg, path: str = "") -> str:
    return f"http://127.0.0.1:{cfg.port}{path}"


async def test_healthz_and_metrics(running_server):
    cfg, _ = running_server
    async with aiohttp.ClientSession() as s:
        r = await s.get(url(cfg, "/healthz"))
        assert r.status == 200 and (await r.text()) == "ok"
        r = await s.get(url(cfg, "/metrics"))
        assert r.status == 200 and "bridge_calls_total" in (await r.text())


async def test_unauthenticated_upgrade_rejected(running_server):
    cfg, _ = running_server
    async with aiohttp.ClientSession() as s:
        with pytest.raises(aiohttp.WSServerHandshakeError) as e:
            await s.ws_connect(url(cfg, "/stream/call-1"))
        assert e.value.status == 401


async def test_full_call_roundtrip(running_server):
    cfg, server = running_server
    async with aiohttp.ClientSession() as s:
        ws = await s.ws_connect(url(cfg, "/stream/call-rt"), headers=signed_headers("call-rt"))
        await ws.send_str(json.dumps({"type": "session.start", "callId": "call-rt", "threadId": "t", "caller": {}}))
        await ws.send_str(json.dumps({"type": "ping", "ts": 7}))
        frame = await asyncio.wait_for(ws.receive(), 3)
        assert json.loads(frame.data) == {"type": "pong", "ts": 7}
        assert "call-rt" in server.sessions
        await ws.close()
        for _ in range(50):
            if "call-rt" not in server.sessions:
                break
            await asyncio.sleep(0.02)
        assert "call-rt" not in server.sessions  # registry evicted on disconnect


async def test_duplicate_call_id_conflict(running_server):
    cfg, _ = running_server
    async with aiohttp.ClientSession() as s:
        ws = await s.ws_connect(url(cfg, "/stream/call-dup"), headers=signed_headers("call-dup"))
        await ws.send_str(json.dumps({"type": "session.start", "callId": "call-dup", "threadId": "t", "caller": {}}))
        with pytest.raises(aiohttp.WSServerHandshakeError) as e:
            await s.ws_connect(url(cfg, "/stream/call-dup"), headers=signed_headers("call-dup"))
        assert e.value.status in (401, 409)  # replay guard or duplicate registry, both refuse
        await ws.close()


async def test_connection_cap_returns_503(running_server):
    cfg, _ = running_server  # max_connections=2
    async with aiohttp.ClientSession() as s:
        ws1 = await s.ws_connect(url(cfg, "/s/cap-1"), headers=signed_headers("cap-1"))
        ws2 = await s.ws_connect(url(cfg, "/s/cap-2"), headers=signed_headers("cap-2"))
        with pytest.raises(aiohttp.WSServerHandshakeError) as e:
            await s.ws_connect(url(cfg, "/s/cap-3"), headers=signed_headers("cap-3"))
        assert e.value.status == 503
        await ws1.close()
        await ws2.close()
        # slots release on close: a new call is admitted again
        for _ in range(50):
            try:
                ws4 = await s.ws_connect(url(cfg, "/s/cap-4"), headers=signed_headers("cap-4"))
                break
            except aiohttp.WSServerHandshakeError:
                await asyncio.sleep(0.02)
        else:
            pytest.fail("slot was never released")
        await ws4.close()


async def test_pre_start_timeout_closes_idle_worker(running_server):
    cfg, _ = running_server  # pre_start_timeout_ms=300
    async with aiohttp.ClientSession() as s:
        ws = await s.ws_connect(url(cfg, "/s/lazy"), headers=signed_headers("lazy"))
        # never send session.start: the bridge must end the session on its own
        got_end = False
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            frame = await ws.receive(timeout=3)
            if frame.type == aiohttp.WSMsgType.TEXT and json.loads(frame.data).get("type") == "session.end":
                got_end = True
            if frame.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                break
        assert got_end


async def test_concurrent_same_callid_upgrades_admit_exactly_one(running_server):
    """Two simultaneous upgrades for one callId: without the synchronous
    pending-callId reservation both pass the registry check and the second
    silently overwrites (and leaks) the first session."""
    cfg, server = running_server
    async with aiohttp.ClientSession() as s:
        h1 = signed_headers("race-1")
        ts2 = int(time.time() * 1000) + 1  # distinct valid tuple, passes the replay guard
        h2 = {TIMESTAMP_HEADER: str(ts2), SIGNATURE_HEADER: sign("test-secret", ts2, "race-1")}
        results = await asyncio.gather(
            s.ws_connect(url(cfg, "/s/race-1"), headers=h1),
            s.ws_connect(url(cfg, "/s/race-1"), headers=h2),
            return_exceptions=True,
        )
        oks = [r for r in results if not isinstance(r, BaseException)]
        errs = [r for r in results if isinstance(r, BaseException)]
        assert len(oks) == 1 and len(errs) == 1
        assert isinstance(errs[0], aiohttp.WSServerHandshakeError) and errs[0].status == 409
        assert server._open_connections == 1
        await oks[0].close()


async def test_slots_release_when_session_constructor_raises(running_server, monkeypatch):
    """A CallSession that fails to construct must give back its connection
    slots and active gauge - a leak here 503s the whole bridge over time."""
    cfg, server = running_server

    def boom(*args, **kwargs):
        raise RuntimeError("constructor exploded")

    monkeypatch.setattr(server_mod, "CallSession", boom)
    async with aiohttp.ClientSession() as s:
        try:
            ws = await s.ws_connect(url(cfg, "/s/boom-1"), headers=signed_headers("boom-1"))
            await ws.receive(timeout=3)  # server closes after the failure
            await ws.close()
        except aiohttp.WSServerHandshakeError:
            pass
        for _ in range(50):
            if server._open_connections == 0:
                break
            await asyncio.sleep(0.02)
        assert server._open_connections == 0
        assert server._per_ip == {}
        assert server._pending_call_ids == set()
        # and the bridge still accepts new calls
        monkeypatch.undo()
        ws2 = await s.ws_connect(url(cfg, "/s/boom-2"), headers=signed_headers("boom-2"))
        await ws2.close()


async def test_deepgram_error_before_welcome_fails_fast(monkeypatch):
    """A JSON Error event before Welcome (e.g. bad API key) must surface its
    description immediately, not hide behind the 10 s Welcome timeout."""
    from aiohttp import web

    async def handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "Error", "code": "AUTH_FAILED", "description": "bad key"})
        # Return promptly: the handler must keep servicing the socket so the
        # client's close handshake (in the retry path) is acked, not stalled.
        await asyncio.sleep(0.1)
        return ws

    app = web.Application()
    app.router.add_get("/v1/agent/converse", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        monkeypatch.setattr(dg_mod, "_WS_SCHEME", "ws")  # loopback fake; production is wss
        cfg = make_config(agent_host=f"127.0.0.1:{port}")
        handlers = DgSessionHandlers(lambda m: None, lambda b: None, lambda c, r: None, lambda e: None)
        t0 = time.monotonic()
        with pytest.raises(RuntimeError) as ei:
            await DeepgramAgentSocket.connect(cfg, logger("test"), handlers)
        assert "bad key" in str(ei.value)
        assert time.monotonic() - t0 < 5  # two fast attempts, not 2 x 10 s timeouts
    finally:
        await runner.cleanup()
