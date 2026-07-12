"""Worker-facing WebSocket server. The StandIn media bridge dials
{wsBaseUrl}/{callId} with an HMAC-signed upgrade
(X-OpenClawTeamsBridge-Timestamp / -Signature over "{timestampMs}.{callId}").

DoS guards - parity with the OpenClaw/Hermes msteams providers. A single shared
secret gates the upgrade, but a buggy or compromised worker (or a leaked
secret) must not be able to exhaust memory/sockets.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Mapping
from urllib.parse import unquote

from aiohttp import WSMsgType, web

from .config import BridgeConfig
from .deepgram import BRIDGE_FUNCTIONS, CustomTool, DgConnector
from .hmac_auth import (
    LEGACY_SIGNATURE_HEADER,
    LEGACY_TIMESTAMP_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    is_fresh,
    verify,
)
from .log import logger
from .metrics import metric_dec, metric_inc, metric_observe, render_metrics
from . import session as _session
from .session import CallSession
from .vision import VisionDescriber

log = logger("server")

# Max inbound WS frame. Caller audio is ~640 B/frame; a JPEG video.frame is the
# large one. 2 MB matches the sibling providers and bounds a single message.
MAX_INBOUND_PAYLOAD_BYTES = 2 * 1024 * 1024
# Max concurrent worker connections (one per live call).
DEFAULT_MAX_CONNECTIONS = 64
# A worker that authenticates but never sends session.start is dropped after this.
DEFAULT_PRE_START_TIMEOUT_MS = 10_000
# Bounded window for queued session.end frames + close handshakes to flush on shutdown.
SHUTDOWN_GRACE_S = 2.0


def call_id_from_path(path: str | None) -> str | None:
    """callId = last non-empty path segment of the upgrade URL."""
    if not path:
        return None
    segments = [s for s in path.split("?")[0].split("/") if s]
    if not segments:
        return None
    return unquote(segments[-1])


class ReplayGuard:
    """Single-use guard for verified upgrade tuples (callId, ts, sig). Even
    inside the freshness window, a captured handshake must not be replayable to
    open a second (ghost) session for the same call. Records survive until the
    timestamp itself stops being fresh (ts + window)."""

    def __init__(self, window_ms: float) -> None:
        self._window_ms = window_ms
        self._seen: dict[str, float] = {}

    def claim(self, call_id: str, ts: float, sig: str, now_ms: float | None = None) -> bool:
        """True if this tuple is NEW (and records it); False if already used."""
        now = time.time() * 1000 if now_ms is None else now_ms
        for key in [k for k, expiry in self._seen.items() if expiry <= now]:
            del self._seen[key]
        key = f"{call_id}.{ts}.{sig}"
        if key in self._seen:
            return False
        # Expire when the timestamp stops being fresh, not "now + window": the
        # tuple is unusable past ts + window anyway (is_fresh would reject it).
        self._seen[key] = ts + self._window_ms
        return True

    @property
    def size(self) -> int:
        return len(self._seen)


def authorize_upgrade(
    cfg: BridgeConfig,
    path: str,
    headers: Mapping[str, str],
    replay: ReplayGuard | None = None,
) -> dict[str, str]:
    """Validate an upgrade request. Returns {"callId": ...} or {"error": ...}."""
    call_id = call_id_from_path(path)
    if not call_id:
        return {"error": "no callId in path"}
    # Fail closed: an empty/unset shared secret must reject every upgrade rather
    # than authenticating anyone. load_config() requires it, but never trust that.
    if not cfg.worker_shared_secret:
        return {"error": "bridge shared secret is not configured"}
    ts_header = (
        headers.get(TIMESTAMP_HEADER)
        or headers.get(TIMESTAMP_HEADER.title())
        or headers.get(LEGACY_TIMESTAMP_HEADER)
        or headers.get(LEGACY_TIMESTAMP_HEADER.title())
        or ""
    )
    sig = (
        headers.get(SIGNATURE_HEADER)
        or headers.get(SIGNATURE_HEADER.title())
        or headers.get(LEGACY_SIGNATURE_HEADER)
        or headers.get(LEGACY_SIGNATURE_HEADER.title())
        or ""
    )
    try:
        ts = float(ts_header)
    except (TypeError, ValueError):
        return {"error": "stale or missing timestamp"}
    if not is_fresh(ts, cfg.hmac_freshness_ms):
        return {"error": "stale or missing timestamp"}
    # The worker signs with the integer millisecond timestamp exactly as sent.
    ts_str = ts_header.strip()
    if not verify(cfg.worker_shared_secret, ts_str, call_id, sig):
        return {"error": "bad signature"}
    # Replay guard runs LAST, so an unauthenticated probe can never consume a
    # replay slot (it fails the signature check first).
    if replay is not None and not replay.claim(call_id, ts, sig):
        return {"error": "replayed handshake"}
    return {"callId": call_id}


def _remote_key(request: web.Request, trust_proxy: bool) -> str:
    """Best-effort remote-IP key for the per-IP connection cap. StandIn is a
    hosted service dialing from a small set of egress IPs, and a reverse
    proxy/LB collapses every client to its own address - so keying on the socket
    address alone makes the per-IP cap either useless (all one IP) or a footgun
    (throttles all calls). When trust_proxy is set, use the FIRST X-Forwarded-For
    hop instead. Only enable it behind a proxy you control (the header is
    client-spoofable otherwise)."""
    if trust_proxy:
        xff = request.headers.get("x-forwarded-for", "")
        first = xff.split(",")[0].strip()
        if first:
            return first
    return request.remote or "unknown"


class _AioWorkerPort:
    """WorkerPort over an aiohttp WebSocketResponse: a writer task drains a FIFO
    queue so sends are non-blocking for the session, with the queued byte total
    exposed for the session's backpressure cap."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self._ws = ws
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._buffered = 0
        self._writer = asyncio.create_task(self._drain())

    @property
    def is_open(self) -> bool:
        return not self._ws.closed

    @property
    def buffered_bytes(self) -> int:
        return self._buffered

    def send_text(self, payload: str) -> None:
        self._buffered += len(payload)
        self._queue.put_nowait(payload)

    def close(self, code: int, reason: str) -> None:
        self._queue.put_nowait(None)  # the writer exits after flushing what's queued

        async def _close_after_flush() -> None:
            # WAIT for the writer to drain before closing the socket - closing
            # concurrently could drop the just-queued session.end frame, which
            # is exactly the frame the graceful paths care about. Bounded so a
            # stalled socket cannot hold the close open.
            try:
                await asyncio.wait_for(asyncio.shield(self._writer), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError, Exception):
                pass
            try:
                await self._ws.close(code=code, message=reason.encode("utf-8")[:100])
            except Exception:
                pass

        asyncio.ensure_future(_close_after_flush())

    async def _drain(self) -> None:
        while True:
            payload = await self._queue.get()
            if payload is None:
                return
            self._buffered -= len(payload)
            if self._ws.closed:
                continue  # drop the rest; close() already queued the sentinel or the read loop ended
            try:
                await self._ws.send_str(payload)
            except Exception:
                return  # socket died; the read loop tears the session down

    def stop_writer(self) -> None:
        # Graceful: let the writer flush what is queued (session.end from the
        # teardown path) and exit on the sentinel; only cancel if it is somehow
        # still running after a bounded window.
        self._queue.put_nowait(None)
        if self._writer.done():
            return
        writer = self._writer

        def _cancel_if_stuck() -> None:
            if not writer.done():
                writer.cancel()

        asyncio.get_running_loop().call_later(3.0, _cancel_if_stuck)


class BridgeServer:
    """Handle returned by start_server(); owns the aiohttp runner and the live
    session registry (used for the SIGTERM/SIGINT drain)."""

    def __init__(
        self,
        cfg: BridgeConfig,
        connect_dg: DgConnector | None,
        vision: VisionDescriber | None | str,
        tools: list[CustomTool] | None = None,
    ) -> None:
        self.cfg = cfg
        self._connect_dg = connect_dg
        self._vision = vision
        # Custom tool names must not shadow the built-in bridge functions - fail
        # loud at startup rather than dispatching ambiguously mid-call.
        builtin = {str(f["name"]) for f in BRIDGE_FUNCTIONS}
        seen: set[str] = set()
        for t in tools or []:
            if t.name in builtin:
                raise ValueError(f'custom tool "{t.name}" collides with a built-in bridge function')
            if t.name in seen:
                raise ValueError(f'custom tool "{t.name}" is registered twice')
            seen.add(t.name)
        self._tools = list(tools or [])
        self.sessions: dict[str, CallSession] = {}
        # callIds reserved SYNCHRONOUSLY at upgrade time, before the first await
        # (ws.prepare): two concurrent upgrades for the same callId would both
        # pass the registry check and the second would silently overwrite the
        # first, leaking a session that holds a billed agent socket forever.
        self._pending_call_ids: set[str] = set()
        self._open_connections = 0
        self._per_ip: dict[str, int] = {}
        self._replay = ReplayGuard(cfg.hmac_freshness_ms)
        self._max_connections = cfg.max_connections if cfg.max_connections > 0 else DEFAULT_MAX_CONNECTIONS
        # Per-IP cap defaults to the TOTAL cap (i.e. effectively off) rather than a
        # low fixed number: the bridge's only legitimate client is StandIn, which
        # dials from a small set of IPs, so a low per-IP cap would silently throttle
        # total concurrent calls. Set MAX_CONNECTIONS_PER_IP explicitly (with
        # TRUST_PROXY_XFF when behind a proxy) if you want a real per-IP limit.
        self._max_per_ip = cfg.max_connections_per_ip if cfg.max_connections_per_ip > 0 else self._max_connections
        self._pre_start_timeout_s = (
            cfg.pre_start_timeout_ms if cfg.pre_start_timeout_ms > 0 else DEFAULT_PRE_START_TIMEOUT_MS
        ) / 1000
        self._runner: web.AppRunner | None = None

    # ---- request handling ----

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/healthz":
            return web.Response(text="ok")
        if request.path == "/metrics":
            return web.Response(text=render_metrics(), content_type="text/plain")
        if (request.headers.get("upgrade") or "").lower() != "websocket":
            return web.Response(status=404)
        return await self._handle_upgrade(request)

    async def _handle_upgrade(self, request: web.Request) -> web.StreamResponse:
        ip = _remote_key(request, self.cfg.trust_proxy)
        # Cheap caps first (before HMAC) so a flood can't force expensive crypto.
        if self._open_connections >= self._max_connections:
            metric_inc("bridge_upgrades_rejected_cap_total")
            log.warn(f"rejected upgrade from {ip}: server connection cap reached")
            return web.Response(status=503)
        if self._per_ip.get(ip, 0) >= self._max_per_ip:
            metric_inc("bridge_upgrades_rejected_cap_total")
            log.warn(f"rejected upgrade from {ip}: per-IP connection cap reached")
            return web.Response(status=503)
        auth = authorize_upgrade(self.cfg, request.path, request.headers, self._replay)
        if "error" in auth:
            metric_inc("bridge_upgrades_rejected_auth_total")
            log.warn(f"rejected upgrade from {ip}: {auth['error']}")
            return web.Response(status=401)
        call_id = auth["callId"]
        # A live session already owns this callId - a retry/rollout reconnect.
        # Reject rather than spin up a second billed Voice Agent session. The
        # pending set covers the window between here and registration: it is
        # claimed before the first await, so a concurrent duplicate cannot slip
        # past both checks.
        if call_id in self.sessions or call_id in self._pending_call_ids:
            metric_inc("bridge_upgrades_rejected_duplicate_total")
            log.warn(f"rejected upgrade from {ip}: callId {call_id[:12]}... already has a live session")
            return web.Response(status=409)
        self._pending_call_ids.add(call_id)

        # Claim the connection slots BEFORE any await - a burst of simultaneous
        # upgrades must not transiently exceed the caps. Released exactly once.
        self._open_connections += 1
        self._per_ip[ip] = self._per_ip.get(ip, 0) + 1
        released = False

        def release_slots() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._open_connections = max(0, self._open_connections - 1)
            n = self._per_ip.get(ip, 1) - 1
            if n <= 0:
                self._per_ip.pop(ip, None)
            else:
                self._per_ip[ip] = n

        # WS-level heartbeat (30 s): detects a half-open TCP peer at the protocol
        # layer long before the 90 s application idle watchdog - important
        # because a dead callId 409-blocks every reconnect until it clears.
        ws = web.WebSocketResponse(max_msg_size=MAX_INBOUND_PAYLOAD_BYTES, heartbeat=30.0)
        try:
            await ws.prepare(request)
        except Exception:
            self._pending_call_ids.discard(call_id)
            release_slots()
            raise

        log.info(f"worker connected for call {call_id[:12]}... ({self._open_connections}/{self._max_connections})")
        metric_inc("bridge_calls_total")
        metric_inc("bridge_calls_active")

        # ONE finally owns every release from here: if any constructor below
        # raises, the slots, the active gauge, and the pending callId must
        # still be given back - a leak here climbs to max_connections and then
        # 503s every new call until restart.
        port: _AioWorkerPort | None = None
        session: CallSession | None = None
        pre_start_timer: asyncio.TimerHandle | None = None
        started = time.monotonic()
        try:
            port = _AioWorkerPort(ws)
            session = CallSession(
                self.cfg,
                port,
                call_id,
                connect_dg=self._connect_dg,
                vision=self._vision,
                tools=self._tools,
                on_closed=lambda: self.sessions.pop(call_id, None),  # evict on teardown
            )
            self.sessions[call_id] = session
            self._pending_call_ids.discard(call_id)  # the registry owns dedup now

            # Drop a worker that authenticates but never STARTS a call. The timer asks
            # the session whether session.start actually arrived - clearing on the
            # first message of any type would let an authenticated client hold the
            # socket forever by sending pings.
            loop = asyncio.get_running_loop()

            def pre_start_check() -> None:
                if session is not None and not session.has_started and not session.closed:
                    log.warn(
                        f"call {call_id[:12]}... sent no session.start in {int(self._pre_start_timeout_s * 1000)}ms; closing"
                    )
                    session.end_call("no session.start")

            pre_start_timer = loop.call_later(self._pre_start_timeout_s, pre_start_check)

            async for frame in ws:
                if frame.type == WSMsgType.TEXT:
                    session.handle_worker_message(frame.data)
                elif frame.type == WSMsgType.BINARY:
                    session.handle_worker_message(frame.data)
                elif frame.type == WSMsgType.ERROR:
                    session.handle_worker_error(ws.exception() or RuntimeError("websocket error"))
                    break
        finally:
            self._pending_call_ids.discard(call_id)
            if pre_start_timer is not None:
                pre_start_timer.cancel()
            if session is not None:
                session.handle_worker_close()
            if port is not None:
                port.stop_writer()
            metric_dec("bridge_calls_active")
            elapsed = time.monotonic() - started
            # cumulative counter (averages) + histogram (p50/p95/p99)
            metric_inc("bridge_call_seconds_total", elapsed)
            metric_observe("bridge_call_duration_seconds", elapsed)
            release_slots()
        return ws

    # ---- lifecycle ----

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("GET", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()

        # Native TLS (wss) when both cert + key are provided; otherwise plain WS,
        # which MUST be fronted by a TLS terminator (tunnel / ingress / LB) -
        # caller audio and video would otherwise cross the wire in plaintext.
        ssl_ctx: ssl.SSLContext | None = None
        if self.cfg.tls_cert_path and self.cfg.tls_key_path:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # never negotiate down to 1.0/1.1
            ssl_ctx.load_cert_chain(self.cfg.tls_cert_path, self.cfg.tls_key_path)
            log.info("native TLS enabled (wss)")
        else:
            log.warn("no TLS_CERT_PATH/TLS_KEY_PATH: serving plain WS - front this with a TLS terminator in production")

        site = web.TCPSite(self._runner, self.cfg.host, self.cfg.port, ssl_context=ssl_ctx)
        await site.start()
        log.info(
            f"deepgram-msteams-bridge listening on {self.cfg.host}:{self.cfg.port} "
            f"(listen {self.cfg.listen_model}, think {self.cfg.think_provider}/{self.cfg.think_model}, "
            f"speak {self.cfg.speak_model})"
        )

    async def drain(self, signal_name: str = "shutdown") -> None:
        """Gracefully end every live call (session.end + close both sockets)
        instead of hard-dropping calls on a redeploy. A session mid-goodbye is
        left to finish (session.shutdown defers to its hard-bounded backstop),
        so the wait is bounded by the worst-case goodbye deadline - a fixed 2 s
        would truncate a goodbye the caller is still hearing."""
        sessions = list(self.sessions.values())
        log.info(f"{signal_name}: draining {len(sessions)} live call(s)")
        for s in sessions:
            try:
                s.shutdown("bridge-shutdown")
            except Exception:
                pass  # keep draining the rest
        if not sessions:
            return
        deadline = time.monotonic() + max(
            SHUTDOWN_GRACE_S,
            (self.cfg.goodbye_grace_ms + _session.GOODBYE_HARD_CAP_MS) / 1000 + 0.5,
        )
        while self.sessions and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        # teardown queues session.end + starts the close handshakes
        # asynchronously; exiting immediately would cut those before they flush.
        await asyncio.sleep(SHUTDOWN_GRACE_S)

    async def close(self) -> None:
        await self.drain()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


async def start_server(
    cfg: BridgeConfig,
    connect_dg: DgConnector | None = None,
    vision: VisionDescriber | None | str = "auto",
    tools: list[CustomTool] | None = None,
) -> BridgeServer:
    """Start the bridge and return its handle.

    `vision` defaults to the env-configured describer; pass an async callable
    to supply your own model, or None to disable the look tool. `tools`
    registers custom client-side function tools the bridge executes - see
    CustomTool; names must not collide with the built-ins."""
    server = BridgeServer(cfg, connect_dg, vision, tools)
    await server.start()
    return server
