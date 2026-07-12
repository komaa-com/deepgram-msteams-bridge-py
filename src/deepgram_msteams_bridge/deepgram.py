"""Deepgram Voice Agent WebSocket client + the one REST call the bridge needs
(standalone Aura TTS for the deterministic governor goodbye).

Wire reference (validated against the Voice Agent API docs): the client
connects to wss://{host}/v1/agent/converse with `Authorization: Token <key>`,
waits for `Welcome`, then sends a `Settings` message configuring audio in/out
(linear16 at 16 kHz both ways - the StandIn wire rate, so the hot path is
COPY-ONLY, no resampling) and the agent (listen/think/speak providers, prompt,
greeting, functions). Caller audio is sent as RAW BINARY frames; agent audio
arrives as raw binary frames. JSON events ride the same socket:
UserStartedSpeaking (barge-in), FunctionCallRequest/FunctionCallResponse
(client-side tools), ConversationText (transcripts), InjectAgentMessage /
UpdatePrompt (client -> server), KeepAlive, Error/Warning. No audio may be
sent before the server acks with SettingsApplied - the session enforces that.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlencode

import aiohttp

from .config import BridgeConfig
from .log import Logger

# The one wire rate: StandIn PCM 16 kHz mono = Deepgram linear16 @ 16000.
WIRE_SAMPLE_RATE = 16_000

# Client keep-alive cadence (the Voice Agent socket idles out without it).
KEEPALIVE_INTERVAL_S = 8.0

# Time bound on the REST TTS call and the WS handshake, so a hung endpoint can
# never wedge a call open.
DG_REST_TIMEOUT_MS = 10_000

# How long to wait for the server's Welcome after the socket opens.
WELCOME_TIMEOUT_S = 10.0

# Outbound (bridge->DG) send-buffer cap, mirroring the worker-side guard: a
# stalled agent socket must not pile up unbounded caller-audio send tasks.
MAX_DG_SEND_BUFFER_BYTES = 1 * 1024 * 1024


# ---- built-in bridge functions ----

# Client-side functions registered on every session (declared in Settings under
# agent.think.functions; entries WITHOUT an endpoint are executed by this
# client via FunctionCallRequest/FunctionCallResponse). The bridge implements
# their behavior; nothing to configure on the Deepgram side.
BRIDGE_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "end_call",
        "description": (
            "Hang up the call. Call this when the conversation is finished, the caller says goodbye, "
            "or the caller asks you to hang up."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "express",
        "description": (
            "Show an emotion on your avatar's face. Use it to react naturally "
            "(e.g. happy when greeting, surprised at unexpected news)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": "The emotion to express, e.g. happy, sad, surprised, neutral.",
                }
            },
            "required": ["emotion"],
        },
    },
    {
        "name": "show_image",
        "description": (
            "Show an image to the caller on your video tile. Provide a public https image URL (jpeg or png). "
            "Use it when a visual would help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Public https URL of a jpeg/png image."},
                "caption": {"type": "string", "description": "Optional short caption."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "look",
        "description": (
            "Look at the caller's camera or shared screen and get a text description of what is visible. "
            "Use it when the caller refers to something they are showing you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": 'Which video to look at: "camera" or "screenshare".'},
                "question": {"type": "string", "description": "What you want to know about the video."},
            },
            "required": [],
        },
    },
]


# ---- tool extensibility ----


@dataclass(frozen=True)
class CustomToolContext:
    """Per-call context handed to custom tool handlers."""

    call_id: str
    participant_count: int
    recording_active: bool
    log: Logger


# Handler: (params, ctx) -> str, sync or async. Keep handlers fast - the model
# (and the caller) is waiting on the result; enforce your own timeout for slow
# backends.
CustomToolHandler = Callable[[dict[str, Any], CustomToolContext], "str | Awaitable[str]"]


@dataclass(frozen=True)
class CustomTool:
    """A custom client-side function the BRIDGE executes: the agent calls it,
    the handler runs in your process, and the returned string goes back as the
    FunctionCallResponse content."""

    name: str
    """Function name the agent calls. Must not collide with the built-in bridge functions."""
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    """JSON schema for the parameters ({type: "object", properties, required})."""
    handler: CustomToolHandler = lambda params, ctx: ""  # noqa: E731 - dataclass default


def custom_tool_schema(tool: CustomTool) -> dict[str, Any]:
    """The Settings functions entry for a custom tool (schema only; the handler stays bridge-side)."""
    return {"name": tool.name, "description": tool.description, "parameters": tool.parameters}


# ---- Settings / prompt builders ----

DEFAULT_INSTRUCTIONS = (
    "You are a helpful voice assistant on a live Microsoft Teams call. You are speaking aloud: "
    "keep replies short, natural and conversational, and never use markdown, lists or emoji."
)


def build_prompt(base: str | None, caller: dict[str, str], context_notes: list[str] | None = None) -> str:
    """The agent prompt: configured base instructions (or the default), per-call
    caller context, and any live context notes (participants, DTMF, active
    speaker) appended as a bounded rolling section - the Voice Agent API has no
    non-interrupting context message, so context rides UpdatePrompt instead.

    caller: {"caller_name", "tenant_id", "direction"}."""
    lines = [
        (base or "").strip() or DEFAULT_INSTRUCTIONS,
        "",
        f"Call context: you are speaking with {caller['caller_name']} "
        f"(tenant: {caller['tenant_id']}) on an {caller['direction']} call.",
    ]
    if context_notes:
        lines.append("")
        lines.append("Live call context (most recent last):")
        lines.extend(f"- {note}" for note in context_notes)
    return "\n".join(lines)


def build_settings(
    cfg: BridgeConfig,
    prompt: str,
    extra_functions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The Settings message sent once per call, right after Welcome. Audio is
    pinned to linear16 @ 16 kHz both ways (the StandIn wire rate - copy-only
    relay, no transcoding), container "none" (raw frames). Language lives on
    the listen/speak providers (the top-level agent.language field is
    deprecated in the Voice Agent API)."""
    think: dict[str, Any] = {
        "provider": {"type": cfg.think_provider, "model": cfg.think_model},
        "prompt": prompt,
        "functions": [*BRIDGE_FUNCTIONS, *(extra_functions or [])],
    }
    if cfg.think_endpoint_url:
        endpoint: dict[str, Any] = {"url": cfg.think_endpoint_url}
        if cfg.think_endpoint_headers:
            endpoint["headers"] = cfg.think_endpoint_headers
        think["endpoint"] = endpoint
    agent: dict[str, Any] = {
        "listen": {"provider": {"type": "deepgram", "model": cfg.listen_model, "language": cfg.language}},
        "think": think,
        "speak": {"provider": {"type": "deepgram", "model": cfg.speak_model, "language": cfg.language}},
    }
    if cfg.greeting:
        agent["greeting"] = cfg.greeting
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": WIRE_SAMPLE_RATE},
            "output": {"encoding": "linear16", "sample_rate": WIRE_SAMPLE_RATE, "container": "none"},
        },
        "agent": agent,
    }


# ---- REST helper (deterministic goodbye TTS) ----


async def synthesize_goodbye(cfg: BridgeConfig, text: str) -> bytes:
    """Standalone Aura TTS for the deterministic governor goodbye: synthesize
    the exact text as raw linear16 @ 16 kHz and return the bytes. Only used when
    DEEPGRAM_TTS_MODEL is set; the fallback speaks through the live agent via
    InjectAgentMessage instead."""
    if not cfg.tts_model:
        raise RuntimeError("DEEPGRAM_TTS_MODEL not configured")
    params = {
        "model": cfg.tts_model,
        "encoding": "linear16",
        "sample_rate": str(WIRE_SAMPLE_RATE),
        "container": "none",
    }
    url = f"https://{cfg.api_host}/v1/speak?{urlencode(params)}"
    timeout = aiohttp.ClientTimeout(total=DG_REST_TIMEOUT_MS / 1000)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            json={"text": text},
            headers={"authorization": f"Token {cfg.deepgram_api_key}"},
        ) as res:
            if res.status != 200:
                raise RuntimeError(f"TTS failed: HTTP {res.status} {await res.text()}")
            return await res.read()


# ---- Voice Agent WebSocket session ----


class DgSessionHandlers:
    """Callbacks the session wires into the agent socket."""

    __slots__ = ("on_message", "on_audio", "on_close", "on_error")

    def __init__(
        self,
        on_message: Callable[[dict[str, Any]], None],
        on_audio: Callable[[bytes], None],
        on_close: Callable[[int, str], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.on_message = on_message
        self.on_audio = on_audio
        self.on_close = on_close
        self.on_error = on_error


class AgentPort(Protocol):
    """What the relay needs from an agent connection; DeepgramAgentSocket is the
    real one, tests fake it."""

    @property
    def is_open(self) -> bool: ...

    def send_audio_chunk(self, base64_pcm: str) -> None: ...
    def send_settings(self, settings: dict[str, Any]) -> None: ...
    def update_prompt(self, prompt: str) -> None: ...
    def inject_agent_message(self, text: str) -> None: ...
    def send_function_call_response(self, call_id: str, name: str, content: str) -> None: ...
    def close(self) -> None: ...


# Async factory signature tests can substitute for a fake agent.
DgConnector = Callable[[BridgeConfig, Logger, DgSessionHandlers], Awaitable[AgentPort]]


class DeepgramAgentSocket:
    """One Voice Agent socket. Thin: framing + send helpers only; relay logic
    lives in session.py."""

    def __init__(self, cfg: BridgeConfig, log: Logger) -> None:
        self._cfg = cfg
        self._log = log
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._read_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        # backpressure bookkeeping for the fire-and-forget send path
        self._pending_send_bytes = 0
        self._dropped_chunks = 0
        self._last_drop_warn = 0.0

    @classmethod
    async def connect(cls, cfg: BridgeConfig, log: Logger, handlers: DgSessionHandlers) -> "DeepgramAgentSocket":
        """Open the agent WS and wire handlers. Resolves once the server's
        Welcome has arrived (the Settings message may be sent from then on).
        One retry on a transient connect failure."""
        sock = cls(cfg, log)
        try:
            await sock._open_once()
        except Exception as err:
            log.warn(f"Deepgram connect failed ({err}); retrying once")
            await sock._dispose_transport()
            await asyncio.sleep(0.25)
            try:
                await sock._open_once()
            except Exception:
                await sock._dispose_transport()
                raise
        sock._read_task = asyncio.create_task(sock._read_loop(handlers))
        # The socket idles out without periodic KeepAlive when no audio is
        # flowing (hold music, silence). Cheap; sent for the call's lifetime.
        sock._keepalive_task = asyncio.create_task(sock._keepalive_loop())
        return sock

    async def _open_once(self) -> None:
        url = f"wss://{self._cfg.agent_host}/v1/agent/converse"
        # Bound the WS open: a blackholed TCP connect or a stalled TLS/upgrade
        # handshake must not hang session.start forever (the governor is only
        # armed after connect).
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self._ws = await asyncio.wait_for(
            self._session.ws_connect(
                url,
                headers={"authorization": f"Token {self._cfg.deepgram_api_key}"},
                max_msg_size=16 * 1024 * 1024,
            ),
            timeout=DG_REST_TIMEOUT_MS / 1000,
        )
        # Gate on the server's Welcome: the Settings message may only be sent
        # after it. Read frames until Welcome (bounded), ignoring junk.
        ws = self._ws
        deadline = time.monotonic() + WELCOME_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("no Welcome from Deepgram within the timeout")
            frame = await asyncio.wait_for(ws.receive(), timeout=remaining)
            if frame.type == aiohttp.WSMsgType.TEXT:
                try:
                    msg = json.loads(frame.data)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "Welcome":
                    return
            elif frame.type == aiohttp.WSMsgType.BINARY:
                continue  # audio cannot arrive before Settings; ignore defensively
            elif frame.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                raise RuntimeError(f"socket closed before Welcome ({frame.data})")
            elif frame.type == aiohttp.WSMsgType.ERROR:
                raise ws.exception() or RuntimeError("websocket error before Welcome")

    async def _dispose_transport(self) -> None:
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
            ws = self._ws
            if ws is None or ws.closed:
                return
            self._send({"type": "KeepAlive"})

    async def _read_loop(self, handlers: DgSessionHandlers) -> None:
        ws = self._ws
        assert ws is not None
        close_code = 1000
        close_reason = ""
        try:
            while True:
                frame = await ws.receive()
                if frame.type == aiohttp.WSMsgType.BINARY:
                    # Agent audio: raw linear16 @ 16 kHz frames.
                    try:
                        handlers.on_audio(frame.data)
                    except Exception as err:
                        self._log.error(f"error handling agent audio: {err}")
                elif frame.type == aiohttp.WSMsgType.TEXT:
                    try:
                        msg = json.loads(frame.data)
                    except ValueError:
                        self._log.warn("Deepgram sent an unparseable text frame; dropping")
                        continue
                    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
                        self._log.warn("Deepgram sent a non-object frame; dropping")
                        continue
                    try:
                        handlers.on_message(msg)
                    except Exception as err:
                        # Never let a handler error escape the read loop - it
                        # would silently kill the relay for this call.
                        self._log.error(f"error handling Deepgram {msg.get('type')}: {err}")
                elif frame.type == aiohttp.WSMsgType.CLOSE:
                    close_code = frame.data or close_code
                    close_reason = frame.extra or ""
                    break
                elif frame.type in (aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    break
                elif frame.type == aiohttp.WSMsgType.ERROR:
                    handlers.on_error(ws.exception() or RuntimeError("Deepgram websocket error"))
                    break
        except Exception as err:  # transport-level failure
            handlers.on_error(err if isinstance(err, Exception) else RuntimeError(str(err)))
        finally:
            close_code = ws.close_code or close_code
            await self._dispose_transport()
            handlers.on_close(close_code, close_reason)

    @property
    def is_open(self) -> bool:
        return self._ws is not None and not self._ws.closed

    def _send(self, obj: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        payload = json.dumps(obj)
        self._pending_send_bytes += len(payload)
        asyncio.ensure_future(self._send_str_safe(ws, payload))

    async def _send_str_safe(self, ws: aiohttp.ClientWebSocketResponse, payload: str) -> None:
        try:
            await ws.send_str(payload)
        except Exception:
            pass  # socket died mid-send; the read loop reports the close
        finally:
            self._pending_send_bytes -= len(payload)

    def send_audio_chunk(self, base64_pcm: str) -> None:
        """Caller audio -> agent, as a raw binary frame (base64 wire payload
        decoded, no transcoding). Droppable under backpressure - realtime audio
        is worthless stale, and a stalled agent socket must not pile up
        unbounded send tasks."""
        ws = self._ws
        if ws is None or ws.closed:
            return
        if self._pending_send_bytes > MAX_DG_SEND_BUFFER_BYTES:
            self._dropped_chunks += 1
            now = time.monotonic()
            if now - self._last_drop_warn >= 1:
                self._log.warn(
                    f"Deepgram send backpressure: dropped {self._dropped_chunks} chunk(s) "
                    f"(buffered {self._pending_send_bytes} bytes)"
                )
                self._last_drop_warn = now
                self._dropped_chunks = 0
            return
        try:
            data = base64.b64decode(base64_pcm)
        except Exception:
            return  # junk payload; nothing to relay
        self._pending_send_bytes += len(data)
        asyncio.ensure_future(self._send_bytes_safe(ws, data))

    async def _send_bytes_safe(self, ws: aiohttp.ClientWebSocketResponse, data: bytes) -> None:
        try:
            await ws.send_bytes(data)
        except Exception:
            pass  # socket died mid-send; the read loop reports the close
        finally:
            self._pending_send_bytes -= len(data)

    def send_settings(self, settings: dict[str, Any]) -> None:
        self._send(settings)

    def update_prompt(self, prompt: str) -> None:
        """Replace the agent's prompt mid-call (live context notes ride this)."""
        self._send({"type": "UpdatePrompt", "prompt": prompt})

    def inject_agent_message(self, text: str) -> None:
        """Make the agent speak this exact text (goodbye fallback). May be
        refused (InjectionRefused) while the agent is mid-utterance."""
        self._send({"type": "InjectAgentMessage", "message": text})

    def send_function_call_response(self, call_id: str, name: str, content: str) -> None:
        """Answer a client-side FunctionCallRequest."""
        self._send({"type": "FunctionCallResponse", "id": call_id, "name": name, "content": content})

    def close(self) -> None:
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        ws = self._ws
        if ws is not None and not ws.closed:
            asyncio.ensure_future(self._close_async())

    async def _close_async(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.close(code=1000, message=b"session-end")
        except Exception:
            pass
