"""One Teams call: pairs the worker WebSocket with one Deepgram Voice Agent
session and relays between them.

Audio is relayed VERBATIM in both directions - the wire is base64 PCM 16 kHz
and the Voice Agent session is pinned to linear16 @ 16 kHz, so the hot path is
copy-only (base64 <-> binary framing, no transcoding).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import time
from collections import deque
from typing import Any, Protocol

from .config import BridgeConfig
from .deepgram import (
    CustomTool,
    CustomToolContext,
    DeepgramAgentSocket,
    DgConnector,
    DgSessionHandlers,
    build_prompt,
    build_settings,
    custom_tool_schema,
    synthesize_goodbye,
)
from .log import logger
from .metrics import metric_inc
from .protocol import parse_worker_message, pcm16k_bytes_to_ms
from .ssrf import fetch_public_image
from .vision import VisionDescriber, make_vision_describer

# show_image fetch cap: display.image goes to a small video tile; 5 MB is generous.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Caller-audio buffer cap while the session becomes ready: 250 x 20 ms = 5 s.
MAX_PENDING_AUDIO_FRAMES = 250

# 20 ms of PCM 16 kHz mono 16-bit = 16000 * 0.02 * 2 = 640 bytes (one hot-path frame).
PCM16K_FRAME_BYTES = 640

# Outbound (bridge->worker) send-buffer cap. Above this, drop realtime frames
# instead of letting a stalled worker balloon memory.
MAX_OUTBOUND_BUFFER_BYTES = 1 * 1024 * 1024

# Live context notes kept in the prompt (participants/dtmf/speaker), most recent last.
MAX_CONTEXT_NOTES = 8

# Extra headroom on top of the goodbye grace before the governor force-ends the
# call, so a hung TTS synth can never wedge a time-limited call open.
GOODBYE_HARD_CAP_MS = 8_000

# Bound the SettingsApplied wait: a hung Settings application must not leave the
# call open and silent until the dead-peer watchdog.
SETTINGS_APPLIED_TIMEOUT_S = 10.0

# Min gap between "now speaking" context notes (group calls), so VAD flapping
# between speakers cannot spam the agent.
SPEAKER_UPDATE_MIN_INTERVAL_MS = 5_000

# Dead-peer window: worker heartbeats every 30 s -> 3 missed pings ends the call.
DEFAULT_WORKER_IDLE_TIMEOUT_MS = 90_000

# Inline show_image dataBase64 cap - same 5 MB bound as the URL path,
# expressed in base64 characters (4 chars per 3 bytes).
MAX_INLINE_IMAGE_B64_CHARS = -(-MAX_IMAGE_BYTES // 3) * 4

# Bounds on agent-supplied strings relayed to the worker as control frames,
# so a misbehaving model cannot emit multi-MB frames.
MAX_EMOTION_CHARS = 64
MAX_CAPTION_CHARS = 500
MAX_MODE_CHARS = 32

_IMAGE_MIME_RE = re.compile(r"^image/(jpeg|png)$")


def _now_ms() -> float:
    return time.monotonic() * 1000


class WorkerPort(Protocol):
    """What the session needs from the worker connection; the server provides
    the real one, tests fake it."""

    @property
    def is_open(self) -> bool: ...

    @property
    def buffered_bytes(self) -> int: ...

    def send_text(self, payload: str) -> None: ...
    def close(self, code: int, reason: str) -> None: ...


class CallSession:
    """Relay for a single authenticated worker connection.

    The server feeds inbound worker frames via handle_worker_message() and
    signals disconnect via handle_worker_close(); everything outbound goes
    through the WorkerPort.
    """

    def __init__(
        self,
        cfg: BridgeConfig,
        worker: WorkerPort,
        call_id: str,
        connect_dg: DgConnector | None = None,
        vision: VisionDescriber | None | str = "auto",
        tools: list[CustomTool] | None = None,
        on_closed: Any = None,
    ) -> None:
        self.cfg = cfg
        self.worker = worker
        self.call_id = call_id
        self.log = logger(f"call:{call_id[:12]}")
        self._connect_dg: DgConnector = connect_dg or DeepgramAgentSocket.connect
        self.vision: VisionDescriber | None = make_vision_describer(cfg) if vision == "auto" else vision  # type: ignore[assignment]
        self._custom_tools: dict[str, CustomTool] = {t.name: t for t in (tools or [])}
        self._on_closed = on_closed

        self.dg: Any = None
        self.closed = False
        self.session_started = False
        # The documented flow forbids audio before the server acks Settings.
        self.settings_applied = False
        self._settings_handle: asyncio.TimerHandle | None = None

        # outbound audio bookkeeping (bridge -> worker)
        self._out_seq = 0
        self._out_timestamp_ms = 0.0
        # backpressure-warn throttle (avoid ~50 warn lines/s when a worker stalls)
        self._dropped_frames = 0
        self._last_backpressure_warn_ms = 0.0

        # Barge-in ghost filter: on UserStartedSpeaking, Deepgram stops
        # generating TTS server-side, but frames already in flight still arrive.
        # Drop agent audio from the cut until the agent audibly starts its NEXT
        # utterance (AgentStartedSpeaking) - state-based, cannot leak memory.
        self._dropping_agent_audio = False
        # hard mute: set ONLY while a deterministic TTS goodbye plays (never for
        # the InjectAgentMessage fallback, which must stay audible)
        self._mute_agent_audio = False
        # first goodbye wins: both governors can race
        self._goodbye_in_progress = False
        # group-call speaker attribution + rate limit
        self._last_speaker_name: str | None = None
        self._last_speaker_update_ms = 0.0
        self._participant_count = 1
        # caller audio buffered from session.start through connect AND SettingsApplied
        self._pending_audio: deque[str] = deque(maxlen=MAX_PENDING_AUDIO_FRAMES)
        # per-call caller context for prompt (re)builds
        self._caller_ctx: dict[str, str] | None = None
        # live context notes carried in the prompt via UpdatePrompt (bounded)
        self._context_notes: deque[str] = deque(maxlen=MAX_CONTEXT_NOTES)

        # Teams recording gate: transcripts may be logged/persisted only when "active"
        self._recording_active = False

        # vision groundwork: latest inbound frame per source, memory only
        self._latest_video_frame: dict[str, dict[str, Any]] = {}

        # bridge-side call governor
        self._governor_handle: asyncio.TimerHandle | None = None
        self._goodbye_handle: asyncio.TimerHandle | None = None

        # Dead-peer detection: a half-open TCP socket (NAT timeout, peer crash)
        # delivers nothing and never closes - the session would stay "live" for
        # hours, holding the billed Voice Agent session open AND blocking every
        # reconnect for this callId with a 409.
        self._last_worker_activity_ms = _now_ms()
        idle_ms = cfg.worker_idle_timeout_ms if cfg.worker_idle_timeout_ms > 0 else DEFAULT_WORKER_IDLE_TIMEOUT_MS
        self._idle_ms = idle_ms
        self._idle_task = asyncio.create_task(self._idle_watchdog(max(0.02, min(idle_ms / 6000, 15.0))))

    # ---- lifecycle wiring (called by the server's read loop) ----

    @property
    def has_started(self) -> bool:
        """Whether session.start has arrived (the server's pre-start timer asks)."""
        return self.session_started

    def handle_worker_message(self, raw: str | bytes) -> None:
        self._last_worker_activity_ms = _now_ms()  # any inbound frame proves the peer is alive
        try:
            self._on_worker_message(raw)
        except Exception as err:
            # a handler error must never escape into the server's read loop
            self.log.error(f"error handling worker message: {err}")

    def handle_worker_close(self) -> None:
        self._teardown("worker-closed")

    def handle_worker_error(self, err: Exception) -> None:
        self.log.warn(f"worker socket error: {err}")
        self._teardown("worker-error")

    async def _idle_watchdog(self, interval_s: float) -> None:
        while not self.closed:
            await asyncio.sleep(interval_s)
            if self.closed:
                return
            if _now_ms() - self._last_worker_activity_ms > self._idle_ms:
                self.log.warn(f"no worker message in {int(self._idle_ms)}ms (dead peer?); ending the call")
                self.end_call("worker-idle-timeout")
                return

    # ---- worker -> bridge ----

    def _on_worker_message(self, raw: str | bytes) -> None:
        msg = parse_worker_message(raw)
        if msg is None:
            self.log.warn("unparseable worker frame; dropping")
            return
        mtype = msg["type"]
        if mtype == "session.start":
            if self.session_started:
                # A second session.start would orphan the first Voice Agent
                # session; the worker sends exactly one per connection.
                self.log.warn("duplicate session.start ignored")
                return
            # Mark started SYNCHRONOUSLY: audio frames can arrive between this
            # message and the scheduled coroutine's first step, and they must be
            # buffered (not dropped) for the flush after SettingsApplied.
            self.session_started = True
            asyncio.ensure_future(self._on_session_start_safe(msg))
        elif mtype == "audio.frame":
            # hot path: caller audio -> agent, verbatim (base64 -> binary
            # frame). Until the server acks Settings (SettingsApplied), buffer
            # (bounded) instead of sending: the documented flow forbids audio
            # before the ack, and this window also covers the connect itself.
            payload = msg.get("payloadBase64")
            if not isinstance(payload, str):
                return
            if self.dg is not None and self.settings_applied:
                self.dg.send_audio_chunk(payload)
                metric_inc("bridge_frames_to_agent_total")
                self._note_speaker(msg.get("speakerName"))
            elif self.session_started:
                self._pending_audio.append(payload)  # deque drops the oldest at cap
        elif mtype == "ping":
            self._send_to_worker({"type": "pong", "ts": msg.get("ts")})
        elif mtype == "participants":
            count = msg.get("count")
            if isinstance(count, (int, float)):
                self._participant_count = int(count)
                self._push_context(
                    "This is a 1:1 call with a single human caller."
                    if count <= 1
                    else f"There are {int(count)} human participants on this call. Stay quiet unless directly addressed."
                )
        elif mtype == "dtmf":
            digit = msg.get("digit")
            if isinstance(digit, str) and digit:
                self._push_context(f'The caller pressed the "{digit}" key on their keypad.')
        elif mtype == "recording.status":
            self._recording_active = msg.get("status") == "active"
            self.log.info(f"recording.status = {msg.get('status')}")
        elif mtype == "video.frame":
            # Known sources only (camera/screenshare): the key comes from the peer,
            # so an unexpected value must not grow the map unbounded.
            source = msg.get("source")
            if source in ("camera", "screenshare"):
                self._latest_video_frame[source] = msg  # buffered for on-demand vision; not persisted
            else:
                self.log.debug(f'ignoring video.frame with unknown source "{source}"')
        elif mtype == "assistant.say":
            # worker-side governor: speak, the worker tears down afterwards
            asyncio.ensure_future(self._perform_goodbye_safe(str(msg.get("text") or "")))
        elif mtype == "session.end":
            self.log.info(f"session.end from worker: {msg.get('reason')}")
            self._teardown("worker-session-end")
        else:
            self.log.debug(f"ignoring worker message type {mtype}")

    async def _on_session_start_safe(self, msg: dict[str, Any]) -> None:
        try:
            await self._on_session_start(msg)
        except Exception as err:
            self.log.error(f"session.start handling failed: {err}")

    async def _on_session_start(self, msg: dict[str, Any]) -> None:
        msg_call_id = msg.get("callId")
        if msg_call_id and msg_call_id != self.call_id:
            # must match the HMAC-authenticated callId in the URL path (wire contract).
            self.log.error(f"session.start callId {msg_call_id} != URL callId {self.call_id}; closing")
            self.end_call("callid-mismatch")
            return
        direction = msg.get("direction") or "inbound"
        recording = msg.get("recordingStatus") or "unknown"
        self.log.info(f"session.start (direction={direction}, recording={recording})")
        self._recording_active = recording == "active"
        # Per-call personalization: caller context lives in the prompt.
        # Caller fields are all nullable - default, never send null.
        caller = msg.get("caller") or {}
        self._caller_ctx = {
            "caller_name": (caller.get("displayName") or "").strip() or "caller",
            "tenant_id": (caller.get("tenantId") or "").strip() or "unknown-tenant",
            "direction": (msg.get("direction") or "").strip() or "inbound",
        }

        handlers = DgSessionHandlers(
            on_message=self._on_dg_message,
            on_audio=self._on_dg_audio,
            on_close=self._on_dg_close,
            on_error=lambda err: self.log.warn(f"Deepgram socket error: {err}"),
        )
        try:
            dg = await self._connect_dg(self.cfg, self.log, handlers)
        except Exception as err:
            metric_inc("bridge_agent_connect_failures_total")
            self.log.error(f"could not open Deepgram Voice Agent session: {err}")
            self.end_call("agent-unavailable")
            return

        # The worker may have dropped (ring cancelled, rollout) DURING the
        # connect above. Keeping the just-opened socket would orphan a live,
        # billed Voice Agent session.
        if self.closed:
            self.log.info("worker closed during Deepgram connect; closing the orphaned agent socket")
            try:
                dg.close()
            except Exception:
                pass
            return
        self.dg = dg

        # The one-time Settings message: audio pinned to linear16 @ 16 kHz both
        # ways, agent config, prompt (incl. any context notes that landed during
        # the connect window), greeting, and the functions list (built-ins +
        # embedder-registered custom tools).
        dg.send_settings(
            build_settings(
                self.cfg,
                prompt=build_prompt(self.cfg.instructions, self._caller_ctx, list(self._context_notes)),
                extra_functions=[custom_tool_schema(t) for t in self._custom_tools.values()],
            )
        )
        # Audio must wait for the server's SettingsApplied ack (see
        # _on_dg_message); bound that wait so a hung Settings application cannot
        # leave the call open and silent until the dead-peer watchdog.
        loop = asyncio.get_running_loop()

        def settings_check() -> None:
            if not self.settings_applied and not self.closed:
                self.log.error("no SettingsApplied from Deepgram within 10s; ending the call")
                self.end_call("agent-unavailable")

        self._settings_handle = loop.call_later(SETTINGS_APPLIED_TIMEOUT_S, settings_check)
        self.log.info("Deepgram Voice Agent session open; waiting for SettingsApplied")

        # Bridge-side governor: Deepgram doesn't know about your billing.
        if self.cfg.max_call_minutes > 0:
            limit_s = self.cfg.max_call_minutes * 60
            self._governor_handle = loop.call_later(
                limit_s, lambda: asyncio.ensure_future(self._on_governor_limit_safe())
            )
            self.log.info(f"governor armed: max {self.cfg.max_call_minutes:g} min")

    async def _on_governor_limit_safe(self) -> None:
        try:
            await self._on_governor_limit()
        except Exception as err:
            self.log.error(f"governor error: {err}")

    async def _on_governor_limit(self) -> None:
        """Time limit hit: speak the goodbye, let it play out, then tear the call down."""
        if self.closed:
            return
        self.log.info("governor: call time limit reached")
        # If the worker-side governor already started a goodbye, its
        # hard-bounded backstop is armed - do NOT overwrite that timer (the call
        # ends either way, and clobbering it could cut off a goodbye that is
        # still playing).
        if self._goodbye_in_progress:
            self.log.info("a goodbye is already in progress; keeping its deadline")
            return
        # Guarantee teardown regardless of the goodbye. Arm a HARD-bounded
        # deadline BEFORE awaiting the goodbye - a hung/slow TTS must never
        # wedge the call open past its limit.
        hard_ms = self.cfg.goodbye_grace_ms + GOODBYE_HARD_CAP_MS
        loop = asyncio.get_running_loop()
        self._goodbye_handle = loop.call_later(hard_ms / 1000, lambda: self.end_call("time-limit"))
        played_ms = await self._perform_goodbye(self.cfg.goodbye_text)
        if self.closed:
            return  # the hard deadline (or another path) already tore down
        # Deterministic TTS reports its real duration; the agent-spoken fallback
        # does not. Reschedule to the real grace, but never later than the hard cap.
        grace_ms = min(played_ms if played_ms is not None else self.cfg.goodbye_grace_ms, hard_ms)
        if self._goodbye_handle:
            self._goodbye_handle.cancel()
        self._goodbye_handle = loop.call_later((grace_ms + 500) / 1000, lambda: self.end_call("time-limit"))

    def _note_speaker(self, name: Any) -> None:
        """Group-call speaker attribution: the worker tags audio.frame with the
        active speaker's display name. Surface it as a context note - only in
        group calls (1:1 attribution is noise), only when the name CHANGES, and
        rate-limited so VAD flapping between speakers cannot spam the agent."""
        if not name or not isinstance(name, str) or self._participant_count <= 1:
            return
        now = _now_ms()
        if name == self._last_speaker_name or now - self._last_speaker_update_ms < SPEAKER_UPDATE_MIN_INTERVAL_MS:
            return
        self._last_speaker_name = name
        self._last_speaker_update_ms = now
        self._push_context(f"The person now speaking is {name}.")

    def _push_context(self, note: str) -> None:
        """Record a live context note (participants/dtmf/speaker) and push the
        rebuilt prompt to the agent. The Voice Agent API has no non-interrupting
        context message, so context rides UpdatePrompt: base instructions +
        caller context + a bounded rolling notes section. Notes recorded while
        the socket is still connecting simply land in the initial Settings prompt."""
        if self.closed:
            return
        self._context_notes.append(note)  # deque drops the oldest at cap
        if self.dg is not None and self._caller_ctx is not None:
            self.dg.update_prompt(build_prompt(self.cfg.instructions, self._caller_ctx, list(self._context_notes)))

    # ---- Deepgram -> bridge ----

    def _on_dg_close(self, code: int, reason: str) -> None:
        self.log.info(f"Deepgram socket closed ({code} {reason})")
        self.end_call("agent-disconnected")

    def _on_dg_audio(self, pcm: bytes) -> None:
        """Agent audio (raw binary linear16 @ 16 kHz): ghost/mute filter, then relay."""
        if self._mute_agent_audio:
            self.log.debug("dropping agent audio (deterministic goodbye playing)")
            return
        if self._dropping_agent_audio:
            self.log.debug("dropping ghost agent audio (after barge-in / goodbye flush)")
            return
        self._emit_audio_to_worker(base64.b64encode(pcm).decode("ascii"))

    def _on_dg_message(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "SettingsApplied":
            # The server is ready for audio (documented ordering contract).
            # Flush the caller speech buffered since session.start, oldest first.
            self.settings_applied = True
            if self._settings_handle:
                self._settings_handle.cancel()
                self._settings_handle = None
            if self.dg is not None:
                while self._pending_audio:
                    self.dg.send_audio_chunk(self._pending_audio.popleft())
                    metric_inc("bridge_frames_to_agent_total")
            self.log.info("SettingsApplied; relaying")
        elif mtype == "UserStartedSpeaking":
            # Caller barge-in. Deepgram stops generating TTS server-side; mirror
            # the cut to the Teams side and ghost-drop frames still in flight
            # until the agent audibly starts its next utterance.
            self._dropping_agent_audio = True
            # turnId is not tracked by this bridge; the worker's flush ignores the value.
            self._send_to_worker({"type": "assistant.cancel", "turnId": 0})
            self.log.info("barge-in: caller speech started")
        elif mtype == "AgentStartedSpeaking":
            # The agent's next utterance begins: stop ghost-dropping (the mute
            # latch, if set by a deterministic goodbye, still wins in _on_dg_audio).
            self._dropping_agent_audio = False
        elif mtype == "FunctionCallRequest":
            calls = msg.get("functions")
            if not isinstance(calls, list):
                self.log.warn("FunctionCallRequest without a functions array; dropping")
                return
            for call in calls:
                if not isinstance(call, dict):
                    continue
                # Server-side functions (declared with an endpoint) are executed
                # by Deepgram itself; only client_side calls are ours to answer.
                if call.get("client_side") is False:
                    continue
                if not isinstance(call.get("id"), str) or not isinstance(call.get("name"), str):
                    self.log.warn("client-side function call missing id/name; dropping")
                    continue
                self._on_function_call(call)
        elif mtype == "ConversationText":
            # Recording gate: never log/persist transcripts unless Teams recording is active.
            if self.cfg.log_transcripts and self._recording_active:
                self.log.info("ConversationText", {"role": msg.get("role"), "content": msg.get("content")})
        elif mtype == "InjectionRefused":
            # The goodbye fallback can be refused while the agent is
            # mid-utterance; the goodbye grace/backstop still ends the call.
            self.log.warn(f"InjectionRefused: {msg.get('message') or 'no detail'}")
        elif mtype == "Error":
            self.log.warn(
                f"Deepgram error event: {msg.get('code') or 'unknown'}: {msg.get('description') or 'no description'}"
            )
        elif mtype == "Warning":
            self.log.warn(f"Deepgram warning: {msg.get('description') or 'no description'}")
        elif mtype in ("Welcome", "AgentThinking", "AgentAudioDone", "PromptUpdated", "History"):
            self.log.debug(f"Deepgram event: {mtype}")
        else:
            self.log.debug(f"ignoring Deepgram event type {mtype}")

    def _on_function_call(self, call: dict[str, Any]) -> None:
        """Map agent client-side functions -> worker capabilities:
        end_call -> session.end, express -> expression,
        show_image -> display.image, look -> vision."""
        call_id: str = call["id"]
        name: str = call["name"]
        raw_args = call.get("arguments")
        params: dict[str, Any] = {}
        # Docs say arguments is a JSON-encoded string; tolerate an already-parsed object.
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    params = parsed
            except ValueError:
                self.log.warn(f"unparseable arguments for tool {name}; treating as empty")
        elif isinstance(raw_args, dict):
            params = raw_args

        if name == "end_call":
            self._reply_tool(call_id, name, "call ended")
            self.log.info("agent requested end_call")
            self.end_call("agent-ended-call")
        elif name == "express":
            emotion = params.get("emotion") if isinstance(params.get("emotion"), str) else ""
            emotion = (emotion or "").strip()
            if not emotion:
                self._reply_tool(call_id, name, "express requires an 'emotion' parameter")
                return
            if len(emotion) > MAX_EMOTION_CHARS:
                self._reply_tool(call_id, name, f"express: 'emotion' must be at most {MAX_EMOTION_CHARS} characters")
                return
            self._send_to_worker({"type": "expression", "emotion": emotion})
            self._reply_tool(call_id, name, f"expressing {emotion}")
        elif name == "show_image":
            asyncio.ensure_future(self._on_show_image(call_id, name, params))
        elif name == "look":
            asyncio.ensure_future(self._on_look(call_id, name, params))
        elif name in self._custom_tools:
            # Embedder-registered custom tools (the extensibility surface): run
            # the handler, return its string as the function output. A raise
            # becomes an error output so the model can recover; it must never
            # escape into the read loop.
            asyncio.ensure_future(self._run_custom_tool(self._custom_tools[name], call_id, params))
        else:
            self._reply_tool(call_id, name, f'tool "{name}" is not implemented by this bridge')
            self.log.warn(f"unmapped function tool: {name}")

    async def _run_custom_tool(self, tool: CustomTool, call_id: str, params: dict[str, Any]) -> None:
        ctx = CustomToolContext(
            call_id=self.call_id,
            participant_count=self._participant_count,
            recording_active=self._recording_active,
            log=self.log,
        )
        try:
            result = tool.handler(params, ctx)
            if inspect.isawaitable(result):
                result = await result
            self._reply_tool(call_id, tool.name, str(result))
        except Exception as err:
            self.log.warn(f"custom tool {tool.name} failed: {err}")
            self._reply_tool(call_id, tool.name, f'tool "{tool.name}" failed: {err}')

    def _reply_tool(self, call_id: str, name: str, content: str) -> None:
        """Answer a client-side function call; the agent continues on its own."""
        if self.dg is not None:
            self.dg.send_function_call_response(call_id, name, content)

    async def _on_show_image(self, call_id: str, name: str, params: dict[str, Any]) -> None:
        """show_image -> display.image on the bot's video tile. Accepts either
        inline base64 ({dataBase64, mime}) or a URL the bridge fetches server-side."""
        try:
            data_base64 = params.get("dataBase64") if isinstance(params.get("dataBase64"), str) else None
            if data_base64 and len(data_base64) > MAX_INLINE_IMAGE_B64_CHARS:
                raise ValueError(
                    f"inline image too large ({len(data_base64)} base64 chars, max {MAX_INLINE_IMAGE_B64_CHARS})"
                )
            mime = params.get("mime") if isinstance(params.get("mime"), str) else None
            url = params.get("url") if isinstance(params.get("url"), str) else None
            if not data_base64 and url:
                # SSRF guard: the URL is agent-(LLM-)controlled, i.e. indirectly
                # caller-controlled. fetch_public_image validates the host, PINS
                # the connect-time DNS resolution through the same private-range
                # check, and follows at most ONE re-validated redirect. Bounded
                # time and size.
                img_bytes, mime = await fetch_public_image(url, MAX_IMAGE_BYTES, 10_000)
                data_base64 = base64.b64encode(img_bytes).decode("ascii")
            if not data_base64 or not mime or not _IMAGE_MIME_RE.match(mime):
                raise ValueError("show_image needs {dataBase64, mime} or {url} resolving to image/jpeg or image/png")
            mode = params.get("mode") if isinstance(params.get("mode"), str) else None
            caption = params.get("caption") if isinstance(params.get("caption"), str) else None
            delivered = self._send_to_worker(
                {
                    "type": "display.image",
                    "dataBase64": data_base64,
                    "mime": mime,
                    "durationMs": params.get("durationMs")
                    if isinstance(params.get("durationMs"), (int, float))
                    else None,
                    "mode": mode[:MAX_MODE_CHARS] if mode else None,
                    "ts": 0,
                    "caption": caption[:MAX_CAPTION_CHARS] if caption else None,
                }
            )
            # Tell the agent the truth: a frame dropped under backpressure was
            # NOT shown - claiming success would leave it talking about an image
            # the caller never saw.
            if not delivered:
                raise ValueError("image could not be delivered (worker connection is congested); try again")
            self._reply_tool(call_id, name, "image is being shown to the caller")
        except Exception as err:
            self.log.warn(f"show_image failed: {err}")
            self._reply_tool(call_id, name, f"show_image failed: {err}")

    async def _on_look(self, call_id: str, name: str, params: dict[str, Any]) -> None:
        """Vision on demand - agent function `look`
        ({source?: "camera"|"screenshare", question?: string}).

        The Voice Agent API is audio-only (no image input), so there is exactly
        one route: describe the buffered frame via YOUR vision model
        (VISION_API_URL or a custom VisionDescriber) and answer in the function
        result. The raw frame is sent to that endpoint - never to Deepgram.
        Without a vision endpoint the tool reports that vision is unavailable."""
        requested = params.get("source") if isinstance(params.get("source"), str) else None
        frame = (
            (self._latest_video_frame.get(requested) if requested else None)
            or self._latest_video_frame.get("screenshare")
            or self._latest_video_frame.get("camera")
        )
        if frame is None:
            self._reply_tool(call_id, name, "no video is available - the caller has not shared their camera or screen")
            return
        if self.vision is None:
            self._reply_tool(
                call_id,
                name,
                "cannot inspect video: no vision endpoint is configured on this bridge (set VISION_API_URL)",
            )
            return
        # Optional compliance gate: camera/screen frames are PII-bearing, so
        # deployments can require Teams recording to be active before any frame
        # is sent to the vision endpoint.
        if self.cfg.vision_requires_recording and not self._recording_active:
            self._reply_tool(
                call_id,
                name,
                "cannot inspect video: Teams recording is not active and this bridge requires recording "
                "before frames may be processed",
            )
            return
        question = params.get("question") if isinstance(params.get("question"), str) else ""
        question = (question or "").strip() or "Describe what is visible."
        try:
            description = await self.vision(frame, question)
            self._reply_tool(call_id, name, description)
        except Exception as err:
            self.log.warn(f"look failed: {err}")
            self._reply_tool(call_id, name, f"look failed: {err}")

    # ---- governor goodbye ----

    async def _perform_goodbye_safe(self, text: str) -> None:
        try:
            await self._perform_goodbye(text)
        except Exception as err:
            self.log.error(f"goodbye failed: {err}")

    async def _perform_goodbye(self, text: str) -> float | None:
        """Speak a goodbye line (both governors: worker assistant.say and the
        bridge-side time limit). Flushes buffered playback first
        (assistant.cancel to the worker + ghost-drop in-flight agent frames) so
        stale agent audio cannot delay the goodbye.

        Preferred: deterministic, the exact text via standalone Aura TTS
        (DEEPGRAM_TTS_MODEL) - the agent is hard-muted while it plays and the
        real duration (ms) is returned. Fallback: the live agent speaks the
        exact text via InjectAgentMessage - its audio MUST keep relaying (mute
        stays off), duration unknown (None)."""
        if self._goodbye_in_progress:
            # Both governors can race; running twice would double-speak and leave
            # the mute latch in an ambiguous state - first one wins.
            self.log.info("goodbye already in progress; ignoring duplicate")
            return None
        self._goodbye_in_progress = True
        self.log.info("speaking goodbye")
        # Backstop teardown for the WORKER-side governor path (assistant.say):
        # the worker is expected to tear the call down after the goodbye, but if
        # it is buggy/slow the call must not sit open (agent muted) until the
        # dead-peer watchdog. The bridge-side governor arms its own tighter
        # deadline first, in which case this is skipped.
        if self._goodbye_handle is None:
            loop = asyncio.get_running_loop()
            self._goodbye_handle = loop.call_later(
                (self.cfg.goodbye_grace_ms + GOODBYE_HARD_CAP_MS) / 1000,
                lambda: self.end_call("goodbye-timeout"),
            )
        self._send_to_worker({"type": "assistant.cancel", "turnId": 0})
        # Flush in-flight agent frames; AgentStartedSpeaking (the injected
        # goodbye, or nothing) clears this.
        self._dropping_agent_audio = True
        if self.cfg.tts_model:
            try:
                self._mute_agent_audio = True  # only the deterministic goodbye may speak now
                pcm = await synthesize_goodbye(self.cfg, text)
                # Emit as 20 ms frames like the hot path, rather than one
                # multi-second frame, so playback does not depend on the worker
                # re-aligning a giant chunk. The goodbye is the LAST thing the
                # caller hears - a load-bearing utterance, never dropped under
                # worker backpressure (undroppable), unlike the normal hot path.
                for off in range(0, len(pcm), PCM16K_FRAME_BYTES):
                    chunk = pcm[off : off + PCM16K_FRAME_BYTES]
                    self._emit_audio_to_worker(base64.b64encode(chunk).decode("ascii"), undroppable=True)
                played_ms = pcm16k_bytes_to_ms(len(pcm))
                # Unmute once the goodbye has played out. Normally the call ends
                # first - but if a peer fails to tear down, the agent must not
                # stay silently muted for the rest of the call.
                asyncio.get_running_loop().call_later(
                    (played_ms + 250) / 1000, lambda: setattr(self, "_mute_agent_audio", False)
                )
                return played_ms
            except Exception as err:
                self._mute_agent_audio = False  # fallback: the agent must stay audible
                self.log.warn(f"goodbye TTS failed ({err}); falling back to InjectAgentMessage")
        # The live agent speaks the exact text in its own voice. May be refused
        # (InjectionRefused) if the agent is mid-utterance; the goodbye backstop
        # still ends the call.
        if self.dg is not None:
            self.dg.inject_agent_message(text)
        return None

    # ---- plumbing ----

    def _emit_audio_to_worker(self, base64_pcm: str, undroppable: bool = False) -> None:
        frame = {
            "type": "audio.frame",
            "seq": self._out_seq,
            "timestampMs": round(self._out_timestamp_ms),
            "payloadBase64": base64_pcm,
        }
        self._out_seq += 1
        # advance the timeline by the actual PCM duration - exact decoded length
        self._out_timestamp_ms += pcm16k_bytes_to_ms(len(base64.b64decode(base64_pcm)))
        metric_inc("bridge_frames_to_worker_total")
        self._send_to_worker(frame, undroppable=undroppable)

    def _send_to_worker(self, msg: dict[str, Any], undroppable: bool = False) -> bool:
        """Send one frame; False when the frame was dropped (socket closed or
        realtime backpressure), True when it was queued for delivery."""
        if not self.worker.is_open:
            return False
        # Backpressure guard: if the worker stalls, the outbound buffer grows
        # unbounded (50 audio.frames/s) and leaks memory. Above the cap, drop
        # this frame rather than queue it - audio is realtime, a stale frame is
        # worthless. ONLY the continuous hot-path type (audio.frame) is
        # droppable: control frames (assistant.cancel, session.end, pong,
        # expression) are tiny and semantically load-bearing, display.image is a
        # one-shot the agent is told about, and goodbye TTS frames are marked
        # undroppable by the caller (the last thing the caller hears).
        droppable = msg.get("type") == "audio.frame" and not undroppable
        if droppable and self.worker.buffered_bytes > MAX_OUTBOUND_BUFFER_BYTES:
            self._dropped_frames += 1
            metric_inc("bridge_frames_dropped_total")
            now = _now_ms()
            # Throttle the log: warn at most once per second with the total.
            if now - self._last_backpressure_warn_ms >= 1000:
                self.log.warn(
                    f"worker send backpressure: dropped {self._dropped_frames} frame(s) "
                    f"(buffered {self.worker.buffered_bytes} bytes)"
                )
                self._last_backpressure_warn_ms = now
                self._dropped_frames = 0
            return False
        self.worker.send_text(json.dumps(msg))
        return True

    def shutdown(self, reason: str) -> None:
        """Graceful external shutdown (e.g. SIGTERM drain): tell the worker the
        call is ending, then close both sockets. Idempotent via the closed flag."""
        self.end_call(reason)

    def end_call(self, reason: str) -> None:
        """Ask the worker to tear the call down, then close both sockets."""
        if not self.closed:
            self._send_to_worker({"type": "session.end", "reason": reason})
        self._teardown(reason)

    def _teardown(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        self.log.info(f"teardown: {reason}")
        # symmetry: the mute latch must never outlive the goodbye that set it
        self._mute_agent_audio = False
        for handle_name in ("_governor_handle", "_goodbye_handle", "_settings_handle"):
            handle = getattr(self, handle_name)
            if handle:
                handle.cancel()
                setattr(self, handle_name, None)
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        try:
            if self.dg is not None:
                self.dg.close()
        except Exception:
            pass
        try:
            self.worker.close(1000, reason)
        except Exception:
            pass
        self._latest_video_frame.clear()
        self._pending_audio.clear()
        self._context_notes.clear()
        # let the server de-register this call (registry eviction, dup-callId dedup)
        try:
            if self._on_closed is not None:
                self._on_closed()
        except Exception:
            pass  # registry callback must never raise back into teardown
