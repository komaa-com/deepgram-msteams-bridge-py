"""Session relay tests against fake worker + agent ports (no network)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from conftest import FakeAgentPort, FakeWorkerPort, make_config, settle

import deepgram_msteams_bridge.session as session_mod
from deepgram_msteams_bridge.deepgram import CustomTool, DgSessionHandlers
from deepgram_msteams_bridge.session import CallSession

PCM_640 = base64.b64encode(bytes(640)).decode("ascii")


class Harness:
    """One CallSession wired to fakes; captures the handlers the session gave
    the connector so tests can push agent events/audio."""

    def __init__(self, cfg: Any = None, vision: Any = None, tools: list[CustomTool] | None = None) -> None:
        self.cfg = cfg or make_config()
        self.worker = FakeWorkerPort()
        self.agent = FakeAgentPort()
        self.handlers: DgSessionHandlers | None = None

        async def connect(cfg: Any, log: Any, handlers: DgSessionHandlers) -> FakeAgentPort:
            self.handlers = handlers
            return self.agent

        self.session = CallSession(self.cfg, self.worker, "call-1", connect_dg=connect, vision=vision, tools=tools)

    def worker_says(self, msg: dict) -> None:
        self.session.handle_worker_message(json.dumps(msg))

    async def start(self, caller: dict | None = None, apply_settings: bool = True) -> None:
        self.worker_says(
            {
                "type": "session.start",
                "callId": "call-1",
                "threadId": "t",
                "caller": caller or {},
                "direction": "inbound",
            }
        )
        await settle()
        if apply_settings:
            assert self.handlers is not None
            self.handlers.on_message({"type": "SettingsApplied"})
            await settle()

    def agent_event(self, msg: dict) -> None:
        assert self.handlers is not None
        self.handlers.on_message(msg)

    def agent_audio(self, pcm: bytes) -> None:
        assert self.handlers is not None
        self.handlers.on_audio(pcm)

    def function_call(self, call_id: str, name: str, arguments: Any = "{}", client_side: bool = True) -> None:
        self.agent_event(
            {
                "type": "FunctionCallRequest",
                "functions": [{"id": call_id, "name": name, "arguments": arguments, "client_side": client_side}],
            }
        )

    def tool_results(self) -> dict[str, tuple[str, str]]:
        return {cid: (name, content) for cid, name, content in self.agent.of_type("tool_result")}


async def test_settings_content_and_ordering_gate() -> None:
    h = Harness()
    h.worker_says(
        {
            "type": "session.start",
            "callId": "call-1",
            "threadId": "t",
            "caller": {"displayName": "Alaa", "tenantId": None, "aadId": None},
            "direction": "inbound",
        }
    )
    await settle()
    settings = h.agent.of_type("settings")[0]
    assert settings["type"] == "Settings"
    audio = settings["audio"]
    assert audio["input"] == {"encoding": "linear16", "sample_rate": 16_000}
    assert audio["output"] == {"encoding": "linear16", "sample_rate": 16_000, "container": "none"}
    agent = settings["agent"]
    # language rides the providers, not the deprecated top-level agent.language
    assert "language" not in agent
    assert agent["listen"]["provider"]["language"] == "en"
    assert agent["speak"]["provider"]["language"] == "en"
    prompt = agent["think"]["prompt"]
    assert "Alaa" in prompt
    assert "unknown-tenant" in prompt  # nullable tenant defaulted, never null
    assert "inbound call" in prompt
    names = sorted(f["name"] for f in agent["think"]["functions"])
    assert names == ["end_call", "express", "look", "show_image"]
    assert "greeting" not in agent  # not configured -> omitted
    assert "endpoint" not in agent["think"]  # no BYO-LLM endpoint -> omitted

    # documented ordering contract: NO audio before the server acks Settings
    h.worker_says({"type": "audio.frame", "seq": 0, "timestampMs": 0, "payloadBase64": "ZWFybHk="})
    await settle()
    assert h.agent.audio == [], "caller audio must be buffered until SettingsApplied"
    h.agent_event({"type": "SettingsApplied"})
    await settle()
    assert h.agent.audio == ["ZWFybHk="], "buffered audio flushes on the ack"

    # live audio flows after the ack
    h.worker_says({"type": "audio.frame", "seq": 1, "timestampMs": 20, "payloadBase64": PCM_640})
    await settle()
    assert h.agent.audio == ["ZWFybHk=", PCM_640]


async def test_settings_carries_greeting_and_think_endpoint() -> None:
    cfg = make_config(
        greeting="Hello!",
        think_provider="google",
        think_model="gemini-2.0-flash",
        think_endpoint_url="https://llm.example.com/v1/chat",
        think_endpoint_headers={"authorization": "Bearer tok"},
    )
    h = Harness(cfg=cfg)
    await h.start(apply_settings=False)
    agent = h.agent.of_type("settings")[0]["agent"]
    assert agent["greeting"] == "Hello!"
    assert agent["think"]["provider"] == {"type": "google", "model": "gemini-2.0-flash"}
    assert agent["think"]["endpoint"] == {
        "url": "https://llm.example.com/v1/chat",
        "headers": {"authorization": "Bearer tok"},
    }


async def test_agent_audio_relay_and_barge_in_ghost_filter() -> None:
    h = Harness()
    await h.start()

    # agent audio (binary) -> worker audio.frame with seq/timestamp bookkeeping
    h.agent_audio(bytes(640))
    h.agent_audio(bytes(640))
    await settle()
    frames = h.worker.of_type("audio.frame")
    assert [f["seq"] for f in frames] == [0, 1]
    assert [f["timestampMs"] for f in frames] == [0, 20]  # advanced by the real duration

    # barge-in: UserStartedSpeaking -> assistant.cancel; in-flight agent audio is
    # ghost-dropped until the agent's NEXT utterance starts
    h.agent_event({"type": "UserStartedSpeaking"})
    await settle()
    cancels = h.worker.of_type("assistant.cancel")
    assert len(cancels) == 1 and cancels[0]["turnId"] == 0
    h.agent_audio(bytes(640))  # ghost - dropped
    await settle()
    assert len(h.worker.of_type("audio.frame")) == 2
    h.agent_event({"type": "AgentStartedSpeaking"})
    h.agent_audio(bytes(640))  # fresh utterance - relayed
    await settle()
    assert len(h.worker.of_type("audio.frame")) == 3


async def test_ping_pong_and_context_notes_ride_update_prompt() -> None:
    h = Harness()
    await h.start()

    h.worker_says({"type": "ping", "ts": 12345})
    assert h.worker.of_type("pong")[0]["ts"] == 12345

    h.worker_says({"type": "participants", "count": 3})
    prompts = h.agent.of_type("update_prompt")
    assert "3 human participants" in prompts[-1]
    h.worker_says({"type": "dtmf", "digit": "7"})
    prompts = h.agent.of_type("update_prompt")
    assert '"7"' in prompts[-1]
    assert "3 human participants" in prompts[-1], "earlier notes stay in the rolling context section"


async def test_context_during_connect_lands_in_settings_prompt() -> None:
    h = Harness()
    gate: asyncio.Event = asyncio.Event()
    original_agent = h.agent

    async def slow_connect(cfg: Any, log: Any, handlers: DgSessionHandlers) -> FakeAgentPort:
        h.handlers = handlers
        await gate.wait()
        return original_agent

    h.session._connect_dg = slow_connect  # type: ignore[assignment]
    h.worker_says({"type": "session.start", "callId": "call-1", "threadId": "t", "caller": {}})
    await settle()
    h.worker_says({"type": "participants", "count": 3})
    h.worker_says({"type": "audio.frame", "seq": 1, "timestampMs": 0, "payloadBase64": "Zmlyc3Q="})
    h.worker_says({"type": "audio.frame", "seq": 2, "timestampMs": 20, "payloadBase64": "c2Vjb25k"})
    h.worker_says({"type": "session.start", "callId": "call-1", "threadId": "t", "caller": {}})  # duplicate - ignored
    gate.set()
    await settle()
    settings = h.agent.of_type("settings")
    assert len(settings) == 1, "duplicate session.start must not open a second agent session"
    assert "3 human participants" in settings[0]["agent"]["think"]["prompt"]
    # nothing flushed before the ack; order preserved after it
    assert h.agent.audio == []
    h.agent_event({"type": "SettingsApplied"})
    await settle()
    assert h.agent.audio == ["Zmlyc3Q=", "c2Vjb25k"], "buffered frames flush in order"


async def test_callid_mismatch_closes() -> None:
    h = Harness()
    h.worker_says({"type": "session.start", "callId": "OTHER", "threadId": "t", "caller": {}})
    await settle()
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "callid-mismatch"
    assert h.session.closed


async def test_function_calls_express_show_image_unknown_and_server_side(monkeypatch: Any) -> None:
    h = Harness()
    await h.start()

    h.function_call("t1", "express", '{"emotion": "happy"}')
    await settle()
    assert h.worker.of_type("expression")[0]["emotion"] == "happy"
    assert h.tool_results()["t1"] == ("express", "expressing happy")

    # arguments as an already-parsed object are tolerated
    h.function_call("t1b", "express", {"emotion": "surprised"})
    await settle()
    assert h.tool_results()["t1b"] == ("express", "expressing surprised")

    # oversize emotion is bounded
    h.function_call("t1c", "express", json.dumps({"emotion": "x" * 100}))
    await settle()
    assert "at most" in h.tool_results()["t1c"][1]

    # show_image url path (the only advertised contract) -> display.image
    async def fake_fetch(url: str, max_bytes: int, timeout_ms: int) -> tuple[bytes, str]:
        assert url == "https://example.com/pic.png"
        return b"img-bytes", "image/png"

    monkeypatch.setattr(session_mod, "fetch_public_image", fake_fetch)
    h.function_call("t2", "show_image", json.dumps({"url": "https://example.com/pic.png", "caption": "chart"}))
    await settle()
    img = h.worker.of_type("display.image")[0]
    assert img["mime"] == "image/png" and img["caption"] == "chart"
    assert img["dataBase64"] == base64.b64encode(b"img-bytes").decode()
    assert "shown" in h.tool_results()["t2"][1]

    # inline dataBase64 is NOT part of the schema the model sees - refused, so
    # the advertised contract and the handler cannot drift apart again
    h.function_call("t3", "show_image", json.dumps({"dataBase64": "aW1n", "mime": "image/png"}))
    await settle()
    assert "url" in h.tool_results()["t3"][1]

    # SSRF: a metadata URL is refused (validation raises before any fetch)
    monkeypatch.undo()
    h.function_call("t4", "show_image", json.dumps({"url": "http://169.254.169.254/latest/meta-data/"}))
    await settle()
    assert "failed" in h.tool_results()["t4"][1]

    # unknown tool -> error result, call keeps running
    h.function_call("t5", "teleport")
    await settle()
    assert "not implemented" in h.tool_results()["t5"][1]

    # server-side calls (client_side False) are Deepgram's own - never answered
    h.function_call("srv1", "server_lookup", "{}", client_side=False)
    await settle()
    assert "srv1" not in h.tool_results()

    # malformed frames must not kill the call
    h.agent_event({"type": "FunctionCallRequest"})
    h.agent_event({"type": "FunctionCallRequest", "functions": [{"client_side": True}]})
    h.agent_event({"type": "Error"})
    h.worker_says({"type": "ping", "ts": 99})
    assert h.worker.of_type("pong")[-1]["ts"] == 99
    assert not h.session.closed


async def test_custom_tools_execute_and_errors_are_reported() -> None:
    async def lookup(params: dict, ctx: Any) -> str:
        return f"order {params.get('orderNumber')} for call {ctx.call_id}: shipped"

    def broken(params: dict, ctx: Any) -> str:
        raise RuntimeError("backend down")

    tools = [
        CustomTool(
            name="lookup_order",
            description="Look up an order.",
            parameters={"type": "object", "properties": {"orderNumber": {"type": "string"}}, "required": []},
            handler=lookup,
        ),
        CustomTool(name="broken_tool", description="Always raises.", handler=broken),
    ]
    h = Harness(tools=tools)
    await h.start()
    names = [f["name"] for f in h.agent.of_type("settings")[0]["agent"]["think"]["functions"]]
    assert "lookup_order" in names and "end_call" in names

    h.function_call("ct1", "lookup_order", '{"orderNumber": "KO-1"}')
    await settle()
    assert h.tool_results()["ct1"] == ("lookup_order", "order KO-1 for call call-1: shipped")

    h.function_call("ct2", "broken_tool")
    await settle()
    assert "failed: backend down" in h.tool_results()["ct2"][1]


async def test_look_paths_no_video_no_endpoint_describer_and_recording_gate() -> None:
    # no vision endpoint configured
    h = Harness(vision=None)
    await h.start()
    h.function_call("v1", "look")
    await settle()
    assert "no video is available" in h.tool_results()["v1"][1]

    h.worker_says(
        {
            "type": "video.frame",
            "source": "screenshare",
            "ts": 1,
            "width": 640,
            "height": 360,
            "mime": "image/jpeg",
            "dataBase64": base64.b64encode(b"jpeg").decode(),
            "participantName": "Sara",
        }
    )
    h.function_call("v2", "look", '{"question": "what is on screen?"}')
    await settle()
    assert "no vision endpoint" in h.tool_results()["v2"][1]

    # with a describer
    async def describe(frame: dict, question: str) -> str:
        return f"I see a {frame['source']} frame. Q was: {question}"

    h2 = Harness(vision=describe)
    await h2.start()
    h2.worker_says(
        {
            "type": "video.frame",
            "source": "camera",
            "ts": 1,
            "width": 640,
            "height": 360,
            "mime": "image/jpeg",
            "dataBase64": base64.b64encode(b"cam").decode(),
        }
    )
    h2.function_call("w1", "look", '{"question": "who is there?"}')
    await settle()
    assert h2.tool_results()["w1"][1] == "I see a camera frame. Q was: who is there?"

    # recording gate opt-in
    h3 = Harness(cfg=make_config(vision_requires_recording=True), vision=describe)
    await h3.start()
    h3.worker_says(
        {
            "type": "video.frame",
            "source": "camera",
            "ts": 1,
            "width": 640,
            "height": 360,
            "mime": "image/jpeg",
            "dataBase64": base64.b64encode(b"cam").decode(),
        }
    )
    h3.function_call("g1", "look")
    await settle()
    assert "requires recording" in h3.tool_results()["g1"][1]
    h3.worker_says({"type": "recording.status", "status": "active"})
    h3.function_call("g2", "look")
    await settle()
    assert h3.tool_results()["g2"][1] == "described camera" or "I see" in h3.tool_results()["g2"][1]


async def test_end_call_tool_tears_down() -> None:
    h = Harness()
    await h.start()
    h.function_call("e1", "end_call")
    await settle()
    assert h.tool_results()["e1"] == ("end_call", "call ended")
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "agent-ended-call"
    assert h.agent.closed


async def test_goodbye_fallback_injects_exact_text_and_stays_audible() -> None:
    h = Harness()
    await h.start()
    h.worker_says({"type": "assistant.say", "text": "Goodbye, thanks for calling."})
    await settle()
    assert h.agent.of_type("inject") == ["Goodbye, thanks for calling."]
    assert h.worker.of_type("assistant.cancel"), "playback is flushed before the goodbye"
    # the injected goodbye is the agent speaking - it must relay (not muted)
    h.agent_event({"type": "AgentStartedSpeaking"})
    h.agent_audio(bytes(640))
    await settle()
    assert h.worker.of_type("audio.frame"), "the injected goodbye must stay audible"
    # duplicate goodbye is ignored (both governors can race)
    h.worker_says({"type": "assistant.say", "text": "Another goodbye."})
    await settle()
    assert h.agent.of_type("inject") == ["Goodbye, thanks for calling."]


async def test_goodbye_deterministic_tts_is_muted_and_undroppable(monkeypatch: Any) -> None:
    pcm = bytes([7]) * 640

    async def fake_tts(cfg: Any, text: str) -> bytes:
        return pcm

    monkeypatch.setattr(session_mod, "synthesize_goodbye", fake_tts)
    h = Harness(cfg=make_config(tts_model="aura-2-thalia-en"))
    await h.start()
    # stall the worker: hot-path audio would be dropped, the goodbye must not be
    h.worker.buffered = 10 * 1024 * 1024
    h.worker_says({"type": "assistant.say", "text": "Goodbye now."})
    await settle()
    frames = h.worker.of_type("audio.frame")
    assert frames and frames[0]["payloadBase64"] == base64.b64encode(pcm).decode("ascii"), (
        "exact synthesized PCM must reach the worker even under backpressure"
    )
    assert h.agent.of_type("inject") == [], "no agent fallback when TTS succeeds"
    # the agent is muted while the deterministic goodbye plays - even after
    # AgentStartedSpeaking clears the barge-in ghost filter
    n_before = len(h.worker.of_type("audio.frame"))
    h.worker.buffered = 0
    h.agent_event({"type": "AgentStartedSpeaking"})
    h.agent_audio(bytes(640))
    await settle()
    assert len(h.worker.of_type("audio.frame")) == n_before, "agent audio is muted during the deterministic goodbye"


async def test_governor_time_limit_ends_call() -> None:
    h = Harness(cfg=make_config(max_call_minutes=0.0005, goodbye_grace_ms=20))  # 30ms limit
    await h.start()
    # goodbye at ~30ms; session.end after grace(20ms) + the fixed 500ms buffer
    await asyncio.sleep(0.8)
    await settle()
    assert h.agent.of_type("inject"), "governor speaks the goodbye"
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "time-limit"
    assert h.session.closed and h.agent.closed


async def test_settings_applied_timeout_ends_call(monkeypatch: Any) -> None:
    monkeypatch.setattr(session_mod, "SETTINGS_APPLIED_TIMEOUT_S", 0.05)
    h = Harness()
    await h.start(apply_settings=False)
    await asyncio.sleep(0.15)
    await settle()
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "agent-unavailable"
    assert h.session.closed


async def test_agent_socket_close_ends_call() -> None:
    h = Harness()
    await h.start()
    assert h.handlers is not None
    h.handlers.on_close(1006, "gone")
    await settle()
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "agent-disconnected"


async def test_worker_close_during_connect_closes_orphaned_agent() -> None:
    h = Harness()
    gate: asyncio.Event = asyncio.Event()
    agent = h.agent

    async def slow_connect(cfg: Any, log: Any, handlers: DgSessionHandlers) -> FakeAgentPort:
        h.handlers = handlers
        await gate.wait()
        return agent

    h.session._connect_dg = slow_connect  # type: ignore[assignment]
    h.worker_says({"type": "session.start", "callId": "call-1", "threadId": "t", "caller": {}})
    await settle()
    h.session.handle_worker_close()
    gate.set()
    await settle()
    assert agent.closed, "the just-opened agent socket must be closed, not orphaned"
    assert agent.of_type("settings") == [], "must not send Settings on a torn-down call"


async def test_hot_path_backpressure_drops_only_disposable_audio() -> None:
    h = Harness()
    await h.start()
    h.worker.buffered = 10 * 1024 * 1024
    h.agent_audio(bytes(640))  # hot-path agent audio - dropped
    await settle()
    assert h.worker.of_type("audio.frame") == []
    h.worker_says({"type": "ping", "ts": 1})  # control frames always pass
    assert h.worker.of_type("pong")


async def test_session_start_failure_tears_the_call_down() -> None:
    """A crash while wiring the agent session must end the call immediately,
    not leave it half-alive (no agent, no governor) until a watchdog."""

    class BadAgent(FakeAgentPort):
        def send_settings(self, settings: dict) -> None:
            raise RuntimeError("settings exploded")

    worker = FakeWorkerPort()

    async def connect(cfg: Any, log: Any, handlers: Any) -> BadAgent:
        return BadAgent()

    session = CallSession(make_config(), worker, "call-1", connect_dg=connect, vision=None)
    session.handle_worker_message(json.dumps({"type": "session.start", "callId": "call-1", "caller": {}}))
    await settle()
    assert session.closed
    ends = worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "session-start-failed"


async def test_agent_error_and_injection_refused_are_counted() -> None:
    from deepgram_msteams_bridge.metrics import render_metrics, reset_metrics

    reset_metrics()
    h = Harness()
    await h.start()
    h.agent_event({"type": "Error", "code": "X", "description": "boom"})
    h.agent_event({"type": "InjectionRefused", "message": "mid-utterance"})
    text = render_metrics()
    assert "bridge_agent_errors_total 1" in text
    assert "bridge_injections_refused_total 1" in text
    assert not h.session.closed  # counted, not fatal


async def test_shutdown_defers_to_goodbye_in_progress() -> None:
    """SIGTERM drain must not cut off a goodbye the caller is still hearing;
    the goodbye's own hard-bounded backstop ends the call."""
    h = Harness()
    await h.start()
    h.worker_says({"type": "assistant.say", "text": "goodbye now"})
    await settle()
    assert h.session._goodbye_in_progress
    h.session.shutdown("bridge-shutdown")
    assert not h.session.closed  # left to finish
    h.session.end_call("test-cleanup")


async def test_drain_waits_for_goodbye_backstop(monkeypatch: Any) -> None:
    import time as _time

    import deepgram_msteams_bridge.server as server_mod

    monkeypatch.setattr(server_mod, "SHUTDOWN_GRACE_S", 0.05)
    monkeypatch.setattr(session_mod, "GOODBYE_HARD_CAP_MS", 150)
    h = Harness(cfg=make_config(goodbye_grace_ms=50))
    await h.start()
    server = server_mod.BridgeServer(h.cfg, None, None)
    server.sessions["call-1"] = h.session
    h.session._on_closed = lambda: server.sessions.pop("call-1", None)

    h.worker_says({"type": "assistant.say", "text": "goodbye now"})
    await settle()
    assert h.session._goodbye_in_progress and not h.session.closed

    t0 = _time.monotonic()
    await server.drain("SIGTERM")
    assert _time.monotonic() - t0 < 2
    assert h.session.closed and not server.sessions
    # the goodbye backstop ended the call, not a hard drain cut
    ends = h.worker.of_type("session.end")
    assert ends and ends[0]["reason"] == "goodbye-timeout"
