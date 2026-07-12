from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from deepgram_msteams_bridge.config import BridgeConfig


def make_config(**overrides: Any) -> BridgeConfig:
    base: dict[str, Any] = dict(
        port=8080,
        host="127.0.0.1",
        worker_shared_secret="test-secret",
        deepgram_api_key="dg-test",
        agent_host="agent.deepgram.com",
        api_host="api.deepgram.com",
        listen_model="nova-3",
        think_provider="open_ai",
        think_model="gpt-4o-mini",
        think_endpoint_url=None,
        think_endpoint_headers=None,
        speak_model="aura-2-thalia-en",
        language="en",
        instructions=None,
        greeting=None,
        tts_model=None,
        vision_api_url=None,
        vision_api_key=None,
        vision_model=None,
        vision_requires_recording=False,
        max_call_minutes=0,
        goodbye_text="goodbye",
        goodbye_grace_ms=100,
        hmac_freshness_ms=60_000,
        max_connections=0,
        max_connections_per_ip=0,
        pre_start_timeout_ms=0,
        worker_idle_timeout_ms=0,
        trust_proxy=False,
        tls_cert_path=None,
        tls_key_path=None,
        log_transcripts=False,
    )
    base.update(overrides)
    return BridgeConfig(**base)


class FakeWorkerPort:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None
        self.buffered = 0

    @property
    def is_open(self) -> bool:
        return self.closed is None

    @property
    def buffered_bytes(self) -> int:
        return self.buffered

    def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)

    def of_type(self, mtype: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == mtype]


class FakeAgentPort:
    """Fake Deepgram Voice Agent: records what the bridge sends; tests push
    JSON events via the session handlers and binary audio via emit_audio."""

    def __init__(self) -> None:
        self.audio: list[str] = []  # base64 of the binary frames the bridge sent
        self.messages: list[tuple[str, Any]] = []
        self.closed = False

    @property
    def is_open(self) -> bool:
        return not self.closed

    def send_audio_chunk(self, b64: str) -> None:
        self.audio.append(b64)

    def send_settings(self, settings: dict) -> None:
        self.messages.append(("settings", settings))

    def update_prompt(self, prompt: str) -> None:
        self.messages.append(("update_prompt", prompt))

    def inject_agent_message(self, text: str) -> None:
        self.messages.append(("inject", text))

    def send_function_call_response(self, call_id: str, name: str, content: str) -> None:
        self.messages.append(("tool_result", (call_id, name, content)))

    def close(self) -> None:
        self.closed = True

    def of_type(self, kind: str) -> list[Any]:
        return [payload for k, payload in self.messages if k == kind]


@pytest.fixture
def fake_worker() -> FakeWorkerPort:
    return FakeWorkerPort()


@pytest.fixture
def fake_agent() -> FakeAgentPort:
    return FakeAgentPort()


async def settle() -> None:
    """Let pending callbacks/tasks run."""
    for _ in range(5):
        await asyncio.sleep(0)
