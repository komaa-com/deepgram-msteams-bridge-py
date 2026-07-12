"""Config loading: fail-loud validation of numerics, hosts, vision URL, and the
BYO-LLM think endpoint."""

from __future__ import annotations

import pytest

from deepgram_msteams_bridge.config import load_config


@pytest.fixture(autouse=True)
def base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DEEPGRAM_AGENT_HOST",
        "DEEPGRAM_API_HOST",
        "DEEPGRAM_HOST_ALLOW_ANY",
        "DEEPGRAM_THINK_ENDPOINT_URL",
        "DEEPGRAM_THINK_ENDPOINT_HEADERS",
        "VISION_API_URL",
        "VISION_REQUIRES_RECORDING",
        "MAX_CALL_MINUTES",
        "PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WORKER_SHARED_SECRET", "s")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg_x")


def test_defaults() -> None:
    cfg = load_config()
    assert cfg.agent_host == "agent.deepgram.com"
    assert cfg.api_host == "api.deepgram.com"
    assert cfg.listen_model == "nova-3"
    assert cfg.think_provider == "open_ai"
    assert cfg.speak_model == "aura-2-thalia-en"
    assert cfg.language == "en"
    assert cfg.tts_model is None
    assert cfg.vision_requires_recording is False


def test_missing_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        load_config()


def test_numeric_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CALL_MINUTES", "abc")
    with pytest.raises(ValueError, match="not a number"):
        load_config()
    monkeypatch.setenv("MAX_CALL_MINUTES", "-1")
    with pytest.raises(ValueError, match="must not be negative"):
        load_config()


def test_host_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_AGENT_HOST", "api.eu.deepgram.com")
    assert load_config().agent_host == "api.eu.deepgram.com", "regional pin allowed"

    monkeypatch.setenv("DEEPGRAM_AGENT_HOST", "evil.example.com")
    with pytest.raises(ValueError, match="not a deepgram.com host"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_AGENT_HOST", "agent.deepgram.com")
    monkeypatch.setenv("DEEPGRAM_API_HOST", "evil.example.com")
    with pytest.raises(ValueError, match="not a deepgram.com host"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_HOST_ALLOW_ANY", "true")
    assert load_config().api_host == "evil.example.com", "explicit override honored"


def test_think_endpoint_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_URL", "not a url")
    with pytest.raises(ValueError, match="https URL"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_URL", "http://llm.example.com/v1")
    with pytest.raises(ValueError, match="https URL"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_URL", "https://user:pass@llm.example.com/v1")
    with pytest.raises(ValueError, match="embedded credentials"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_HEADERS", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_HEADERS", '{"authorization": 42}')
    with pytest.raises(ValueError, match="must be a string"):
        load_config()

    monkeypatch.setenv("DEEPGRAM_THINK_ENDPOINT_HEADERS", '{"authorization": "Bearer x"}')
    cfg = load_config()
    assert cfg.think_endpoint_url == "https://llm.example.com/v1"
    assert cfg.think_endpoint_headers == {"authorization": "Bearer x"}


def test_vision_url_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_API_URL", "not a url")
    with pytest.raises(ValueError, match="VISION_API_URL"):
        load_config()

    monkeypatch.setenv("VISION_API_URL", "ftp://vision.example.com/api")
    with pytest.raises(ValueError, match="VISION_API_URL"):
        load_config()

    monkeypatch.setenv("VISION_API_URL", "https://user:pass@vision.example.com/api")
    with pytest.raises(ValueError, match="embedded credentials"):
        load_config()

    # local Ollama is allowed (warned about, not rejected)
    monkeypatch.setenv("VISION_API_URL", "http://127.0.0.1:11434/v1/chat/completions")
    assert load_config().vision_api_url == "http://127.0.0.1:11434/v1/chat/completions"


def test_vision_requires_recording_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_REQUIRES_RECORDING", "true")
    assert load_config().vision_requires_recording is True
