"""Bridge configuration, entirely from environment variables.

The worker-side contract (HMAC secret, wire protocol) must match the StandIn
media bridge; the Deepgram side needs an API key - the agent itself
(listen/think/speak, prompt, greeting) is configured per session from the
variables below, no dashboard required. Environment variable names are
identical to the Node package (@komaa/deepgram-msteams-bridge), so the two are
drop-in interchangeable behind the same .env file.
"""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from .log import logger

log = logger("config")

DEFAULT_GOODBYE = "I'm sorry, we've reached the time limit for this call. Thank you for calling, goodbye!"


@dataclass(frozen=True)
class BridgeConfig:
    port: int
    """TCP port the bridge listens on for worker WebSocket upgrades."""
    host: str
    """Bind address."""
    worker_shared_secret: str
    """Must equal the shared secret the StandIn media bridge signs with (HMAC upgrade check)."""
    deepgram_api_key: str
    """Server-side Deepgram key; opens Voice Agent sessions and calls Aura TTS."""
    agent_host: str
    """Voice Agent WebSocket host. Restricted to *.deepgram.com."""
    api_host: str
    """REST API host (goodbye TTS). Restricted to *.deepgram.com."""
    listen_model: str
    """STT model for agent.listen (e.g. nova-3, flux-general-en)."""
    think_provider: str
    """LLM provider for agent.think (e.g. open_ai, anthropic, google)."""
    think_model: str
    """LLM model for agent.think (e.g. gpt-4o-mini)."""
    think_endpoint_url: str | None
    """BYO-LLM endpoint - REQUIRED by Deepgram for third-party think providers
    (google, groq, aws_bedrock, ...); Deepgram-managed open_ai/anthropic work
    without it. Deepgram dials this URL itself."""
    think_endpoint_headers: dict[str, str] | None
    """Headers for the think endpoint (e.g. {"authorization": "Bearer ..."})."""
    speak_model: str
    """Aura TTS voice model for agent.speak (e.g. aura-2-thalia-en)."""
    language: str
    """Agent language (set on the listen and speak providers)."""
    instructions: str | None
    """Base agent prompt. None = a built-in default; per-call caller context is appended either way."""
    greeting: str | None
    """Deterministic opening line (Settings agent.greeting). None = the agent opens naturally."""
    tts_model: str | None
    """Aura model for the deterministic governor goodbye via standalone TTS.
    None = the goodbye is spoken verbatim by the live agent via InjectAgentMessage."""
    vision_api_url: str | None
    """Vision path 2: OpenAI-compatible chat-completions URL for describe-then-answer.
    None = the look tool reports vision unavailable (the Voice Agent API is audio-only)."""
    vision_api_key: str | None
    """Bearer key for the vision endpoint (optional - local endpoints may not need one)."""
    vision_model: str | None
    """Vision model name (required when vision_api_url is set)."""
    vision_requires_recording: bool
    """Gate the look tool on Teams recording being active. Camera/screen frames are
    PII-bearing; when True, the bridge refuses to send a frame to the vision endpoint
    unless recording.status is "active"."""
    max_call_minutes: float
    """Bridge-side call governor: hard cap on call duration in minutes (fractional
    allowed). 0 = disabled. Deepgram doesn't know about your billing; on limit the
    bridge speaks a goodbye and ends the call."""
    goodbye_text: str
    """Goodbye line the governor speaks (deterministic via TTS when DEEPGRAM_TTS_MODEL is set)."""
    goodbye_grace_ms: float
    """How long to let the goodbye play out before session.end when its duration is unknown."""
    hmac_freshness_ms: float
    """Allowed clock skew for the HMAC timestamp, in ms (the worker documents +/-60s)."""
    max_connections: int
    """Max concurrent worker connections (0 = default 64)."""
    max_connections_per_ip: int
    """Max concurrent connections from one remote IP (0 = default: same as max_connections)."""
    pre_start_timeout_ms: float
    """Drop a worker that authenticates but never sends session.start after this many ms (0 = default 10s)."""
    worker_idle_timeout_ms: float
    """Dead-peer window: end the call after this many ms without ANY worker message
    (0 = default 90s; the worker heartbeats every 30s)."""
    trust_proxy: bool
    """Trust X-Forwarded-For for the per-IP cap (only behind a proxy you control)."""
    tls_cert_path: str | None
    """PEM cert path for native TLS (wss). When cert + key are both set the bridge serves
    TLS itself; otherwise it is plain WS and MUST be fronted by a TLS terminator."""
    tls_key_path: str | None
    """PEM key path for native TLS (wss)."""
    log_transcripts: bool
    """Log transcripts (ConversationText) - still gated on Teams recording.status == "active"."""


def _validate_deepgram_host(name: str, host: str) -> str:
    """DEEPGRAM_API_KEY is sent as `Authorization: Token ...` to these hosts, so
    an attacker-influenced or fat-fingered host would exfiltrate the key.
    Restrict both to Deepgram's own domain. Set DEEPGRAM_HOST_ALLOW_ANY=true
    only for a deliberate proxy/test host."""
    if os.environ.get("DEEPGRAM_HOST_ALLOW_ANY") == "true":
        return host
    h = host.lower()
    if h == "deepgram.com" or h.endswith(".deepgram.com"):
        return host
    raise ValueError(
        f'{name} "{host}" is not a deepgram.com host; the API key must not be sent elsewhere. '
        "Set DEEPGRAM_HOST_ALLOW_ANY=true to override for a trusted proxy."
    )


def _validate_vision_url(raw: str | None) -> str | None:
    """Vision path 2 sends caller video frames to VISION_API_URL, so validate it
    at startup: a well-formed http(s) URL with no embedded credentials. A literal
    private/loopback IP is allowed (local Ollama/vLLM is a documented use case)
    but WARNED about, so a fat-fingered internal host is visible."""
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f'VISION_API_URL "{raw}" must be a well-formed http(s) URL')
    if parts.username or parts.password:
        raise ValueError("VISION_API_URL must not contain embedded credentials")
    try:
        ip = ipaddress.ip_address(parts.hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            log.warn(
                f"VISION_API_URL points at private/reserved address {parts.hostname} - fine for a local vision "
                "endpoint (Ollama/vLLM), but make sure this is intentional: caller video frames will be sent there"
            )
    except ValueError:
        pass  # hostname, not a literal IP
    return raw


def _validate_think_endpoint_url(raw: str | None) -> str | None:
    """The think endpoint is dialed BY DEEPGRAM (not this bridge), so the
    posture is simply: https only, well-formed, no embedded credentials
    (credentials belong in the headers object)."""
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f'DEEPGRAM_THINK_ENDPOINT_URL "{raw}" must be a well-formed https URL (Deepgram dials it)')
    if parts.username or parts.password:
        raise ValueError(
            "DEEPGRAM_THINK_ENDPOINT_URL must not contain embedded credentials; use DEEPGRAM_THINK_ENDPOINT_HEADERS"
        )
    return raw


def _parse_think_endpoint_headers(raw: str | None) -> dict[str, str] | None:
    """Parse DEEPGRAM_THINK_ENDPOINT_HEADERS: a JSON object of string values. Fail loud on junk."""
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ValueError('DEEPGRAM_THINK_ENDPOINT_HEADERS is not valid JSON (expected {"header": "value"})') from None
    if not isinstance(parsed, dict):
        raise ValueError("DEEPGRAM_THINK_ENDPOINT_HEADERS must be a JSON object")
    headers: dict[str, str] = {}
    for k, v in parsed.items():
        if not isinstance(v, str):
            raise ValueError(f'DEEPGRAM_THINK_ENDPOINT_HEADERS["{k}"] must be a string')
        headers[str(k)] = v
    return headers


def _required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise ValueError(f"Missing required env var {name}")
    return v


def _optional(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def _num_from_env(name: str, fallback: float) -> float:
    """Parse a numeric env var, failing LOUD on a non-numeric value. float("abc")
    raising is the point: a typo must stop startup with a clear message, not
    silently disable the governor (MAX_CALL_MINUTES) or misbind (PORT).
    Negatives fail too: all these knobs are counts/durations where a negative is
    never meaningful and would silently disable checks guarded by `> 0`."""
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return fallback
    try:
        n = float(raw)
    except ValueError:
        raise ValueError(f'Env var {name}="{raw}" is not a number') from None
    if n != n or n in (float("inf"), float("-inf")):
        raise ValueError(f'Env var {name}="{raw}" is not a number')
    if n < 0:
        raise ValueError(f'Env var {name}="{raw}" must not be negative')
    return n


def load_config() -> BridgeConfig:
    return BridgeConfig(
        port=int(_num_from_env("PORT", 8080)),
        host=os.environ.get("BIND", "").strip() or "0.0.0.0",
        worker_shared_secret=_required("WORKER_SHARED_SECRET"),
        deepgram_api_key=_required("DEEPGRAM_API_KEY"),
        agent_host=_validate_deepgram_host(
            "DEEPGRAM_AGENT_HOST", os.environ.get("DEEPGRAM_AGENT_HOST", "").strip() or "agent.deepgram.com"
        ),
        api_host=_validate_deepgram_host(
            "DEEPGRAM_API_HOST", os.environ.get("DEEPGRAM_API_HOST", "").strip() or "api.deepgram.com"
        ),
        listen_model=os.environ.get("DEEPGRAM_LISTEN_MODEL", "").strip() or "nova-3",
        think_provider=os.environ.get("DEEPGRAM_THINK_PROVIDER", "").strip() or "open_ai",
        think_model=os.environ.get("DEEPGRAM_THINK_MODEL", "").strip() or "gpt-4o-mini",
        think_endpoint_url=_validate_think_endpoint_url(_optional("DEEPGRAM_THINK_ENDPOINT_URL")),
        think_endpoint_headers=_parse_think_endpoint_headers(_optional("DEEPGRAM_THINK_ENDPOINT_HEADERS")),
        speak_model=os.environ.get("DEEPGRAM_SPEAK_MODEL", "").strip() or "aura-2-thalia-en",
        language=os.environ.get("DEEPGRAM_LANGUAGE", "").strip() or "en",
        instructions=_optional("DEEPGRAM_PROMPT"),
        greeting=_optional("DEEPGRAM_GREETING"),
        tts_model=_optional("DEEPGRAM_TTS_MODEL"),
        vision_api_url=_validate_vision_url(_optional("VISION_API_URL")),
        vision_api_key=_optional("VISION_API_KEY"),
        vision_model=_optional("VISION_MODEL"),
        vision_requires_recording=os.environ.get("VISION_REQUIRES_RECORDING") == "true",
        max_call_minutes=_num_from_env("MAX_CALL_MINUTES", 0),
        goodbye_text=os.environ.get("GOODBYE_TEXT") or DEFAULT_GOODBYE,
        goodbye_grace_ms=_num_from_env("GOODBYE_GRACE_MS", 8000),
        hmac_freshness_ms=_num_from_env("HMAC_FRESHNESS_MS", 60_000),
        max_connections=int(_num_from_env("MAX_CONNECTIONS", 0)),
        max_connections_per_ip=int(_num_from_env("MAX_CONNECTIONS_PER_IP", 0)),
        pre_start_timeout_ms=_num_from_env("PRE_START_TIMEOUT_MS", 0),
        worker_idle_timeout_ms=_num_from_env("WORKER_IDLE_TIMEOUT_MS", 0),
        trust_proxy=os.environ.get("TRUST_PROXY_XFF") == "true",
        tls_cert_path=_optional("TLS_CERT_PATH"),
        tls_key_path=_optional("TLS_KEY_PATH"),
        log_transcripts=os.environ.get("LOG_TRANSCRIPTS") == "true",
    )
