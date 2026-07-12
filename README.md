# Microsoft Teams Bridge for Deepgram Voice Agents (Python)

[![CI](https://github.com/komaa-com/deepgram-msteams-bridge-py/actions/workflows/ci.yml/badge.svg)](https://github.com/komaa-com/deepgram-msteams-bridge-py/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/deepgram-msteams-bridge.svg)](https://pypi.org/project/deepgram-msteams-bridge/)
[![Python versions](https://img.shields.io/pypi/pyversions/deepgram-msteams-bridge.svg)](https://pypi.org/project/deepgram-msteams-bridge/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/komaa-com/deepgram-msteams-bridge-py)

Bridge **Microsoft Teams voice/video calls** to a **Deepgram Voice Agent** (Nova STT + your chosen
LLM + Aura TTS, with turn-taking, all run by Deepgram).

> PyPI package: **`deepgram-msteams-bridge`** - the `-py` suffix is only in this repository's
> name, to distinguish it from the [Node.js sibling repo](https://github.com/komaa-com/deepgram-msteams-bridge).

This is the Python sibling of [`@komaa/deepgram-msteams-bridge`](https://www.npmjs.com/package/@komaa/deepgram-msteams-bridge)
(Node.js) - same wire contract, same environment variables, drop-in interchangeable behind the same
`.env` file. It terminates the StandIn media bridge wire protocol on one side and the Deepgram
Voice Agent WebSocket (`wss://agent.deepgram.com/v1/agent/converse`) on the other:

- **No transcoding**: the StandIn wire is base64 PCM 16 kHz mono and the Voice Agent session is
  pinned to `linear16` at 16 kHz both ways - the hot path is copy-only (base64/binary framing).
- **No dashboard**: the bridge configures each session itself from environment variables - STT
  model (`nova-3`), LLM (Deepgram-managed `open_ai`/`anthropic`, or any BYO provider via a think
  endpoint), Aura voice, prompt, greeting.
- **Protocol-correct**: `Settings` only after the server's `Welcome`; **no audio until the
  `SettingsApplied` ack** (buffered, bounded, then flushed oldest-first, with a 10 s ack timeout).
- **Barge-in**: `UserStartedSpeaking` maps to a Teams-side playback flush, with state-based
  ghost-audio filtering until the agent's next utterance.
- **Extensible tools**: the built-ins (`end_call`, `express`, `show_image`, `look`) plus your own
  client-side function tools, executed in-process inside your trust boundary.
- **Live context**: participant counts, DTMF and active-speaker changes ride `UpdatePrompt` as a
  bounded rolling notes section (the Voice Agent API has no non-interrupting context message).
- **Call governors**: a bridge-side hard time cap plus the worker-side governor; goodbyes are the
  exact text (standalone Aura TTS with the agent muted, or the live agent via `InjectAgentMessage`),
  never dropped under backpressure, and always backstopped.
- **On-demand vision**: the `look` tool answers from the caller's camera or screen-share via your
  own OpenAI-compatible vision endpoint (the raw frame is sent there - never to Deepgram; optional
  `VISION_REQUIRES_RECORDING` compliance gate).
- **Hardened**: HMAC-signed upgrades with replay guard, connection caps, SSRF-guarded image fetches
  (one re-validated redirect), dead-peer detection, `*.deepgram.com` host allowlists, graceful
  SIGTERM drain, Prometheus `/metrics` with a call-duration histogram.

[StandIn](https://standin.komaa.com) is the hosted media bridge that joins the Teams call and dials
this bridge - you run no Teams media stack yourself.

**Documentation**: [github.com/komaa-com/deepgram-msteams-bridge-py](https://github.com/komaa-com/deepgram-msteams-bridge-py)
and the Node sibling's [docs site](https://komaa-com.github.io/deepgram-msteams-bridge/) (same wire
contract and environment variables). Teams/StandIn setup lives at
[docs.komaa.com](https://docs.komaa.com/deepgram/installation).

## Install

```bash
pip install deepgram-msteams-bridge
```

Requires Python 3.10+. One runtime dependency (`aiohttp`).

## Run

```bash
DEEPGRAM_API_KEY=dg_... \
WORKER_SHARED_SECRET=... \
deepgram-msteams-bridge
```

A `.env` file in the working directory is loaded automatically (existing environment wins). The
bridge listens on `ws://0.0.0.0:8080/voice/msteams/stream` by default; StandIn appends `/{callId}`
per call. Expose the port with a tunnel and register the `wss://` URL as your identity's
**Agent voice URL** in the StandIn dashboard.

Optionally shape the agent (all have defaults):

```bash
DEEPGRAM_PROMPT="You are Komaa's friendly receptionist. Keep replies short."
DEEPGRAM_GREETING="Hello! You've reached Komaa. How can I help?"
DEEPGRAM_SPEAK_MODEL=aura-2-thalia-en
DEEPGRAM_THINK_MODEL=gpt-4o-mini
```

Third-party LLMs (`google`, `groq`, `aws_bedrock`, ...) additionally require
`DEEPGRAM_THINK_ENDPOINT_URL` (+ `DEEPGRAM_THINK_ENDPOINT_HEADERS` for auth) - Deepgram dials that
endpoint itself. See [`.env.example`](./.env.example) for every variable.

## Embed in your own asyncio app

```python
import asyncio
from deepgram_msteams_bridge import load_config, start_server

async def main():
    server = await start_server(load_config())
    try:
        await asyncio.Event().wait()   # run until cancelled
    finally:
        await server.close()           # drains live calls (session.end + close)

asyncio.run(main())
```

### Custom function tools

Register client-side tools the bridge executes; the handler's returned string goes back to the
agent as the function result, and a raise becomes an error result the model can recover from:

```python
from deepgram_msteams_bridge import CustomTool, load_config, start_server

async def lookup_order(params, ctx):
    # ctx: call_id, participant_count, recording_active, log
    return f"Order {params.get('orderNumber')} shipped yesterday and arrives tomorrow."

tools = [
    CustomTool(
        name="lookup_order",
        description="Look up the status of a customer order by its order number.",
        parameters={
            "type": "object",
            "properties": {"orderNumber": {"type": "string", "description": "e.g. KO-1234"}},
            "required": ["orderNumber"],
        },
        handler=lookup_order,
    )
]

server = await start_server(load_config(), tools=tools)
```

Names must not collide with the built-ins; collisions fail at startup. Keep handlers fast - the
caller is waiting on the answer.

### Custom vision hook

The `vision` argument answers the agent's `look` tool with your own model; pass `None` to disable
vision entirely, or omit it to use the env-configured describer (`VISION_API_URL`):

```python
async def describe(frame: dict, question: str) -> str:
    # frame: {"source": "camera"|"screenshare", "mime": ..., "dataBase64": ..., ...}
    return "a short description the agent relays aloud"

server = await start_server(load_config(), vision=describe)
```

**Know the data flow:** the raw frame is sent to the vision endpoint you configure (never to
Deepgram). Set `VISION_REQUIRES_RECORDING=true` to refuse frames unless Teams recording is active,
and run a local model (Ollama/vLLM) if frames must not leave your infrastructure.

### Custom agent transport (testing)

The `connect_dg` argument substitutes the Deepgram connection; the repository's own
[test suite](./tests) runs entirely against a fake, no Deepgram account needed.

## Governors and privacy

- **Two governors** (StandIn tier cutoffs and the bridge-side `MAX_CALL_MINUTES`) both speak a
  goodbye before hanging up, backstopped so a call can never sit open half-dead.
- Transcripts (`ConversationText`) are never logged unless `LOG_TRANSCRIPTS=true` **and** Teams
  recording is active. Video frames are buffered in memory only.
- Regional routing: set `DEEPGRAM_AGENT_HOST` / `DEEPGRAM_API_HOST` to `api.eu.deepgram.com` /
  `api.au.deepgram.com` to keep traffic in-region.

## Siblings

The same bridge exists for [ElevenLabs](https://github.com/komaa-com/elevenlabs-msteams-bridge-py),
[LiveKit](https://github.com/komaa-com/livekit-msteams-bridge-py) (Python), and as Node packages
for [ElevenLabs](https://github.com/komaa-com/elevenlabs-msteams-bridge),
[LiveKit](https://github.com/komaa-com/livekit-msteams-bridge),
[OpenAI Realtime](https://github.com/komaa-com/openai-msteams-bridge), and
[Deepgram](https://github.com/komaa-com/deepgram-msteams-bridge) - same wire protocol, same
hardening, pick the agent platform and runtime that fit.

## License

[MIT](./LICENSE)
