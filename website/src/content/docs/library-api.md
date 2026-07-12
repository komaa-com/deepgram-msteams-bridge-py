---
title: "Library API"
description: "Embed the bridge in your own asyncio app: start_server options, custom tools, vision hooks, custom agent transports, HMAC helpers, protocol helpers."
---

The package is both a CLI and an importable Python library. Everything below is exported from the package root.

```python
from deepgram_msteams_bridge import load_config, start_server
```

## Run the bridge in your own service

`load_config()` reads the same environment variables as the CLI and raises a clear `ValueError` when a required variable is missing or a numeric one is not a number. `start_server(cfg)` is a coroutine that starts listening and returns a `BridgeServer` handle.

```python
import asyncio
from deepgram_msteams_bridge import load_config, start_server

async def main():
    server = await start_server(load_config())
    print("bridge up")
    try:
        await asyncio.Event().wait()   # run until cancelled
    finally:
        await server.close()           # drains live calls (session.end + close)

asyncio.run(main())
```

`server.drain()` ends every live call gracefully without stopping the listener; `server.close()` drains and stops. A session mid-goodbye is left to finish (its hard-bounded backstop tears it down), so a rolling deploy never cuts off the last thing the caller hears. The CLI wires SIGTERM/SIGINT to this for you - in your own app, hook your shutdown path to `server.close()`.

## Custom function tools

The `tools` argument to `start_server` registers client-side function tools the **bridge executes**. Each `CustomTool` is a name, description, JSON schema, and handler (sync or async); the handler's returned string goes back to the agent as the `FunctionCallResponse` content, and a raise becomes an error result the model can recover from.

```python
from deepgram_msteams_bridge import CustomTool, load_config, start_server

async def lookup_order(params: dict, ctx) -> str:
    # ctx: CustomToolContext(call_id, participant_count, recording_active, log)
    ctx.log.info(f"lookup_order {params.get('orderNumber')}")
    return await my_backend.order_status(str(params.get("orderNumber")))  # the agent speaks this

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
    ),
]

server = await start_server(load_config(), tools=tools)
```

Names must not collide with the built-ins (`end_call`, `express`, `show_image`, `look`) - collisions and duplicates fail at startup. Keep handlers fast (the caller is waiting on the answer) and enforce your own timeout for slow backends. See [Extending the Agent's Tools](/deepgram-msteams-bridge-py/extending-tools/) for the trust model.

## Custom vision hook

The `vision` argument to `start_server` is a `VisionDescriber` - your own answer to the agent's `look` tool. This example uses OpenAI's vision API (`pip install openai`):

```python
from openai import AsyncOpenAI
from deepgram_msteams_bridge import load_config, start_server

openai = AsyncOpenAI()  # reads OPENAI_API_KEY

async def describe(frame: dict, question: str) -> str:
    # frame: {"source": "camera" | "screenshare", "mime": ..., "dataBase64": ...,
    #         "width": ..., "height": ..., "participantName": ...}
    who = "the caller's shared screen" if frame["source"] == "screenshare" else "the caller's camera"
    res = await openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This is {who}. {question}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{frame['mime']};base64,{frame['dataBase64']}", "detail": "low"},
                    },
                ],
            }
        ],
    )
    return res.choices[0].message.content or "I could not make out the image."

server = await start_server(load_config(), vision=describe)
```

The frame is passed to your describer (and, in this example, on to OpenAI) - it never reaches Deepgram. Pass `vision=None` to disable vision entirely; omit the argument to use the built-in `make_vision_describer(cfg)`, driven by `VISION_API_URL`.

## Custom agent transport (testing)

The `connect_dg` argument to `start_server` is an async factory that returns an `AgentPort`. The default opens a real Voice Agent socket; tests substitute a fake so no network or API key is needed.

```python
from deepgram_msteams_bridge import load_config, start_server

async def fake_connector(cfg, log, handlers):
    class FakePort:
        is_open = True
        def send_audio_chunk(self, b64): ...
        def send_settings(self, settings): ...
        def update_prompt(self, prompt): ...
        def inject_agent_message(self, text): ...
        def send_function_call_response(self, call_id, name, content): ...
        def close(self): ...
    # push server->bridge events with handlers.on_message({...}) and
    # agent audio with handlers.on_audio(b"...")
    return FakePort()

server = await start_server(load_config(), connect_dg=fake_connector, vision=None)
```

The repository's own [test suite](https://github.com/komaa-com/deepgram-msteams-bridge-py/tree/main/tests) uses exactly this shape - `tests/conftest.py` has a reusable `FakeAgentPort`.

## HMAC helpers

Useful if you build tools that talk to the bridge, or want to test the upgrade.

```python
import time
from deepgram_msteams_bridge import sign, verify, is_fresh, TIMESTAMP_HEADER, SIGNATURE_HEADER

ts = int(time.time() * 1000)
signature = sign(secret, ts, call_id)   # HMAC-SHA256(secret, f"{ts}.{call_id}") hex
# send as headers X-StandIn-Timestamp / -Signature
verify(secret, ts, call_id, signature)  # constant-time, False on any missing input
is_fresh(ts, 60_000)                    # within the freshness window?
```

## Deepgram-side helpers

Exported for tooling and tests: `DeepgramAgentSocket`, `build_settings`, `build_prompt`, `synthesize_goodbye`, `custom_tool_schema`, `BRIDGE_FUNCTIONS`, and the `WIRE_SAMPLE_RATE` constant (16000).

## Protocol helpers

Wire messages are plain dicts (they arrive and leave as JSON). `parse_worker_message(raw)` is the guarded parser (returns `None` on junk), and `pcm16k_bytes_to_ms(n)` converts PCM byte counts to milliseconds. See the [Wire Protocol](/deepgram-msteams-bridge-py/wire-protocol/) for the full contract.

## Also exported

- `authorize_upgrade`, `call_id_from_path`, `ReplayGuard` - the upgrade-authorization primitives.
- `CallSession`, `WorkerPort`, `AgentPort`, `DgConnector`, `DgSessionHandlers` - the per-call relay class and its transport protocols (advanced embedding).
- `CustomTool`, `CustomToolContext`, `VisionDescriber`, `BridgeConfig`, `BridgeServer` - the public types.
- `assert_public_http_url`, `is_forbidden_ip`, `fetch_public_image` - the SSRF-guard primitives.
- `load_dotenv` - the tiny `.env` loader the CLI uses.
- `render_metrics`, `reset_metrics`, `logger`, `Logger` - metrics text and the minimal leveled logger.
