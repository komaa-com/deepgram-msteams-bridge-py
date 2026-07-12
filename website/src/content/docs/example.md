---
title: "Run the Example"
description: "A guided walkthrough of examples/basic-bridge: what each line does, how the vision hook and the custom lookup_order tool work, and how to grow it into your own service."
---

The repository ships one example, [`examples/basic-bridge`](https://github.com/komaa-com/deepgram-msteams-bridge-py/tree/main/examples/basic-bridge) - a complete, working embedding in about 50 lines. This page walks through it so you understand every moving part before writing your own.

## What the example is

A single `main.py` that:

1. loads a `.env` file,
2. starts the bridge with `start_server()`,
3. plugs a **custom vision hook** into the agent's `look` tool,
4. registers a **custom `lookup_order` function tool** the agent can call,
5. shuts down gracefully on Ctrl-C / SIGTERM.

## Run it

```bash
pip install deepgram-msteams-bridge
git clone https://github.com/komaa-com/deepgram-msteams-bridge-py
cd deepgram-msteams-bridge-py/examples/basic-bridge
cp ../../.env.example .env    # fill in the two required values
python main.py
```

Expose port 8080 with a tunnel (see [Getting Started](/deepgram-msteams-bridge-py/getting-started/)), set your StandIn identity's **agent WebSocket URL** to the `wss://…/voice/msteams/stream` form, and place a Teams call - your Deepgram agent answers.

The two required values in `.env`:

| Variable | What to put there |
|---|---|
| `DEEPGRAM_API_KEY` | Your Deepgram API key with Voice Agent access (server-side only; never sent to the Teams side). |
| `WORKER_SHARED_SECRET` | The shared secret from StandIn pairing - both sides must match exactly. |

Everything else has a default (`nova-3` STT, `open_ai/gpt-4o-mini` thinking, `aura-2-thalia-en` voice); set `DEEPGRAM_PROMPT` and `DEEPGRAM_GREETING` to give the agent a personality.

## The code, line by line

```python
from deepgram_msteams_bridge import CustomTool, load_config, load_dotenv, start_server

async def describe(frame: dict, question: str) -> str:
    # frame["dataBase64"] is the JPEG/PNG frame, frame["mime"] its type,
    # frame["source"] is "camera" or "screenshare".
    return f"(stub) I received a {frame.get('mime')} frame from the {frame.get('source')}"

async def lookup_order(params: dict, ctx) -> str:
    ctx.log.info(f"lookup_order {params.get('orderNumber')}")
    return f"Order {params.get('orderNumber')} shipped yesterday and arrives tomorrow."

TOOLS = [
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

async def main() -> None:
    cfg = load_config()
    server = await start_server(cfg, vision=describe, tools=TOOLS)
    ...
    await stop.wait()      # run until SIGTERM / Ctrl-C
    await server.close()   # drain live calls gracefully
```

- **`load_dotenv()`** reads `.env` from the working directory (existing environment always wins), so the example runs the same way the CLI does.
- **`load_config()`** reads every setting from environment variables and fails loud on a missing required variable or a non-numeric number - a typo stops startup with a clear message instead of silently misbehaving.
- **`start_server(cfg, vision=describe, tools=TOOLS)`** starts the WebSocket server and returns a handle. The two keyword arguments are the interesting parts, below.
- **`await server.close()`** ends every live call with a spoken-protocol `session.end` (not a hard socket drop) before the process exits - and lets an in-progress goodbye finish first.

## The vision hook

When your agent calls its `look` function ("what do you see?"), the bridge hands **your** function the latest camera or screen-share frame and the agent's question. Whatever text you return is what the agent gets as the function result - the frame goes to your model, **never to Deepgram** (the Voice Agent API is audio-only).

The stub just proves the wiring. Replace it with any vision-capable model:

```python
async def describe(frame: dict, question: str) -> str:
    # e.g. call OpenAI, Azure OpenAI, Claude, or a local VLM here with
    # the data URL f"data:{frame['mime']};base64,{frame['dataBase64']}"
    return await my_vision_model(frame, question)
```

Prefer configuration over code? Leave `vision=` out and set `VISION_API_URL` / `VISION_MODEL` instead - the built-in describer calls any OpenAI-compatible chat-completions endpoint. Pass `vision=None` to disable vision entirely. The data-flow implications are covered in [Vision and Tools](/deepgram-msteams-bridge-py/vision-and-tools/).

## The custom tool

`lookup_order` is a complete custom function tool: a name, a description, a JSON schema, and a handler that runs **in your process** when the agent calls it. The returned string is what the agent speaks from. Swap the stub body for a call to your own backend - and see [Extending the Agent's Tools](/deepgram-msteams-bridge-py/extending-tools/) for the rules (name collisions, error handling, keeping handlers fast).

## From example to your own service

The example **is** the recommended embedding shape. To grow it:

- add your own logic around `start_server()` (it is just an awaitable in your event loop);
- swap the vision stub for a real model and the `lookup_order` stub for your backend;
- set the [governor variables](/deepgram-msteams-bridge-py/governors-and-privacy/) (`MAX_CALL_MINUTES`, `DEEPGRAM_TTS_MODEL`, `GOODBYE_TEXT`) before going to production;
- for tests, inject a fake agent with the `connect_dg` argument - see [Library API](/deepgram-msteams-bridge-py/library-api/).

If you only need the stock behavior, skip the embedding entirely and run the `deepgram-msteams-bridge` CLI.
