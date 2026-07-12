"""Minimal embedding of deepgram-msteams-bridge with a custom vision hook and a
custom function tool.

Run: `python main.py` (reads .env from this directory; see the repo root's
.env.example). The custom `describe` function answers the agent's `look` tool -
the raw frame goes to YOUR model, never to Deepgram. The `lookup_order` tool
shows how to wire the agent to your own systems.
"""

from __future__ import annotations

import asyncio
import signal

from deepgram_msteams_bridge import CustomTool, load_config, load_dotenv, start_server


async def describe(frame: dict, question: str) -> str:
    # frame["dataBase64"] is the JPEG/PNG frame, frame["mime"] its type,
    # frame["source"] is "camera" or "screenshare".
    # Call your vision model here (OpenAI, Azure OpenAI, a local VLM...).
    return f"(stub) I received a {frame.get('mime')} frame from the {frame.get('source')} and the question: {question}"


async def lookup_order(params: dict, ctx) -> str:
    # Call your own backend here; the returned string goes to the agent.
    ctx.log.info(f"lookup_order {params.get('orderNumber')}")
    return f"Order {params.get('orderNumber')} shipped yesterday and arrives tomorrow."


TOOLS = [
    CustomTool(
        name="lookup_order",
        description="Look up the status of a customer order by its order number.",
        parameters={
            "type": "object",
            "properties": {"orderNumber": {"type": "string", "description": "The order number, e.g. KO-1234."}},
            "required": ["orderNumber"],
        },
        handler=lookup_order,
    )
]


async def main() -> None:
    cfg = load_config()
    server = await start_server(cfg, vision=describe, tools=TOOLS)
    print(f"Point your StandIn identity's agent WebSocket URL at ws://<this-host>:{cfg.port}/voice/msteams/stream")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await server.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
