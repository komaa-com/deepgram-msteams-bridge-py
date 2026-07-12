"""Vision path 2 ("describe-then-inject"): run the buffered frame through YOUR
vision model and return a short text description. Model-agnostic: any
OpenAI-compatible chat-completions endpoint with image_url input (OpenAI,
Azure OpenAI, Ollama, vLLM, ...). Frames are sent transiently for inference,
not persisted - which is why this path is allowed even before the Teams
recording gate requirement is enforced by the session when configured."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import aiohttp

from .config import BridgeConfig

# frame dict (a video.frame message) + question -> description
VisionDescriber = Callable[[dict[str, Any], str], Awaitable[str]]

# Hard bound on the vision-model round trip; the agent is mid-call waiting on the tool result.
VISION_TIMEOUT_MS = 20_000


def make_vision_describer(cfg: BridgeConfig) -> VisionDescriber | None:
    if not cfg.vision_api_url or not cfg.vision_model:
        return None
    url = cfg.vision_api_url
    model = cfg.vision_model
    key = cfg.vision_api_key

    async def describe(frame: dict[str, Any], question: str) -> str:
        who = (
            f"screen shared by {frame.get('participantName') or 'a participant'}"
            if frame.get("source") == "screenshare"
            else f"camera of {frame.get('participantName') or 'the caller'}"
        )
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        body = {
            "model": model,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"This is a live frame from a Microsoft Teams call ({who}). "
                                f"Answer concisely for a voice agent to relay aloud. Question: {question}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{frame.get('mime')};base64,{frame.get('dataBase64')}"},
                        },
                    ],
                }
            ],
        }
        timeout = aiohttp.ClientTimeout(total=VISION_TIMEOUT_MS / 1000)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as res:
                if res.status != 200:
                    raise RuntimeError(f"vision endpoint HTTP {res.status} {await res.text()}")
                data = await res.json()
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if not text:
            raise RuntimeError("vision endpoint returned no content")
        return text

    return describe
