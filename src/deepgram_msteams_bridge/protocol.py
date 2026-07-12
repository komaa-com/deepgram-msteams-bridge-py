"""Worker wire protocol: the JSON messages the StandIn media bridge speaks.

JSON, camelCase properties, discriminated on "type"; keep field names in exact
sync with the wire. Messages are handled as plain dicts (they arrive and leave
as JSON); this module holds the parse guard and the audio-time helper.

Worker -> bridge types: session.start, session.end, recording.status,
audio.frame, video.frame, participants, dtmf, ping, assistant.say.
Bridge -> worker types: audio.frame, assistant.cancel, pong, session.end,
expression, display.image.
"""

from __future__ import annotations

import json
from typing import Any


def parse_worker_message(raw: str | bytes) -> dict[str, Any] | None:
    """Parse a worker frame; None on junk rather than raising (drop + log at call site)."""
    try:
        obj = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("type"), str):
        return None
    return obj


def pcm16k_bytes_to_ms(n_bytes: float) -> float:
    """PCM16K byte length -> milliseconds (16 kHz x 2 bytes = 32 bytes per ms)."""
    return n_bytes / 32
