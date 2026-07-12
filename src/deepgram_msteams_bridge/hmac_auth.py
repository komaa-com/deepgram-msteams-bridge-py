"""The HMAC scheme the StandIn media bridge signs upgrades with.

signature = HMAC-SHA256(secret, "{timestampMs}.{callId}") hex-lowercased.
The worker sends it on the WS upgrade in X-OpenClawTeamsBridge-Timestamp /
X-OpenClawTeamsBridge-Signature; the bridge replays the computation.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import math
import time

TIMESTAMP_HEADER = "x-standin-timestamp"
SIGNATURE_HEADER = "x-standin-signature"
# Legacy header names (pre-rename). Still accepted during the transition; the
# StandIn media bridge sends BOTH pairs, so either version interoperates.
LEGACY_TIMESTAMP_HEADER = "x-openclawteamsbridge-timestamp"
LEGACY_SIGNATURE_HEADER = "x-openclawteamsbridge-signature"


def sign(secret: str, timestamp_ms: int | str, call_id: str) -> str:
    payload = f"{timestamp_ms}.{call_id}".encode("utf-8")
    return _hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp_ms: int | str, call_id: str, signature: str) -> bool:
    """Constant-time verification; False on any missing input rather than raising."""
    if not secret or not call_id or not signature:
        return False
    expected = sign(secret, timestamp_ms, call_id)
    return _hmac.compare_digest(expected, signature.lower())


def is_fresh(timestamp_ms: float, window_ms: float, now_ms: float | None = None) -> bool:
    """Timestamp freshness check (the worker documents a +/-60s replay window)."""
    if not isinstance(timestamp_ms, (int, float)) or not math.isfinite(timestamp_ms):
        return False
    now = time.time() * 1000 if now_ms is None else now_ms
    return abs(now - timestamp_ms) <= window_ms
