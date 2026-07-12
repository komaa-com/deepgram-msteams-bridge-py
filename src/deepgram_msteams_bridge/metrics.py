"""Dependency-free counters + a call-duration histogram exposed at GET /metrics
in the Prometheus text exposition format (0.0.4). Telephony ops need at
minimum: how many calls, how many right now, how long (p50/p95/p99), and what
is being rejected/dropped.
"""

from __future__ import annotations

_META: dict[str, tuple[str, str]] = {
    "bridge_calls_total": ("Calls accepted (worker sessions created)", "counter"),
    "bridge_calls_active": ("Live calls right now", "gauge"),
    "bridge_call_seconds_total": ("Total call duration in seconds", "counter"),
    "bridge_upgrades_rejected_auth_total": ("Upgrades rejected: bad/stale/replayed HMAC", "counter"),
    "bridge_upgrades_rejected_cap_total": ("Upgrades rejected: connection caps", "counter"),
    "bridge_upgrades_rejected_duplicate_total": ("Upgrades rejected: callId already live (409)", "counter"),
    "bridge_frames_to_agent_total": ("Caller audio frames relayed to Deepgram", "counter"),
    "bridge_frames_to_worker_total": ("Agent audio frames relayed to the worker", "counter"),
    "bridge_frames_dropped_total": ("Frames dropped under worker backpressure", "counter"),
    "bridge_agent_connect_failures_total": ("Deepgram Voice Agent connect failures", "counter"),
    "bridge_agent_errors_total": ("Error events from the Deepgram Voice Agent", "counter"),
    "bridge_injections_refused_total": ("InjectAgentMessage attempts Deepgram refused", "counter"),
}

_counts: dict[str, float] = {}

# Call-duration histogram: what telephony ops actually query (p50/p95/p99).
_HIST_META: dict[str, tuple[str, list[float]]] = {
    "bridge_call_duration_seconds": (
        "Call duration distribution in seconds",
        [30, 60, 120, 300, 600, 1200, 1800, 3600],
    ),
}

_hist: dict[str, dict[str, object]] = {}


def metric_inc(name: str, by: float = 1) -> None:
    if name in _META:
        _counts[name] = _counts.get(name, 0) + by


def metric_dec(name: str) -> None:
    metric_inc(name, -1)


def metric_observe(name: str, value: float) -> None:
    meta = _HIST_META.get(name)
    if meta is None:
        return
    h = _hist.setdefault(name, {"counts": [0] * len(meta[1]), "sum": 0.0, "count": 0})
    counts: list[int] = h["counts"]  # type: ignore[assignment]
    for i, bound in enumerate(meta[1]):
        if value <= bound:
            counts[i] += 1
    h["sum"] = float(h["sum"]) + value  # type: ignore[arg-type]
    h["count"] = int(h["count"]) + 1  # type: ignore[arg-type]


def render_metrics() -> str:
    lines: list[str] = []
    for name, (help_text, mtype) in _META.items():
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        value = _counts.get(name, 0)
        lines.append(f"{name} {value:g}")
    for name, (help_text, buckets) in _HIST_META.items():
        h = _hist.get(name, {"counts": [0] * len(buckets), "sum": 0.0, "count": 0})
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        counts: list[int] = h["counts"]  # type: ignore[assignment]
        for i, bound in enumerate(buckets):
            lines.append(f'{name}_bucket{{le="{bound:g}"}} {counts[i]}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {h["count"]}')
        lines.append(f"{name}_sum {float(h['sum']):g}")  # type: ignore[arg-type]
        lines.append(f"{name}_count {h['count']}")
    return "\n".join(lines) + "\n"


def reset_metrics() -> None:
    """Test isolation: metrics are process-global; tests that assert on them
    call this to start from a clean slate."""
    _counts.clear()
    _hist.clear()
