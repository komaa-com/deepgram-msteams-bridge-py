"""deepgram-msteams-bridge - public API.

Typical embedding:

    from deepgram_msteams_bridge import load_config, start_server

    server = await start_server(load_config())

Or run the CLI: `deepgram-msteams-bridge` (env-configured, see .env.example).
"""

from .cli import load_dotenv
from .config import BridgeConfig, load_config
from .deepgram import (
    BRIDGE_FUNCTIONS,
    WIRE_SAMPLE_RATE,
    AgentPort,
    CustomTool,
    CustomToolContext,
    DeepgramAgentSocket,
    DgConnector,
    DgSessionHandlers,
    build_prompt,
    build_settings,
    custom_tool_schema,
    synthesize_goodbye,
)
from .hmac_auth import SIGNATURE_HEADER, TIMESTAMP_HEADER, is_fresh, sign, verify
from .log import Logger, logger
from .metrics import render_metrics, reset_metrics
from .protocol import parse_worker_message, pcm16k_bytes_to_ms
from .server import BridgeServer, ReplayGuard, authorize_upgrade, call_id_from_path, start_server
from .session import CallSession, WorkerPort
from .ssrf import assert_public_http_url, fetch_public_image, is_forbidden_ip
from .vision import VisionDescriber, make_vision_describer

__version__ = "0.1.0"

__all__ = [
    "AgentPort",
    "BRIDGE_FUNCTIONS",
    "BridgeConfig",
    "BridgeServer",
    "CallSession",
    "CustomTool",
    "CustomToolContext",
    "DeepgramAgentSocket",
    "DgConnector",
    "DgSessionHandlers",
    "Logger",
    "ReplayGuard",
    "SIGNATURE_HEADER",
    "TIMESTAMP_HEADER",
    "VisionDescriber",
    "WIRE_SAMPLE_RATE",
    "WorkerPort",
    "__version__",
    "assert_public_http_url",
    "authorize_upgrade",
    "build_prompt",
    "build_settings",
    "call_id_from_path",
    "custom_tool_schema",
    "fetch_public_image",
    "is_forbidden_ip",
    "is_fresh",
    "load_config",
    "load_dotenv",
    "logger",
    "make_vision_describer",
    "parse_worker_message",
    "pcm16k_bytes_to_ms",
    "render_metrics",
    "reset_metrics",
    "sign",
    "start_server",
    "synthesize_goodbye",
    "verify",
]
