"""CLI entry point: `deepgram-msteams-bridge`.

Entirely env-configured - see .env.example in the package root. A `.env` file
in the working directory is loaded automatically (existing environment wins).
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys

from .config import load_config
from .server import start_server


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so the CLI matches the Node package's
    `node --env-file=.env` convenience without a dependency. Supports
    `KEY=VALUE`, an optional `export ` prefix (files shared with `source`),
    quoted values, and inline ` # comments` on unquoted values. Single-line
    values only. Existing environment variables are never overwritten."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]  # quoted: keep content verbatim (incl. '#')
                else:
                    value = re.sub(r"\s+#.*$", "", value).strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass  # no .env is fine


async def _run() -> None:
    cfg = load_config()
    server = await start_server(cfg)

    # SIGTERM/SIGINT drain: gracefully end every live call instead of
    # hard-dropping calls on a redeploy.
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-POSIX platforms
            pass
    await stop.wait()
    await server.close()


def main() -> None:
    load_dotenv()
    try:
        asyncio.run(_run())
    except ValueError as err:  # config errors (missing/invalid env vars)
        print(f"deepgram-msteams-bridge: {err}", file=sys.stderr)
        print(
            "Required env: DEEPGRAM_API_KEY, WORKER_SHARED_SECRET (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as err:
        if getattr(err, "errno", None) == 48 or "address already in use" in str(err).lower():
            print(f"deepgram-msteams-bridge: port already in use ({err}). Set PORT to a free port.", file=sys.stderr)
        else:
            print(f"deepgram-msteams-bridge: server error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
