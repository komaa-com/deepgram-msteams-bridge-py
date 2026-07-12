---
title: "Contributing"
description: "Local setup, conventions, and the documentation policy."
---

Contributions are welcome.

## Quick start for contributors

```bash
git clone https://github.com/komaa-com/deepgram-msteams-bridge-py
cd deepgram-msteams-bridge-py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # 101 tests, no network or Deepgram account needed
ruff check src tests
ruff format --check src tests
```

- **One runtime dependency** (`aiohttp`); everything else is dev-only.
- **The relay is copy-only at 16 kHz** - the Voice Agent session is pinned to `linear16` @ 16000, and the provider adapter (`deepgram.py`) owns the base64/binary framing. Keep that boundary.
- **Tests use a fake `AgentPort`** (see `tests/conftest.py`), so the suite runs without a Deepgram account - including the `SettingsApplied` ordering gate and the barge-in ghost filter.
- CI runs ruff + the test suite on Python **3.10 through 3.13** for every push and PR.
- Releases are tagged `v*` and published to PyPI by CI via trusted publishing - the tag must match the `pyproject.toml` version.
- **Docs live in `website/`** (this site). Any merged change to `website/` redeploys the site automatically.

## Parity with the Node.js sibling

This package mirrors [`@komaa/deepgram-msteams-bridge`](https://github.com/komaa-com/deepgram-msteams-bridge): same wire contract, same environment variables, same behaviors. When you change observable behavior here, check whether the Node implementation needs the same change (and vice versa) so the two stay drop-in interchangeable.

## Documentation policy

Document how to **connect to** the hosted StandIn service and how the bridge behaves on the wire. Do not document the internals of the hosted media bridge - this repository only depends on its published wire contract.
