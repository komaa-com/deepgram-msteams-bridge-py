# Contributing

Thanks for helping improve `deepgram-msteams-bridge`.

## Local setup

```bash
git clone https://github.com/komaa-com/deepgram-msteams-bridge-py
cd deepgram-msteams-bridge-py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # the full suite runs with no network and no Deepgram account
ruff check src tests
ruff format --check src tests
```

## Conventions

- **One runtime dependency** (`aiohttp`); everything else is dev-only. Please do not add runtime dependencies without discussing first.
- **The relay is verbatim at 16 kHz** - audio is `linear16` at 16 kHz in both directions and the base64 payload must stay untouched on the hot path; there is no resampler. The provider adapter (`deepgram.py`) owns the Voice Agent event framing. Keep that boundary.
- **Tests use a fake `AgentPort`** (see `tests/conftest.py`), so the suite runs without a Deepgram account - including the keepalive loop, the barge-in ghost filter, and the goodbye paths.
- The wire contract with the StandIn media bridge (`protocol.py`, `hmac_auth.py`) is shared with the sibling bridges; changes there need to stay interoperable.
- Error paths matter: a malformed frame from either peer must never escape a WebSocket read loop (that would take down every live call).

## Parity with the Node.js sibling

This package mirrors [`@komaa/deepgram-msteams-bridge`](https://github.com/komaa-com/deepgram-msteams-bridge): same wire contract, same environment variables, same behaviors. When you change observable behavior here, check whether the Node implementation needs the same change (and vice versa) so the two stay drop-in interchangeable.

## Release flow (maintainers)

1. Bump `version` in `pyproject.toml` and commit.
2. Tag `vX.Y.Z` and push the tag; the publish workflow verifies tag == version, runs the suite, and publishes to PyPI via trusted publishing.

## Documentation policy

Document how to **connect to** the hosted StandIn service and how the bridge behaves on the wire. Do not document the internals of the hosted media bridge - this repository only depends on its published wire contract.
