# basic-bridge example

A minimal embedding of `deepgram-msteams-bridge`: env-driven config, a custom
vision hook (your model answers the agent's `look` tool), and a custom
`lookup_order` function tool the bridge executes.

```bash
pip install deepgram-msteams-bridge
cp ../../.env.example .env    # fill in DEEPGRAM_API_KEY and WORKER_SHARED_SECRET
python main.py
```

Expose port 8080 with a tunnel (e.g. `tailscale funnel --bg --https=8080 8080`)
and register the printed `wss://.../voice/msteams/stream` URL as your StandIn
identity's Agent voice URL. Place a Teams call: ask the agent "where is order
KO-12?" to see the custom tool fire.
