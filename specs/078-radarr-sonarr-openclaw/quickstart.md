# Quickstart: Radarr and Sonarr OpenClaw Tools

## Local

1. Build the harness:

```bash
npm --prefix tools/openclaw-harness run build
```

2. Run the harness tests:

```bash
npm --prefix tools/openclaw-harness test
```

3. Sync the runtime catalog:

```bash
node tools/openclaw-harness/dist/index.js sync-runtime-catalog
```

4. Inspect the emitted external capabilities:

```bash
node tools/openclaw-harness/dist/index.js runtime-catalog
```

## Host

1. Pull the branch and rebuild:

```bash
tailscale ssh tyler-araujo@100.83.122.99
cd /home/tyler-araujo/Projects/vicuna
bash tools/ops/rebuild-vicuna-runtime.sh
```

2. Verify the runtime and bridge:

```bash
curl http://127.0.0.1:8080/health
journalctl -u vicuna-runtime.service --no-pager -n 120
journalctl -u vicuna-telegram-bridge.service --no-pager -n 120
```

3. Confirm OpenClaw capability visibility:

```bash
journalctl -u vicuna-runtime.service --no-pager -n 200 | rg "OpenClaw tool fabric|radarr|sonarr"
```
