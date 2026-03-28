# Systemd

The retained deployment surface is:

- `vicuna-runtime.service`
- `vicuna-telegram-bridge.service`
- `vicuna-webgl-renderer.service`
- `vicuna-policy-nightly.service`
- `vicuna-policy-nightly.timer`

This is a system-scoped deployment surface. The host should not also keep
active Vicuña user-scoped units under `systemd --user`.

## Install

```bash
sudo bash tools/ops/install-vicuna-system-service.sh
```

The install flow is also the convergence flow. It is responsible for:

- removing stale Vicuña user unit files
- disabling stale Vicuña user services when a user bus is present
- clearing stale system override drop-ins that point at older repo roots
- reinstalling all canonical system units from the current repo root
- provisioning durable runtime roots such as `/var/lib/vicuna/memories`,
  `/var/lib/vicuna/ongoing-tasks`, and `/home/vicuna/home/docs`

Recurring background work is no longer a server-owned idle worker. The install
and rebuild flows must therefore preserve the `vicuna` user's cron capability
and the retained runner path configured through `VICUNA_ONGOING_TASKS_*`.

The nightly policy batch is intentionally separate from the runtime install
flow. Operators can copy the timer and service units manually after they have
provisioned the required policy-learning env vars.

## Rebuild

```bash
bash tools/ops/rebuild-vicuna-runtime.sh
```

The rebuild flow assumes the host is already converged. It now fails fast if:

- both user-scoped and system-scoped Vicuña units are installed
- the active unit repo roots do not match the configured canonical repo root
- the runtime port is still owned after stopping the canonical runtime

## Verify

```bash
sudo systemctl status vicuna-runtime.service --no-pager
sudo systemctl status vicuna-telegram-bridge.service --no-pager
sudo systemctl status vicuna-webgl-renderer.service --no-pager
sudo systemctl status vicuna-policy-nightly.timer --no-pager
systemctl --user list-units --type=service --all --no-pager | grep -E 'vicuna|telegram|webgl'
sudo ss -ltnp | grep ':8080'
curl http://127.0.0.1:8080/health
```

Healthy steady state means:

- one active system-scoped runtime stack
- no active Vicuña user-scoped units
- one listener on the configured runtime port
- no recurring Telegram `409` polling conflicts in journald
- one active nightly timer when offline training automation is desired
