# Systemd

The retained deployment surface is:

- `vicuna-runtime.service`
- `vicuna-telegram-bridge.service`
- `vicuna-webgl-renderer.service`
- `vicuna-policy-registry.service`
- `vicuna-policy-rollout.service`
- `vicuna-policy-rollout.timer`
- `vicuna-policy-nightly.service`
- `vicuna-policy-nightly.timer`
- `vicuna-runpod-capture-sync.service`
- `vicuna-runpod-capture-sync.timer`
- `vicuna-runpod-mistral-relay.service`
- `vicuna-host-capture-render.service`
- `vicuna-host-capture-render.timer`

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
  `/home/vicuna/home/docs`, and the active
  log root `/var/log/vicuna`

The policy rollout surface is intentionally separate from the runtime install
flow. Operators can copy the policy registry service, rollout controller
service and timer, plus the nightly service and timer units manually after
they have provisioned the required policy-learning env vars. The retained
RunPod capture sync timer may also be enabled when pod-side observation data
should land automatically on the host training capture directory.

## Policy Setup

Install the retained policy units after the main runtime stack is healthy:

```bash
sudo sed \
  -e "s|@VICUNA_REPO_ROOT@|$PWD|g" \
  -e "s|@VICUNA_SERVICE_USER@|vicuna|g" \
  -e "s|@VICUNA_SERVICE_GROUP@|vicuna|g" \
  -e "s|@VICUNA_ENV_FILE@|/etc/vicuna/vicuna.env|g" \
  tools/ops/systemd/vicuna-policy-registry.system.service \
  >/etc/systemd/system/vicuna-policy-registry.service

sudo sed \
  -e "s|@VICUNA_REPO_ROOT@|$PWD|g" \
  -e "s|@VICUNA_SERVICE_USER@|vicuna|g" \
  -e "s|@VICUNA_SERVICE_GROUP@|vicuna|g" \
  -e "s|@VICUNA_ENV_FILE@|/etc/vicuna/vicuna.env|g" \
  tools/ops/systemd/vicuna-policy-nightly.system.service \
  >/etc/systemd/system/vicuna-policy-nightly.service

sudo cp tools/ops/systemd/vicuna-policy-nightly.timer /etc/systemd/system/
sudo sed \
  -e "s|@VICUNA_REPO_ROOT@|$PWD|g" \
  -e "s|@VICUNA_SERVICE_USER@|vicuna|g" \
  -e "s|@VICUNA_SERVICE_GROUP@|vicuna|g" \
  -e "s|@VICUNA_ENV_FILE@|/etc/vicuna/vicuna.env|g" \
  tools/ops/systemd/vicuna-policy-rollout.system.service \
  >/etc/systemd/system/vicuna-policy-rollout.service

sudo cp tools/ops/systemd/vicuna-policy-rollout.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vicuna-policy-registry.service
sudo systemctl enable --now vicuna-policy-rollout.timer
sudo systemctl enable --now vicuna-policy-nightly.timer
sudo systemctl enable --now vicuna-runpod-capture-sync.timer
sudo systemctl enable --now vicuna-runpod-mistral-relay.service
sudo systemctl enable --now vicuna-host-capture-render.timer
```

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
sudo systemctl status vicuna-policy-registry.service --no-pager
sudo systemctl status vicuna-policy-rollout.timer --no-pager
sudo systemctl status vicuna-policy-nightly.timer --no-pager
sudo systemctl status vicuna-runpod-capture-sync.timer --no-pager
sudo systemctl status vicuna-runpod-mistral-relay.service --no-pager
sudo systemctl status vicuna-host-capture-render.timer --no-pager
systemctl --user list-units --type=service --all --no-pager | grep -E 'vicuna|telegram|webgl'
sudo ss -ltnp | grep ':8080'
curl http://127.0.0.1:18081/health
curl http://127.0.0.1:8080/health
sudo systemctl start vicuna-runpod-capture-sync.service
sudo journalctl -u vicuna-runpod-capture-sync.service -n 50 --no-pager
sudo systemctl start vicuna-host-capture-render.service
sudo journalctl -u vicuna-host-capture-render.service -n 50 --no-pager
find /var/log/vicuna -maxdepth 2 -type f | sort
rg -n 'request_id|job_id|operation_id' /var/log/vicuna
```

Healthy steady state means:

- one active system-scoped runtime stack
- no active Vicuña user-scoped units
- one listener on the configured runtime port
- one active policy proposal listener when rollout is enabled
- one active rollout timer when automated candidate progression is desired
- no recurring Telegram `409` polling conflicts in journald
- one active nightly timer when offline training automation is desired
- one active RunPod capture sync timer when pod capture rows should flow to the
  host automatically
- one active RunPod relay service when experimental host mode should forward
  stable OpenAI chat payloads to the pod
- one active host capture render timer when synced emotive trace rows should
  materialize into MP4 artifacts automatically
- active application debugging is centered on `/var/log/vicuna`; journald
  should primarily reflect unit lifecycle and restart state
