# Systemd

The retained deployment surface is:

- `vicuna-runtime.service`
- `vicuna-telegram-bridge.service`

## Install

```bash
sudo bash tools/ops/install-vicuna-system-service.sh
```

## Rebuild

```bash
bash tools/ops/rebuild-vicuna-runtime.sh
```

## Verify

```bash
sudo systemctl status vicuna-runtime.service --no-pager
sudo systemctl status vicuna-telegram-bridge.service --no-pager
curl http://127.0.0.1:8080/health
```
