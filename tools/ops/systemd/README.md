# System Service Migration

Vicuña can run either as a user service or as a dedicated system service. For a
durable workstation deployment, prefer a dedicated `vicuna` account managed by
root-owned systemd units.

## Install

Run the installer as root from the repository checkout:

```bash
sudo bash tools/ops/install-vicuna-system-service.sh
```

The installer:

- creates or updates the `vicuna` service account
- adds GPU-related supplementary groups when present (`video`, `render`)
- grants ACL-based read/write access to the repository worktree without
  changing the interactive owner's primary ownership
- writes `/etc/vicuna/vicuna.env`
- installs system units into `/etc/systemd/system/`
- disables the matching `--user` services when it can reach the owner's user
  bus
- enables and restarts the system services
- sets a durable Node-wrapper file-descriptor floor through
  `VICUNA_BASH_TOOL_MAX_OPEN_FILES=256` and `LimitNOFILE=65536` on the runtime
  unit

## Rebuild

After migration, rebuild using the same repo script. It understands
`VICUNA_SYSTEMD_SCOPE=system` from `/etc/vicuna/vicuna.env` and will switch to
systemd's system scope automatically:

```bash
bash tools/ops/rebuild-vicuna-runtime.sh --allow-busy-stop
```

If you are not root, the rebuild helper will invoke `sudo systemctl` for the
service stop/start/reset steps.

## Verify

```bash
sudo systemctl status vicuna-runtime.service --no-pager
sudo systemctl status vicuna-telegram-bridge.service --no-pager
systemctl cat vicuna-runtime.service | grep LimitNOFILE
grep '^VICUNA_BASH_TOOL_MAX_OPEN_FILES=' /etc/vicuna/vicuna.env
curl http://127.0.0.1:8080/health
```

## Notes

- The installer expects a Node.js `>= 20.16` binary for the Telegram bridge. If
  it cannot auto-detect one, set `TELEGRAM_BRIDGE_NODE_BIN` when running the
  installer.
- Root is still required for account creation, ACL changes outside the current
  user, and system service installation.
- Worktree write access is granted through POSIX ACLs using `setfacl`, not by
  reassigning repository ownership to the service account.

## Remote Ops Over Tailscale SSH

For workstation-style remote administration, prefer Tailscale SSH instead of
opening the host's LAN SSH port. After installing Tailscale and bringing the
host up with `sudo tailscale up --ssh`, operators can connect over the
machine's Tailscale identity on port `22` and use the same local commands:

```bash
tailscale ssh tyler-araujo@<machine-name-or-tailscale-ip>
sudo systemctl status vicuna-runtime.service --no-pager
sudo systemctl status vicuna-telegram-bridge.service --no-pager
journalctl -u vicuna-runtime.service --no-pager -n 120
cd /home/tyler-araujo/Projects/vicuna
bash tools/ops/rebuild-vicuna-runtime.sh
```

Tailscale enrollment may still require an interactive browser login or an auth
key supplied by the tailnet operator.
