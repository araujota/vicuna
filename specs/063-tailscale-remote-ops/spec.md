# Feature Spec: Tailscale Remote Ops Access

## Summary

Install Tailscale on the workstation and enable a remote access path that lets
operators inspect system logs and run runtime rebuild commands without exposing
the host broadly on the LAN or public internet.

## Requirements

1. The workstation must install and enable the official Tailscale service for
   Ubuntu 24.04.
2. Remote operator access must use an explicit, inspectable path suitable for
   log inspection and rebuild commands.
3. The chosen access path must be documented in the repository's ops notes so
   future operators can repeat it.
4. Runtime and Telegram bridge service management commands must remain the same
   once connected remotely.
5. Verify the installed host state and report any remaining interactive auth
   step required to complete enrollment.
