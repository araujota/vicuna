# Runtime Rebuild Process

Use [`tools/ops/rebuild-vicuna-runtime.sh`](/home/tyler-araujo/Projects/vicuna/tools/ops/rebuild-vicuna-runtime.sh)
for host rebuilds that should preserve runtime state.

## Default Contract

The rebuild flow is state-preserving by default:

- it requires `VICUNA_RUNTIME_STATE_PATH`
- it refuses to rebuild while runtime persistence is disabled or unhealthy
- it refuses to stop a runtime that still reports active or pending work unless
  explicitly overridden
- it stops the service, backs up the current snapshot and provenance log, rebuilds
  `llama-server`, restarts the service, and verifies the restored runtime health

The shared environment defaults live in
[`tools/ops/runtime-env.sh`](/home/tyler-araujo/Projects/vicuna/tools/ops/runtime-env.sh).
If no explicit snapshot path is supplied, the runtime now defaults to:

- snapshot: `$REPO_ROOT/.cache/vicuna/runtime-state.json`
- backups: `$REPO_ROOT/.cache/vicuna/runtime-state-backups/`

## Preserved Surfaces

When runtime persistence is enabled and compatible, the server snapshot preserves
the runtime-managed state currently serialized by `server_context`, including:

- self-model updater program and self-state trace
- self-model extensions
- proactive mailbox state
- bash-tool and hard-memory configuration
- functional LoRA snapshot archives
- process-functional entries and snapshot archives

That is the supported rebuild path for stateful runtime upgrades.

## Current Limits

The rebuild contract should not overclaim beyond the current snapshot schema.

- functional and process-functional LoRA snapshot state is preserved
- self-model state is preserved
- temporal self-improvement traces are currently exposed through provenance and
  observability, but are not serialized into the runtime snapshot in the same
  way as the functional/process-functional snapshot archives

If a future rebuild needs temporal trace continuity to survive restart, that
requires an additional persistence change rather than only using the rebuild
script.

## Explicit Reset Exception

Use `--allow-state-reset` only when the rebuild is intentionally changing the
exact persisted surfaces or their compatibility boundary. Examples:

- snapshot schema/version changes
- self-model serialization layout changes
- functional or process-functional snapshot blob compatibility changes
- any rebuild where replaying the old snapshot would be misleading or unsafe

If the rebuild is changing those surfaces, the operator must make that reset
decision explicit instead of silently restarting against incompatible state.

## Usage

```bash
tools/ops/rebuild-vicuna-runtime.sh
```

For intentional state-breaking rebuilds:

```bash
tools/ops/rebuild-vicuna-runtime.sh --allow-state-reset
```
