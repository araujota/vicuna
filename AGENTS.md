# Instructions for Vicuña

This repository began as a fork of `llama.cpp`, but Vicuña is now developed as its own project and follows its own contribution and architecture rules.

## AI Usage

AI-assisted development is permitted in this repository.

Contributors remain responsible for:
- understanding the code they submit
- reviewing generated changes before merging
- validating behavior with targeted tests and benchmarks when relevant
- keeping architecture, data-model, and implementation artifacts aligned

AI should be used as a force multiplier, not as a substitute for engineering judgment.

## Mandatory Workflow

For every work request in this repository, AI agents must:
- perform GitHub codebase research using the `github-codebase-research` skill when repository history, comparable implementations, upstream/downstream patterns, or cross-repo examples could inform the work
- perform web research for current external technical context, papers, standards, methods, or implementation guidance whenever the work depends on knowledge outside the local repository
- use the full Spec Kit waterfall for all work requests, in order: `specify`, `plan`, `tasks`, then implementation
- do not create or switch to a new branch on every Spec Kit invocation; stay on the current branch unless the user explicitly requests a new branch or the workflow requires a one-time branch creation for the task
- treat implementation as blocked until the relevant Spec Kit artifacts exist and reflect the current request
- keep runtime policy explicit and inspectable in CPU-side control code
- preserve mathematical and typed representations for memory and self-state surfaces
- add or update tests and documentation alongside behavior changes

These requirements apply to all work requests, including research, design, implementation, refactors, fixes, and documentation changes.

## Preferred AI-Agent Behavior

AI agents working in this repository should:
- read and follow local architecture and Spec Kit artifacts first
- treat GitHub research, web research, and Spec Kit artifacts as first-class inputs, not optional supplements
- avoid skipping directly to implementation unless the full research-and-spec workflow for the current request is already complete and still current

## Project References

For build and development guidance, start with:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/build.md](docs/build.md)
- [tools/server/README-dev.md](tools/server/README-dev.md)

## Workstation Access

As of 2026-03-20, this workstation reports:
- hostname: `tyler-araujo-MS-7E12`
- primary LAN IPv4: `10.0.0.20`
- Tailscale IPv4: `100.83.122.99`
- Tailscale DNS name: `vicuna-host`
- repo path: `/home/tyler-araujo/Projects/vicuna`
- default user: `tyler-araujo`

Preferred remote admin path:
- `tailscale ssh tyler-araujo@100.83.122.99`
- `tailscale ssh tyler-araujo@vicuna-host`

Requirements:
- the client device must also be enrolled in the same Tailscale tailnet
- Tailscale policy must allow SSH access to this node for the connecting user

Remote rebuild/deploy workflow:
- `tailscale ssh tyler-araujo@100.83.122.99`
- `cd /home/tyler-araujo/Projects/vicuna`
- `bash tools/ops/rebuild-vicuna-runtime.sh`

Runtime verification after rebuild:
- `curl http://127.0.0.1:8080/health`
- `journalctl -u vicuna-runtime.service --no-pager -n 120`
- `journalctl -u vicuna-telegram-bridge.service --no-pager -n 120`

## Active Technologies
- C++17 with the existing `llama.cpp`/Vicuña runtime and C API surfaces + Existing Active LoRA manager, cognitive loop runtime, self-state and server serialization surfaces (044-process-functional-lora)
- In-memory runtime bank plus existing server export/import and typed state surfaces (044-process-functional-lora)
- C++17 for runtime and server code, existing TypeScript/Node.js support surfaces where already presen + Existing `llama.cpp`/Vicuña runtime, self-state runtime, Active LoRA manager, functional LoRA stack, HTTP server/OpenClaw tool fabric, existing telemetry/provenance surfaces (070-truth-runtime-refactor)
- In-memory runtime state with existing snapshot/export/import surfaces for self-state, provenance, and LoRA archives (070-truth-runtime-refactor)
- C++17 and TypeScript/Node.js + `server_context`, `server_task`, OpenClaw fabric, common OpenClaw catalog types, Telegram bridge middleware, TypeScript OpenClaw harness, existing media wrappers (083-async-ack-react)
- Runtime snapshot JSON, proactive mailbox state, Telegram dialogue state, OpenClaw catalog JSON, OpenClaw tool secrets JSON (083-async-ack-react)

## Recent Changes
- 044-process-functional-lora: Added C++17 with the existing `llama.cpp`/Vicuña runtime and C API surfaces + Existing Active LoRA manager, cognitive loop runtime, self-state and server serialization surfaces
