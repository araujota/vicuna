# Instructions for Vicuña

Vicuña is a provider-first runtime repository.

## Mandatory Workflow

For every work request in this repository, AI agents must:

- perform GitHub codebase research using the `github-codebase-research` skill
  when repository history, comparable implementations, upstream/downstream
  patterns, or cross-repo examples could inform the work
- perform web research when external technical context, papers, standards,
  methods, or current implementation guidance could materially affect the work
- use the full Spec Kit waterfall for all non-trivial work: `specify`, `plan`,
  `tasks`, then implementation
- treat implementation as blocked until the relevant Spec Kit artifacts exist
  and reflect the current request
- keep runtime behavior explicit and inspectable in CPU-side control code
- update tests and documentation with behavior changes

## Repository Direction

The active product surface is:

- `tools/server`: the provider-first HTTP server
- `tools/server/server-emotive-runtime.*`: block-wise emotive moment estimation
- `tools/server/server-embedding-backend.*`: optional local embedding-only
  backend
- `tools/telegram-bridge`: the retained Telegram dialogue transport

Work should remove or avoid reintroducing:

- DMN and self-state orchestration surfaces
- OpenClaw harness and catalog plumbing
- generic upstream `llama.cpp` distribution assets that are not required by the
  provider-first runtime plus bridge

## Project References

- [README.md](/Users/tyleraraujo/vicuna/README.md)
- [ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [docs/build.md](/Users/tyleraraujo/vicuna/docs/build.md)
- [tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md)

## Active Technologies
- C++17 native server/runtime code plus existing Python provider integration tests + existing `cpp-httplib` client/server stack, `nlohmann::json`, existing server-local emotive runtime and VAD projector (114-tool-vad-interleaving)
- In-memory per-request state plus existing bounded latest-trace history only (114-tool-vad-interleaving)
- C++17 native server/runtime code plus existing Python provider integration tests + existing `cpp-httplib`, `nlohmann::json`, current DeepSeek provider flow, current emotive runtime, shared HTTP helpers in `common/http.h` (118-idle-ongoing-tasks)
- hard-memory-backed ongoing-task registry read over HTTP plus existing in-memory worker state and emotive trace history (118-idle-ongoing-tasks)
- C++17 native server/runtime code plus Python provider integration tests + existing `nlohmann::json`, current provider request assembly, current emotive runtime, existing tool catalog metadata in `tools/openclaw-harness` (119-staged-tool-loop)
- existing in-memory request/controller state plus existing hard-memory/replay persistence (119-staged-tool-loop)
- C++17 and Python 3 + `tools/server/server.cpp`, `tools/server/server-deepseek.cpp`, provider test harness in `tools/server/tests/unit/test_deepseek_provider.py` (121-reasoning-token-cap)
- C++17, JavaScript ESM on Node 20+, and TypeScript for OpenClaw harness utilities + `tools/telegram-bridge/index.mjs`, `tools/telegram-bridge/lib.mjs`, `tools/openclaw-harness/src/*.ts`, `tools/server/server.cpp`, Node child-process execution, existing OpenClaw wrapper binaries under `tools/openclaw-harness/bin/` (122-tool-catalog-execution)
- file-backed OpenClaw secrets/runtime catalog under `.cache/vicuna`, bridge state JSON under `/var/lib/vicuna`, and existing hard-memory-backed wrapper backends (122-tool-catalog-execution)
- C++17 for runtime code, Python 3 for unit tests, shell for ops scripts + `httplib`, `nlohmann::json`, pytest provider harness (124-deepseek-v32-thinking)

## Recent Changes
- 114-tool-vad-interleaving: Added C++17 native server/runtime code plus existing Python provider integration tests + existing `cpp-httplib` client/server stack, `nlohmann::json`, existing server-local emotive runtime and VAD projector
