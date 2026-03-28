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
- C++17 runtime code, JavaScript ESM on Node 20+, Python 3 tests + `tools/server/server.cpp`, `tools/server/server-emotive-runtime.*`, `tools/telegram-bridge/index.mjs`, `tools/telegram-bridge/lib.mjs`, existing `nlohmann::json`, current Telegram bridge fetch flow, direct Node rendering dependencies for `three` and canvas rasterization, host `ffmpeg` CLI (133-emotive-telegram-video)
- in-memory request/outbox state, bridge state JSON, bounded temporary frame/video artifacts under the bridge host temp directory (133-emotive-telegram-video)
- Bash startup scripts, C++17 runtime, Node 20 harness build, Python 3 pytes + existing `tools/ops/runtime-env.sh`, `tools/ops/sync-openclaw-runtime-state.sh`, OpenClaw harness `npm`/TypeScript build, `nlohmann::json`, pytest runtime harness (134-fix-runtime-tool-catalog)
- existing checkout-local `tools/openclaw-harness/dist/`, stable OpenClaw secrets/catalog paths outside the checkou (134-fix-runtime-tool-catalog)
- C++17 for the provider server, JavaScript ESM and TypeScript on Node 20+ for the harness and Telegram bridge, shell for ops and cron wrappers, Python 3 for targeted tests and Docling integration + existing `tools/server/server.cpp`, `tools/openclaw-harness` runtime catalog/invoke path, `tools/telegram-bridge` Docling ingestion flow, host `cron`, host `flock`, existing pytest and Node test runners (152-cron-task-docs)
- user crontab plus local recurring-task metadata under `/var/lib/vicuna`; local parsed-document bundles under `/home/vicuna/home/docs`; repo-local `.cache/vicuna` fallbacks for developmen (152-cron-task-docs)
- C++17 for the runtime server plus Python 3 for policy-learning tooling and tests + existing `cpp-httplib` client/server stack, `nlohmann::json`, existing metacognitive controller in `tools/server/server-emotive-runtime.*`, existing DeepSeek adapter in `tools/server/server-deepseek.*`, existing policy-learning trainer/evaluator/registry modules under `tools/policy-learning/` (153-policy-api-control)
- in-memory bounded runtime and transition state plus existing file-backed offline policy datasets and registry artifacts (153-policy-api-control)
- C++17 in `tools/server`, TypeScript on Node 20+ in `tools/openclaw-harness`, shell for host install/runtime env, Python 3 for targeted server tests + existing provider request assembly in `tools/server/server.cpp`, existing flattened runtime-tool harness in `tools/openclaw-harness`, existing markdown hard-memory format in `tools/openclaw-harness/src/hard-memory.ts`, pytest provider harness (154-skills-memory-prompt)
- filesystem markdown files under `/home/vicuna/home/skills` for skills and `/var/lib/vicuna/memories` for durable memories (154-skills-memory-prompt)
- C++17 native server/runtime code, Python 3 offline tooling and tests + existing `nlohmann::json`, `cpp-httplib`, current DeepSeek provider runtime, existing pytest provider harness (155-desired-state-reward)
- in-memory bounded transition window plus existing file-backed offline datasets and registry artifacts (155-desired-state-reward)

## Recent Changes
- 114-tool-vad-interleaving: Added C++17 native server/runtime code plus existing Python provider integration tests + existing `cpp-httplib` client/server stack, `nlohmann::json`, existing server-local emotive runtime and VAD projector
