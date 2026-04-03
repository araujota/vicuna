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

- `tools/server`: host runtime, DeepSeek standard path, RunPod experimental relay,
  and experimental capture persistence
- `tools/telegram-bridge`: Telegram delivery plus WebGL-backed response video
- `tools/openclaw-harness`: host-owned hard memory, skills, host shell, and
  Telegram relay tool surfaces
- `tools/policy-learning`: offline PPO / GRU / cvec training, evaluation, and
  registry infrastructure
- `tools/ops`: host service install, RunPod connector helpers, capture sync, and
  render automation

Work should remove or avoid reintroducing:

- DMN and self-state orchestration surfaces
- pod-local inference runtime code and bootstrap assets
- generic upstream `llama.cpp` distribution assets
- media-management, literature-search, and unrelated experimental wrappers

## Project References

- [README.md](/Users/tyleraraujo/vicuna/README.md)
- [ARCHITECTURE.md](/Users/tyleraraujo/vicuna/ARCHITECTURE.md)
- [docs/build.md](/Users/tyleraraujo/vicuna/docs/build.md)
- [tools/server/README-dev.md](/Users/tyleraraujo/vicuna/tools/server/README-dev.md)

## Active Technologies
- C++17 in `tools/server` with `cpp-httplib` and `nlohmann::json`
- JavaScript ESM on Node 20+ in `tools/telegram-bridge` for Telegram transport and
  WebGL rendering orchestration
- TypeScript on Node 20+ in `tools/openclaw-harness` for host-owned tool access
- Python 3 in `tools/policy-learning` for offline dataset, trainer, evaluator,
  and registry flows
- file-backed state under `/var/lib/vicuna` plus host-owned memories and skills

## Recent Changes
- Repository narrowed to host runtime, bridge, RunPod connector, RL data
  infrastructure, tool harness, and renderer surfaces only
