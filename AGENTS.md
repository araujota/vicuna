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
