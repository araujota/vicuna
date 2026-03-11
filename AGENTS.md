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

## Preferred AI-Agent Behavior

AI agents working in this repository should:
- read and follow local architecture and spec artifacts first
- prefer Spec Kit artifacts for non-trivial work
- keep runtime policy explicit and inspectable in CPU-side control code
- preserve mathematical and typed representations for memory and self-state surfaces
- add or update tests and documentation alongside behavior changes

## Project References

For build and development guidance, start with:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/build.md](docs/build.md)
- [tools/server/README-dev.md](tools/server/README-dev.md)
