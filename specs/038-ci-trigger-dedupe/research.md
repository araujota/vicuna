# Research: CI Trigger Dedupe

## Local Findings

- PR `#2` currently shows duplicate validation results because the same branch update launches both `push` and `pull_request` runs of `Vicuna CI`.
- The duplicate failed checks came from two workflow runs of the same workflow:
  - `23102747534` (`push`)
  - `23102748010` (`pull_request`)
- Additional PR check duplication exists for standalone validation workflows such as `pyright type-check`, `WebUI Checks`, and `check-vendor`, each of which still has both `push` and `pull_request` triggers locally.

## Workflow Scope

Validation workflows still using both `push` and `pull_request` include:

- `.github/workflows/vicuna-ci.yml`
- `.github/workflows/python-lint.yml`
- `.github/workflows/python-type-check.yml`
- `.github/workflows/server-webui.yml`
- `.github/workflows/check-vendor.yml`
- `.github/workflows/pre-tokenizer-hashes.yml`
- `.github/workflows/python-check-requirements.yml`
- `.github/workflows/editorconfig.yml`
- `.github/workflows/update-ops-docs.yml`
- `.github/workflows/copilot-setup-steps.yml`

PR-specific automation that should remain PR-triggered includes:

- `.github/workflows/labeler.yml`

## External References

- GitHub Actions event model allows independent `push` and `pull_request` triggers, which means using both on the same validation workflow creates separate runs unless the workflow is intentionally scoped otherwise.
- Upstream `ggerganov/llama.cpp` still uses both triggers for some workflows, but Vicuña's current goal is different: reduce redundant PR validation noise and run these checks only on branch pushes.
- Repository-wide style/static-analysis scans immediately surfaced inherited whole-tree debt, especially `clang-format` drift outside the user-touched surface. To preserve full-tree visibility without blocking every push forever, those gates need checked-in baselines analogous to the lizard baseline already used in this branch.
- `python-server-tests` currently depends on `lint-type-quality`, which suppresses independent test results whenever lint fails. Removing that dependency is necessary to satisfy the “single push shows the full blocking set” goal.
- The Clang sanitizer job currently fails while linking `tests/test-c.c` because the target links a UBSan-instrumented C++ shared library via the C linker, which omits the C++ UBSan runtime symbols. For sanitizer coverage to stay meaningful, the mixed-language test target needs to link as C++.
