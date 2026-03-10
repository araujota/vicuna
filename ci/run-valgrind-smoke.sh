#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-build-valgrind}"
log_dir="${repo_root}/${build_dir}/valgrind-logs"

mkdir -p "${log_dir}"
cd "${repo_root}"

valgrind_args=(
  --tool=memcheck
  --error-exitcode=1
  --leak-check=full
  --show-leak-kinds=all
  --errors-for-leak-kinds=definite,possible
  --track-origins=yes
  --num-callers=50
)

tests=(
  test-alloc
  test-arg-parser
  test-chat-auto-parser
  test-chat-peg-parser
  test-chat-template
  test-gguf
  test-grammar-integration
  test-grammar-parser
  test-jinja
  test-json-partial
  test-llama-grammar
  test-log
  test-mtmd-c-api
  test-regex-partial
  test-sampling
)

for test_name in "${tests[@]}"; do
  binary="${repo_root}/${build_dir}/bin/${test_name}"
  log_file="${log_dir}/${test_name}.log"

  if [[ ! -x "${binary}" ]]; then
    echo "Missing expected test binary: ${binary}" >&2
    exit 1
  fi

  echo "::group::Valgrind ${test_name}"
  valgrind "${valgrind_args[@]}" \
    --log-file="${log_file}" \
    "${binary}"
  echo "::endgroup::"
done
