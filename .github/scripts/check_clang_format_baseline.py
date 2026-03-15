#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def normalize_path(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return path


def parse_diagnostics(text: str, root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    needle = ": error: code should be clang-formatted"

    for raw_line in text.splitlines():
        if needle not in raw_line:
            continue
        _, _, tail = raw_line.partition("Z ")
        line = tail or raw_line
        path, _, _ = line.partition(":")
        if path:
            counts[normalize_path(path, root)] += 1

    return counts


def print_stream(text: str, *, stream: object = sys.stdout) -> None:
    if text:
        print(text, file=stream, end="" if text.endswith("\n") else "\n")


def run_clang_format_cmd(args: list[str], binary: str, root: Path) -> tuple[int, Counter[str]]:
    proc = subprocess.run([binary, "--dry-run", "--Werror", *args], capture_output=True, text=True)
    print_stream(proc.stdout)
    print_stream(proc.stderr, stream=sys.stderr)
    counts = parse_diagnostics(proc.stdout, root)
    counts.update(parse_diagnostics(proc.stderr, root))
    return proc.returncode, counts


def run_clang_format(files: list[str], binary: str, root: Path) -> tuple[int, Counter[str], list[tuple[str, int]]]:
    status = 0
    counts: Counter[str] = Counter()
    tool_failures: list[tuple[str, int]] = []

    for chunk in chunked(files, 100):
        rc, chunk_counts = run_clang_format_cmd(chunk, binary, root)
        counts.update(chunk_counts)
        if rc in (0, 1):
            if rc != 0:
                status = 1
            continue

        status = 1
        for file_path in chunk:
            rc, single_counts = run_clang_format_cmd([file_path], binary, root)
            counts.update(single_counts)
            if rc not in (0, 1):
                tool_failures.append((file_path, rc))
                continue
            if rc != 0:
                status = 1
        if rc != 0:
            status = 1

    return status, counts, tool_failures


def read_baseline(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        count_str, _, file_path = line.partition("\t")
        counts[file_path] = int(count_str)
    return counts


def write_baseline(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{counts[file_path]}\t{file_path}" for file_path in sorted(counts)]
    path.write_text("".join(f"{line}\n" for line in lines))


def compare(current: Counter[str], baseline: Counter[str]) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    new_files: list[tuple[str, int]] = []
    regressed: list[tuple[str, int, int]] = []
    resolved: list[tuple[str, int, int]] = []

    current_files = set(current)
    baseline_files = set(baseline)

    for file_path in sorted(current_files - baseline_files):
        new_files.append((file_path, current[file_path]))

    for file_path in sorted(current_files & baseline_files):
        current_count = current[file_path]
        baseline_count = baseline[file_path]
        if current_count > baseline_count:
            regressed.append((file_path, baseline_count, current_count))
        elif current_count < baseline_count:
            resolved.append((file_path, baseline_count, current_count))

    for file_path in sorted(baseline_files - current_files):
        resolved.append((file_path, baseline[file_path], 0))

    return new_files, regressed, resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clang-format across the repository and compare against a checked-in baseline.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline file, relative to the repo root unless absolute.")
    parser.add_argument("--clang-format", default="clang-format", dest="binary", help="clang-format executable to use.")
    parser.add_argument("--write-baseline", action="store_true", help="Overwrite the baseline file with the current violation counts.")
    parser.add_argument("files", nargs="+", help="Tracked native source files to validate.")
    args = parser.parse_args()

    root = repo_root()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    status, counts, tool_failures = run_clang_format(args.files, args.binary, root)
    if args.write_baseline:
        write_baseline(baseline_path, counts)
        print(f"Wrote {len(counts)} clang-format baseline entries to {baseline_path.relative_to(root)}")
        return 0

    baseline = read_baseline(baseline_path)
    new_files, regressed, resolved = compare(counts, baseline)

    print("::group::clang-format baseline comparison")
    print(f"baseline files with formatting drift: {len(baseline)}")
    print(f"current files with formatting drift: {len(counts)}")

    if new_files:
        print("new files with formatting drift:")
        for file_path, count in new_files:
            print(f"{count}\t{file_path}")
    else:
        print("new files with formatting drift: none")

    if regressed:
        print("regressed files with additional formatting drift:")
        for file_path, old_count, new_count in regressed:
            print(f"{old_count} -> {new_count}\t{file_path}")
    else:
        print("regressed files with additional formatting drift: none")

    if resolved:
        print("resolved or improved baseline files:")
        for file_path, old_count, new_count in resolved:
            print(f"{old_count} -> {new_count}\t{file_path}")
    else:
        print("resolved or improved baseline files: none")

    if tool_failures:
        print("tool execution failures:")
        for file_path, rc in tool_failures:
            print(f"{rc}\t{file_path}")
    else:
        print("tool execution failures: none")
    print("::endgroup::")

    return 1 if new_files or regressed or tool_failures or status else 0


if __name__ == "__main__":
    raise SystemExit(main())
