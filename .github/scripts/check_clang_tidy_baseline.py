#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_stream(text: str, *, stream: object = sys.stdout) -> None:
    if not text:
        return
    suffix = "" if text.endswith("\n") else "\n"
    stream.write(f"{text}{suffix}")


def write_line(text: str = "", *, stream: object = sys.stdout) -> None:
    stream.write(f"{text}\n")


def normalize_path(path: str, root: Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = root / path_obj
    try:
        return str(path_obj.resolve().relative_to(root))
    except ValueError:
        return path


def parse_diagnostics(text: str, root: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()

    for raw_line in text.splitlines():
        _, _, tail = raw_line.partition("Z ")
        line = tail or raw_line

        if ": warning: " not in line and ": error: " not in line:
            continue
        if line.startswith("Error while") or line.startswith("Suppressed "):
            continue

        parts = line.rsplit("[", 1)
        if len(parts) != 2 or not parts[1].endswith("]"):
            continue

        head = parts[0]
        check_name = parts[1][:-1].strip()
        file_path, _, _ = head.partition(":")

        if file_path and check_name:
            counts[(normalize_path(file_path, root), check_name)] += 1

    return counts


def run_clang_tidy(
    files: list[str],
    build_dir: str,
    header_filter: str,
    binary: str,
    root: Path,
) -> tuple[int, Counter[tuple[str, str]], list[tuple[str, int]]]:
    counts: Counter[tuple[str, str]] = Counter()
    tool_failures: list[tuple[str, int]] = []

    for file_path in files:
        cmd = [
            binary,
            "-p",
            build_dir,
            "--quiet",
            "--warnings-as-errors=*",
            f"--header-filter={header_filter}",
            file_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        write_stream(proc.stdout)
        write_stream(proc.stderr, stream=sys.stderr)
        counts.update(parse_diagnostics(proc.stdout, root))
        counts.update(parse_diagnostics(proc.stderr, root))
        if proc.returncode not in (0, 1):
            tool_failures.append((file_path, proc.returncode))
            continue

    return 0, counts, tool_failures


def read_baseline(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not path.exists():
        return counts

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        count_str, _, remainder = line.partition("\t")
        check_name, _, file_path = remainder.partition("\t")
        counts[(normalize_path(file_path, repo_root()), check_name)] = int(count_str)
    return counts


def write_baseline(path: Path, counts: Counter[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{counts[(file_path, check_name)]}\t{check_name}\t{file_path}"
        for file_path, check_name in sorted(counts)
    ]
    path.write_text("".join(f"{line}\n" for line in lines))


def compare(
    current: Counter[tuple[str, str]],
    baseline: Counter[tuple[str, str]],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int, int]], list[tuple[str, str, int, int]]]:
    new_findings: list[tuple[str, str, int]] = []
    regressed: list[tuple[str, str, int, int]] = []
    resolved: list[tuple[str, str, int, int]] = []

    current_keys = set(current)
    baseline_keys = set(baseline)

    for file_path, check_name in sorted(current_keys - baseline_keys):
        new_findings.append((file_path, check_name, current[(file_path, check_name)]))

    for key in sorted(current_keys & baseline_keys):
        current_count = current[key]
        baseline_count = baseline[key]
        if current_count > baseline_count:
            regressed.append((key[0], key[1], baseline_count, current_count))
        elif current_count < baseline_count:
            resolved.append((key[0], key[1], baseline_count, current_count))

    for file_path, check_name in sorted(baseline_keys - current_keys):
        resolved.append((file_path, check_name, baseline[(file_path, check_name)], 0))

    return new_findings, regressed, resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clang-tidy across the repository and compare against a checked-in baseline.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline file, relative to the repo root unless absolute.")
    parser.add_argument("--build-dir", required=True, help="Compilation database directory for clang-tidy.")
    parser.add_argument("--header-filter", required=True, help="Header filter passed through to clang-tidy.")
    parser.add_argument("--clang-tidy", default="clang-tidy", dest="binary", help="clang-tidy executable to use.")
    parser.add_argument("--write-baseline", action="store_true", help="Overwrite the baseline file with the current finding counts.")
    parser.add_argument("files", nargs="+", help="Tracked native source files to validate.")
    args = parser.parse_args()

    root = repo_root()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    _, counts, tool_failures = run_clang_tidy(args.files, args.build_dir, args.header_filter, args.binary, root)
    if args.write_baseline:
        write_baseline(baseline_path, counts)
        write_line(f"Wrote {len(counts)} clang-tidy baseline entries to {baseline_path.relative_to(root)}")
        return 0

    baseline = read_baseline(baseline_path)
    new_findings, regressed, resolved = compare(counts, baseline)

    write_line("::group::clang-tidy baseline comparison")
    write_line(f"baseline diagnostics: {sum(baseline.values())}")
    write_line(f"current diagnostics: {sum(counts.values())}")

    if new_findings:
        write_line("new diagnostics:")
        for file_path, check_name, count in new_findings:
            write_line(f"{count}\t{check_name}\t{file_path}")
    else:
        write_line("new diagnostics: none")

    if regressed:
        write_line("regressed diagnostics:")
        for file_path, check_name, old_count, new_count in regressed:
            write_line(f"{old_count} -> {new_count}\t{check_name}\t{file_path}")
    else:
        write_line("regressed diagnostics: none")

    if resolved:
        write_line("resolved or improved diagnostics:")
        for file_path, check_name, old_count, new_count in resolved:
            write_line(f"{old_count} -> {new_count}\t{check_name}\t{file_path}")
    else:
        write_line("resolved or improved diagnostics: none")

    if tool_failures:
        write_line("tool execution failures:")
        for file_path, rc in tool_failures:
            write_line(f"{rc}\t{file_path}")
    else:
        write_line("tool execution failures: none")
    write_line("::endgroup::")

    return 1 if new_findings or regressed or tool_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
