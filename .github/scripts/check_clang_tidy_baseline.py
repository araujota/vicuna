#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_path(path: str, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
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
) -> tuple[int, Counter[tuple[str, str]]]:
    status = 0
    counts: Counter[tuple[str, str]] = Counter()

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
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        counts.update(parse_diagnostics(proc.stdout, root))
        counts.update(parse_diagnostics(proc.stderr, root))
        if proc.returncode not in (0, 1):
            return proc.returncode, counts
        if proc.returncode != 0:
            status = 1

    return status, counts


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
        counts[(file_path, check_name)] = int(count_str)
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

    rc, counts = run_clang_tidy(args.files, args.build_dir, args.header_filter, args.binary, root)
    if rc not in (0, 1):
        print(f"clang-tidy exited with unexpected status {rc}", file=sys.stderr)
        return rc

    if args.write_baseline:
        write_baseline(baseline_path, counts)
        print(f"Wrote {len(counts)} clang-tidy baseline entries to {baseline_path.relative_to(root)}")
        return 0

    baseline = read_baseline(baseline_path)
    new_findings, regressed, resolved = compare(counts, baseline)

    print("::group::clang-tidy baseline comparison")
    print(f"baseline diagnostics: {sum(baseline.values())}")
    print(f"current diagnostics: {sum(counts.values())}")

    if new_findings:
        print("new diagnostics:")
        for file_path, check_name, count in new_findings:
            print(f"{count}\t{check_name}\t{file_path}")
    else:
        print("new diagnostics: none")

    if regressed:
        print("regressed diagnostics:")
        for file_path, check_name, old_count, new_count in regressed:
            print(f"{old_count} -> {new_count}\t{check_name}\t{file_path}")
    else:
        print("regressed diagnostics: none")

    if resolved:
        print("resolved or improved diagnostics:")
        for file_path, check_name, old_count, new_count in resolved:
            print(f"{old_count} -> {new_count}\t{check_name}\t{file_path}")
    else:
        print("resolved or improved diagnostics: none")
    print("::endgroup::")

    return 1 if new_findings or regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
