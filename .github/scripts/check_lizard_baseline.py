#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOURCE_PATTERNS = [
    "*.c",
    "*.cc",
    "*.cpp",
    "*.cxx",
    "*.h",
    "*.hh",
    "*.hpp",
    "*.py",
    "*.js",
    "*.mjs",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_stream(text: str, *, stream: object = sys.stdout) -> None:
    if not text:
        return
    suffix = "" if text.endswith("\n") else "\n"
    stream.write(f"{text}{suffix}")


def write_line(text: str = "", *, stream: object = sys.stdout) -> None:
    stream.write(f"{text}\n")


def tracked_files(root: Path) -> list[str]:
    cmd = ["git", "ls-files", *SOURCE_PATTERNS]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    files: list[str] = []
    for raw_line in proc.stdout.splitlines():
        path = raw_line.strip()
        if not path or path.startswith("vendor/") or path.startswith("build"):
            continue
        files.append(path)
    return files


def parse_warning_lines(stdout: str) -> list[str]:
    warnings: list[str] = []
    in_warning_section = False

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("!!!! Warnings"):
            in_warning_section = True
            continue

        if not in_warning_section:
            continue

        if stripped.startswith("Total nloc"):
            break

        if (
            not stripped
            or stripped.startswith("NLOC")
            or set(stripped) <= {"=", "-"}
        ):
            continue

        if raw_line[:1].isspace() and stripped[:1].isdigit():
            warnings.append(" ".join(stripped.split()))

    return warnings


def run_lizard(root: Path, files: list[str], complexity: int, length: int) -> tuple[int, str, str]:
    cmd = ["lizard", "-C", str(complexity), "-L", str(length), *files]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_baseline(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_baseline(path: Path, warnings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{line}\n" for line in warnings)
    path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lizard across the repository and compare against a checked-in baseline.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline file, relative to the repo root unless absolute.")
    parser.add_argument("--complexity", type=int, default=100, help="Cyclomatic complexity threshold.")
    parser.add_argument("--length", type=int, default=500, help="Function length threshold.")
    parser.add_argument("--write-baseline", action="store_true", help="Overwrite the baseline file with the current warning set.")
    args = parser.parse_args()

    root = repo_root()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    files = tracked_files(root)
    if not files:
        write_line("No files found for lizard analysis.")
        return 0

    rc, stdout, stderr = run_lizard(root, files, args.complexity, args.length)
    write_stream(stdout)
    write_stream(stderr, stream=sys.stderr)

    warnings = parse_warning_lines(stdout)

    if args.write_baseline:
        write_baseline(baseline_path, warnings)
        write_line(f"Wrote {len(warnings)} lizard baseline entries to {baseline_path.relative_to(root)}")
        return 0

    if rc not in (0, 1):
        write_line(f"lizard exited with unexpected status {rc}", stream=sys.stderr)
        return rc

    baseline = read_baseline(baseline_path)
    baseline_set = set(baseline)
    warning_set = set(warnings)

    unexpected = sorted(warning_set - baseline_set)
    resolved = sorted(baseline_set - warning_set)

    write_line("::group::lizard baseline comparison")
    write_line(f"baseline warnings: {len(baseline)}")
    write_line(f"current warnings: {len(warnings)}")

    if unexpected:
        write_line("new or changed warnings:")
        for line in unexpected:
            write_line(line)
    else:
        write_line("new or changed warnings: none")

    if resolved:
        write_line("resolved baseline warnings:")
        for line in resolved:
            write_line(line)
    else:
        write_line("resolved baseline warnings: none")
    write_line("::endgroup::")

    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
