#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvec_generator import infer_cvec_from_artifact, load_cvec_generator_artifact
from policy_registry import resolve_artifact_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registry-backed cvec generator artifact")
    parser.add_argument("--artifact")
    parser.add_argument("--registry-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--alias")
    parser.add_argument("--version", type=int)
    return parser


def resolve_input_artifact(args: argparse.Namespace) -> Path:
    if args.artifact:
        return Path(args.artifact)
    if not args.registry_dir or not args.model_name:
        raise ValueError("--artifact or (--registry-dir and --model-name) is required")
    return resolve_artifact_path(
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        alias=args.alias,
        version=args.version,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    artifact_path = resolve_input_artifact(args)
    artifact = load_cvec_generator_artifact(artifact_path)
    payload = json.loads(sys.stdin.read() or "{}")
    prediction = infer_cvec_from_artifact(
        artifact=artifact,
        observation=payload.get("observation", {}),
    )
    print(json.dumps(prediction, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
