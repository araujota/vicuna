#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policy_registry import resolve_artifact_path
from ppo_policy import load_ppo_policy_artifact, predict_ppo_action_with_confidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registry-backed PPO policy artifact")
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
    artifact = load_ppo_policy_artifact(artifact_path)
    payload = json.loads(sys.stdin.read() or "{}")
    prediction = predict_ppo_action_with_confidence(
        artifact=artifact,
        observation=payload.get("observation", {}),
        action_mask=payload.get("action_mask", {}),
    )
    print(
        json.dumps(
            {
                "policy_version": artifact["policy_version"],
                "artifact_kind": "ppo_policy",
                "action": prediction["action"],
                "confidence": prediction["confidence"],
                "rollout": prediction["rollout"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
