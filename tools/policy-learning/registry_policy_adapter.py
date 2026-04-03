#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policy_registry import resolve_artifact_path
from policy_trainer import load_policy_artifact, predict_action_with_confidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registry-backed policy artifact")
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
    raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    schema_version = str(raw_artifact.get("schema_version", ""))
    payload = json.loads(sys.stdin.read() or "{}")
    if schema_version == "vicuna.policy_artifact.v1":
        artifact = load_policy_artifact(artifact_path)
        prediction = predict_action_with_confidence(
            artifact=artifact,
            observation=payload.get("observation", {}),
            action_mask=payload.get("action_mask", {}),
        )
        artifact_kind = "policy"
    elif schema_version == "vicuna.ppo_policy_artifact.v1":
        from ppo_policy import load_ppo_policy_artifact, predict_ppo_action_with_confidence

        artifact = load_ppo_policy_artifact(artifact_path)
        prediction = predict_ppo_action_with_confidence(
            artifact=artifact,
            observation=payload.get("observation", {}),
            action_mask=payload.get("action_mask", {}),
        )
        artifact_kind = "ppo_policy"
    else:
        raise ValueError(f"unsupported artifact schema_version {schema_version}")
    print(
        json.dumps(
            {
                "artifact_kind": artifact_kind,
                "policy_version": artifact["policy_version"],
                "action": prediction["action"],
                "confidence": prediction["confidence"],
                "rollout": prediction.get("rollout"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
