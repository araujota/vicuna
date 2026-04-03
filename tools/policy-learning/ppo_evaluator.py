#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ppo_policy import load_ppo_policy_artifact, predict_ppo_action_with_confidence
from ppo_training_contract import load_ppo_training_records


PPO_EVALUATION_REPORT_SCHEMA_VERSION = "vicuna.ppo_policy_evaluation.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_ppo_policy(
    dataset_dir: Path,
    *,
    artifact_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    artifact = load_ppo_policy_artifact(artifact_path)
    training_manifest, records = load_ppo_training_records(dataset_dir)
    if not records:
        raise ValueError("no PPO training records found")
    exact_matches = 0
    reward_on_match = []
    invalid_action_count = 0
    for record in records:
        prediction = predict_ppo_action_with_confidence(
            artifact,
            record["observation"],
            record["action_mask"],
        )
        action = prediction["action"]
        target = record["targets"]
        all_match = True
        for key, value in target.items():
            if action.get(key) != value:
                all_match = False
                break
        if all_match:
            exact_matches += 1
            reward_on_match.append(float(record["reward_total"]))
        if action["tool_parallelism_cap"] > int(record["action_mask"]["max_tool_parallelism_cap"]):
            invalid_action_count += 1
    report = {
        "schema_version": PPO_EVALUATION_REPORT_SCHEMA_VERSION,
        "artifact_kind": "ppo_policy",
        "policy_version": artifact["policy_version"],
        "dataset_id": training_manifest["dataset_id"],
        "record_count": len(records),
        "exact_match_rate": exact_matches / max(len(records), 1),
        "invalid_action_rate": invalid_action_count / max(len(records), 1),
        "reward_total_mean_on_match": sum(reward_on_match) / len(reward_on_match) if reward_on_match else None,
        "artifact_path": str(artifact_path),
    }
    if report_path is not None:
        _write_json(report_path, report)
        report["report_path"] = str(report_path)
    return report
