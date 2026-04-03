#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policy_dataset import load_dataset_rows, load_manifest
from policy_training_contract import normalize_action_mask, normalize_action_targets
from ppo_policy import flatten_policy_observation


PPO_TRAINING_RECORD_SCHEMA_VERSION = "vicuna.ppo_training_record.v1"
PPO_TRAINING_MANIFEST_SCHEMA_VERSION = "vicuna.ppo_training_manifest.v1"
PPO_TRAINING_CONTRACT_VERSION = "vicuna.request_level_ppo.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_ppo_training_record(dataset_id: str, dataset_row: dict[str, Any]) -> dict[str, Any]:
    transition = dataset_row["transition"]
    observation = transition["observation"]
    normalized_mask = normalize_action_mask(transition["action_mask"])
    targets = normalize_action_targets(transition["executed_action"], normalized_mask)
    rollout = transition.get("policy_rollout") or {}
    return {
        "schema_version": PPO_TRAINING_RECORD_SCHEMA_VERSION,
        "contract_version": PPO_TRAINING_CONTRACT_VERSION,
        "dataset_id": dataset_id,
        "export_key": dataset_row["export_key"],
        "observation_features": flatten_policy_observation(observation),
        "observation": observation,
        "action_mask": normalized_mask,
        "targets": targets,
        "reward_total": float(transition.get("reward_total", 0.0)),
        "candidate_executed_live": bool(transition.get("candidate_executed_live", False)),
        "rollout": {
            "available": bool(rollout.get("available", False)),
            "artifact_kind": rollout.get("artifact_kind"),
            "policy_version": rollout.get("policy_version"),
            "selected_log_prob": rollout.get("selected_log_prob"),
            "value_estimate": rollout.get("value_estimate"),
            "entropy": rollout.get("entropy"),
        },
    }


def build_ppo_training_corpus(dataset_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    rows = load_dataset_rows(dataset_dir)
    records = [build_ppo_training_record(manifest["dataset_id"], row) for row in rows]
    training_dir = dataset_dir / manifest["training_dir"]
    training_records_path = training_dir / "ppo_training_records_v1.jsonl"
    training_manifest_path = training_dir / "ppo_training_manifest_v1.json"
    _write_jsonl(training_records_path, records)
    _write_json(
        training_manifest_path,
        {
            "schema_version": PPO_TRAINING_MANIFEST_SCHEMA_VERSION,
            "contract_version": PPO_TRAINING_CONTRACT_VERSION,
            "dataset_id": manifest["dataset_id"],
            "record_count": len(records),
            "feature_dimension": len(records[0]["observation_features"]) if records else 0,
            "reward_model_version": manifest.get("source_reward_model_version"),
            "training_records_path": str(training_records_path.relative_to(training_dir)),
        },
    )
    return {
        "dataset_id": manifest["dataset_id"],
        "record_count": len(records),
        "training_manifest_path": str(training_manifest_path),
        "training_records_path": str(training_records_path),
    }


def load_ppo_training_records(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    training_dir = dataset_dir / manifest["training_dir"]
    training_manifest_path = training_dir / "ppo_training_manifest_v1.json"
    training_records_path = training_dir / "ppo_training_records_v1.jsonl"
    if not training_manifest_path.exists() or not training_records_path.exists():
        raise FileNotFoundError(
            f"ppo training contract not found in {training_dir}; run build-ppo-training-set first"
        )
    manifest_payload = _read_json(training_manifest_path)
    rows = []
    with training_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
    return manifest_payload, rows
