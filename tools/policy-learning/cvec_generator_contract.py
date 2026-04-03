#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policy_dataset import load_dataset_rows, load_manifest


CVEC_TRAINING_RECORD_SCHEMA_VERSION = "vicuna.cvec_training_record.v1"
CVEC_TRAINING_MANIFEST_SCHEMA_VERSION = "vicuna.cvec_training_manifest.v1"
CVEC_TRAINING_CONTRACT_VERSION = "vicuna.emvad_to_cvec.v1"

MOMENT_FIELD_ORDER = [
    "confidence",
    "curiosity",
    "frustration",
    "satisfaction",
    "momentum",
    "caution",
    "stall",
    "epistemic_pressure",
    "planning_clarity",
    "user_alignment",
    "semantic_novelty",
    "runtime_trust",
    "runtime_failure_pressure",
    "contradiction_pressure",
]
VAD_FIELD_ORDER = ["valence", "arousal", "dominance"]


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
    return rows


def flatten_state_input(observation: dict[str, Any]) -> list[float]:
    moment = observation.get("moment") or {}
    vad = observation.get("vad") or {}
    values = []
    for field in MOMENT_FIELD_ORDER:
        values.append(float(moment.get(field, 0.0)))
    for field in VAD_FIELD_ORDER:
        values.append(float(vad.get(field, 0.0)))
    return values


def _reward_weight(reward_total: Any) -> float:
    try:
        reward = float(reward_total)
    except (TypeError, ValueError):
        return 1.0
    return min(4.0, max(0.25, 1.0 + reward))


def _normalize_vector(raw: Any, *, expected_dim: int) -> list[float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("target_control_vector must be a list")
    if len(raw) != expected_dim:
        raise ValueError(
            f"target_control_vector length {len(raw)} does not match expected_dim {expected_dim}"
        )
    return [float(value) for value in raw]


def _extract_profile_id(transition: dict[str, Any]) -> str | None:
    for key in [
        "target_control_profile_id",
        "control_profile_id",
        "cvec_profile_id",
    ]:
        value = transition.get(key)
        if isinstance(value, str) and value:
            return value

    executed_action = transition.get("executed_action") or {}
    for key in ["target_control_profile_id", "control_profile_id", "cvec_profile_id"]:
        value = executed_action.get(key)
        if isinstance(value, str) and value:
            return value

    steering = executed_action.get("steering") or {}
    for key in ["profile_id", "cvec_profile_id"]:
        value = steering.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _load_vector_library(path: Path | None, *, expected_dim: int) -> dict[str, list[float]]:
    if path is None:
        return {}
    payload = _read_json(path)
    library = payload.get("vectors", payload)
    if not isinstance(library, dict):
        raise ValueError("vector library must be a dict or contain a 'vectors' dict")
    normalized: dict[str, list[float]] = {}
    for key, raw_vector in library.items():
        normalized[str(key)] = _normalize_vector(raw_vector, expected_dim=expected_dim) or []
    return normalized


def build_cvec_training_record(
    dataset_id: str,
    dataset_row: dict[str, Any],
    *,
    expected_dim: int,
    vector_library: dict[str, list[float]] | None = None,
) -> dict[str, Any] | None:
    transition = dataset_row["transition"]
    observation = transition.get("observation") or {}
    if not observation.get("moment") or not observation.get("vad"):
        return None

    target_vector = _normalize_vector(
        transition.get("target_control_vector"),
        expected_dim=expected_dim,
    )
    profile_id = _extract_profile_id(transition)
    if target_vector is None and profile_id and vector_library and profile_id in vector_library:
        target_vector = list(vector_library[profile_id])
    if target_vector is None:
        return None

    reward_total = float(transition.get("reward_total", 0.0))
    return {
        "schema_version": CVEC_TRAINING_RECORD_SCHEMA_VERSION,
        "contract_version": CVEC_TRAINING_CONTRACT_VERSION,
        "dataset_id": dataset_id,
        "export_key": dataset_row["export_key"],
        "reward_total": reward_total,
        "reward_model_version": (transition.get("reward_model") or {}).get("model_version"),
        "state_input": flatten_state_input(observation),
        "target": {
            "vector": target_vector,
            "profile_id": profile_id,
        },
        "weight": _reward_weight(reward_total),
        "metadata": {
            "request_id": transition.get("request_id"),
            "transition_id": transition.get("transition_id"),
        },
    }


def build_cvec_training_corpus(
    dataset_dir: Path,
    *,
    target_embedding_dim: int,
    vector_library_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    vector_library = _load_vector_library(
        vector_library_path,
        expected_dim=target_embedding_dim,
    )
    rows = load_dataset_rows(dataset_dir)
    records: list[dict[str, Any]] = []
    for row in rows:
        record = build_cvec_training_record(
            manifest["dataset_id"],
            row,
            expected_dim=target_embedding_dim,
            vector_library=vector_library,
        )
        if record is not None:
            records.append(record)

    training_dir = dataset_dir / manifest["training_dir"]
    training_records_path = training_dir / "cvec_training_records_v1.jsonl"
    training_manifest_path = training_dir / "cvec_training_manifest_v1.json"
    _write_jsonl(training_records_path, records)
    training_manifest = {
        "schema_version": CVEC_TRAINING_MANIFEST_SCHEMA_VERSION,
        "contract_version": CVEC_TRAINING_CONTRACT_VERSION,
        "dataset_id": manifest["dataset_id"],
        "record_count": len(records),
        "input_dimension": len(MOMENT_FIELD_ORDER) + len(VAD_FIELD_ORDER),
        "target_embedding_dim": int(target_embedding_dim),
        "vector_library_path": str(vector_library_path) if vector_library_path else None,
        "reward_model_version": manifest.get("source_reward_model_version"),
        "training_records_path": str(training_records_path.relative_to(training_dir)),
    }
    _write_json(training_manifest_path, training_manifest)
    return {
        "dataset_id": manifest["dataset_id"],
        "record_count": len(records),
        "training_manifest_path": str(training_manifest_path),
        "training_records_path": str(training_records_path),
    }


def load_cvec_training_records(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    training_dir = dataset_dir / manifest["training_dir"]
    training_manifest_path = training_dir / "cvec_training_manifest_v1.json"
    training_records_path = training_dir / "cvec_training_records_v1.jsonl"
    if not training_manifest_path.exists() or not training_records_path.exists():
        raise FileNotFoundError(
            f"cvec training contract not found in {training_dir}; run build-cvec-training-set first"
        )
    return _read_json(training_manifest_path), _read_jsonl(training_records_path)
