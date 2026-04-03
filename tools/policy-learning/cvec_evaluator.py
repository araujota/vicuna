#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "cvec generator tooling requires numpy; use a numpy-enabled Python "
        "interpreter such as python3.10 or install numpy into the active environment"
    ) from exc

from cvec_generator import infer_cvec_from_artifact, load_cvec_generator_artifact
from cvec_generator_contract import load_cvec_training_records


CVEC_EVALUATION_REPORT_SCHEMA_VERSION = "vicuna.cvec_generator_evaluation.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_cvec_generator(
    dataset_dir: Path,
    *,
    artifact_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    artifact = load_cvec_generator_artifact(artifact_path)
    training_manifest, records = load_cvec_training_records(dataset_dir)
    if not records:
        raise ValueError("no cvec training records found")

    predictions = []
    targets = []
    weights = []
    for record in records:
        observation = {
            "moment": {
                field: record["state_input"][idx]
                for idx, field in enumerate(
                    [
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
                )
            },
            "vad": {
                "valence": record["state_input"][14],
                "arousal": record["state_input"][15],
                "dominance": record["state_input"][16],
            },
        }
        prediction = infer_cvec_from_artifact(artifact, observation)
        predictions.append(prediction["vector"])
        targets.append(record["target"]["vector"])
        weights.append(float(record.get("weight", 1.0)))

    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = sample_weights / np.maximum(np.sum(sample_weights), 1e-12)
    diff = pred - target
    weighted_mse = float(np.sum(normalized_weights[:, None] * np.square(diff)))
    dot = np.sum(pred * target, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = dot / np.maximum(pred_norm * target_norm, 1e-12)
    weighted_cosine = float(np.sum(normalized_weights * cosine))

    report = {
        "schema_version": CVEC_EVALUATION_REPORT_SCHEMA_VERSION,
        "artifact_kind": "cvec_generator",
        "generator_version": artifact["generator_version"],
        "dataset_id": training_manifest["dataset_id"],
        "record_count": len(records),
        "weighted_mse": weighted_mse,
        "weighted_cosine": weighted_cosine,
        "mean_pred_norm": float(np.mean(pred_norm)),
        "mean_target_norm": float(np.mean(target_norm)),
        "artifact_path": str(artifact_path),
    }
    if report_path is not None:
        _write_json(report_path, report)
        report["report_path"] = str(report_path)
    return report
