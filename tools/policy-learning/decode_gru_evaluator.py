#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from decode_gru_contract import ACTION_BOOL_FIELDS, ACTION_NUMERIC_FIELDS, ACTION_PROFILE_FIELDS, load_decode_gru_training_records
from decode_gru_model import infer_decode_controller_action, load_decode_controller_artifact


DECODE_GRU_EVALUATION_REPORT_SCHEMA_VERSION = "vicuna.decode_controller_evaluation.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric_close(lhs: float, rhs: float) -> bool:
    return abs(float(lhs) - float(rhs)) <= 0.15


def _action_valid(prediction: dict[str, Any], mask: dict[str, Any]) -> bool:
    booleans = prediction["boolean"]
    numerics = prediction["numeric"]
    if booleans["sampling.enabled"] and not mask["allow_sampling"]:
        return False
    if booleans["repetition.enabled"] and not mask["allow_repetition"]:
        return False
    if booleans["structure.enabled"] and not mask["allow_structure"]:
        return False
    if booleans["branch.enabled"] and not mask["allow_branch"]:
        return False
    if booleans["steering.enabled"] and not mask["allow_steering"]:
        return False
    if numerics["branch.branch_sample_count"] > float(mask["max_branch_sample_count"]):
        return False
    return True


def evaluate_decode_gru_controller(
    dataset_dir: Path,
    *,
    artifact_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    artifact = load_decode_controller_artifact(artifact_path)
    training_manifest, records = load_decode_gru_training_records(dataset_dir)
    if not records:
        raise ValueError("no decode GRU training records found")

    teacher_bool_matches = 0
    teacher_bool_total = 0
    teacher_numeric_matches = 0
    teacher_numeric_total = 0
    teacher_profile_matches = 0
    teacher_profile_total = 0
    valid_predictions = 0
    outcome_scores = []
    step_count = 0

    for record in records:
        history: list[list[float]] = []
        for step in record["steps"]:
            history.append(step["input_features"])
            prediction = infer_decode_controller_action(
                artifact,
                history,
                action_mask=step["mask"],
            )
            step_count += 1
            if _action_valid(prediction, step["mask"]):
                valid_predictions += 1

            target = step["teacher_target"]
            bool_match_count = 0
            for field in ACTION_BOOL_FIELDS:
                teacher_bool_total += 1
                if bool(prediction["boolean"][field]) == bool(target["boolean"][field]):
                    teacher_bool_matches += 1
                    bool_match_count += 1
            numeric_match_count = 0
            numeric_count = 0
            for field in ACTION_NUMERIC_FIELDS:
                if float(target["numeric_masks"][field]) <= 0.0:
                    continue
                numeric_count += 1
                teacher_numeric_total += 1
                if _numeric_close(prediction["numeric"][field], target["numeric"][field]):
                    teacher_numeric_matches += 1
                    numeric_match_count += 1
            profile_match_count = 0
            profile_count = 0
            for field in ACTION_PROFILE_FIELDS:
                if float(target["numeric_masks"][field]) <= 0.0:
                    continue
                profile_count += 1
                teacher_profile_total += 1
                if str(prediction["profiles"][field]) == str(target["profiles"][field]):
                    teacher_profile_matches += 1
                    profile_match_count += 1

            next_outcome = step.get("next_outcome") or {}
            if next_outcome.get("available"):
                outcome_score = (
                    max(0.0, float(next_outcome.get("d_confidence", 0.0)))
                    + max(0.0, -float(next_outcome.get("d_mean_entropy", 0.0)))
                    + max(0.0, -float(next_outcome.get("d_stall", 0.0)))
                )
                match_fraction = (
                    bool_match_count + numeric_match_count + profile_match_count
                ) / max(
                    1,
                    len(ACTION_BOOL_FIELDS) + numeric_count + profile_count,
                )
                outcome_scores.append(outcome_score * match_fraction)

    report = {
        "schema_version": DECODE_GRU_EVALUATION_REPORT_SCHEMA_VERSION,
        "artifact_kind": "decode_controller",
        "controller_version": artifact["controller_version"],
        "dataset_id": training_manifest["dataset_id"],
        "record_count": len(records),
        "step_count": step_count,
        "valid_action_rate": valid_predictions / max(step_count, 1),
        "teacher_bool_match_rate": teacher_bool_matches / max(teacher_bool_total, 1),
        "teacher_numeric_match_rate": teacher_numeric_matches / max(teacher_numeric_total, 1),
        "teacher_profile_match_rate": teacher_profile_matches / max(teacher_profile_total, 1),
        "matched_outcome_proxy_mean": float(np.mean(outcome_scores)) if outcome_scores else None,
        "artifact_path": str(artifact_path),
    }
    if report_path is not None:
        _write_json(report_path, report)
        report["report_path"] = str(report_path)
    return report
