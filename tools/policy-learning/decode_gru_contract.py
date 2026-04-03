#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from policy_dataset import load_decode_trace_rows, load_manifest


DECODE_GRU_TRAINING_RECORD_SCHEMA_VERSION = "vicuna.decode_gru_training_record.v1"
DECODE_GRU_TRAINING_MANIFEST_SCHEMA_VERSION = "vicuna.decode_gru_training_manifest.v1"
DECODE_GRU_TRAINING_CONTRACT_VERSION = "vicuna.decode_level_gru.v1"

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
RUNTIME_NUMERIC_FIELDS = [
    "mean_entropy",
    "max_entropy",
    "mean_margin",
    "sampled_prob",
    "stop_prob",
    "repeat_hit_rate",
    "route_entropy_mean",
    "route_entropy_max",
    "route_top1_weight_mean",
    "route_top1_weight_max",
    "attention_entropy_mean",
    "attention_entropy_max",
    "attention_top1_mass_mean",
    "attention_top1_mass_max",
    "agreement_score",
    "consistency_entropy",
    "branch_disagreement",
    "verifier_disagreement",
    "graph_value_mean_abs",
    "graph_value_rms",
    "graph_value_max_abs",
    "dominant_expert_fraction_top1",
    "dominant_expert_fraction_mass",
    "timing_decode_ms",
    "timing_sample_ms",
    "timing_delta_ms",
    "memory_budget_ratio",
    "attention_budget_ratio",
    "recurrent_budget_ratio",
    "candidate_count",
    "attention_pos_min",
    "attention_pos_max",
    "recurrent_pos_min",
    "recurrent_pos_max",
    "expert_count",
    "experts_selected",
    "dominant_expert_count",
    "comparison_count",
    "semantic_group_count",
    "status_code",
]
RUNTIME_BOOL_FIELDS = [
    "runtime_failure",
    "verifier_active",
    "grammar_active",
    "logit_bias_active",
    "backend_sampler",
    "optimized",
    "prompt_section_changed",
]
BUNDLE_FIELDS = [
    "uncertainty_regulation",
    "anti_repetition_recovery",
    "structural_validity",
    "verification_pressure",
    "commit_efficiency",
    "steering_pressure",
]
DECODE_POLICY_NUMERIC_FIELDS = [
    "base_temperature",
    "base_top_k",
    "base_top_p",
    "base_min_p",
    "control_limits.min_temperature",
    "control_limits.max_temperature",
    "control_limits.min_top_p",
    "control_limits.max_top_p",
    "control_limits.min_min_p",
    "control_limits.max_min_p",
    "control_limits.min_typical_p",
    "control_limits.max_typical_p",
    "control_limits.max_top_n_sigma",
    "control_limits.min_repeat_penalty",
    "control_limits.max_repeat_penalty",
    "control_limits.max_frequency_penalty",
    "control_limits.max_presence_penalty",
    "control_limits.max_dry_multiplier",
    "control_limits.max_penalty_last_n",
    "control_limits.max_dry_allowed_length",
    "control_limits.max_dry_penalty_last_n",
]
MASK_FIELDS = [
    "allow_sampling",
    "allow_repetition",
    "allow_structure",
    "allow_branch",
    "allow_steering",
    "max_branch_sample_count",
]
PREVIOUS_ACTION_BOOL_FIELDS = [
    ("valid", []),
    ("sampling.enabled", []),
    ("sampling.has_temperature", []),
    ("sampling.has_top_k", []),
    ("sampling.has_top_p", []),
    ("sampling.has_min_p", []),
    ("sampling.has_typical_p", []),
    ("sampling.has_top_n_sigma", []),
    ("repetition.enabled", []),
    ("repetition.has_repeat_penalty", []),
    ("repetition.has_frequency_penalty", []),
    ("repetition.has_presence_penalty", []),
    ("repetition.has_penalty_last_n", []),
    ("structure.enabled", []),
    ("structure.clear_grammar", []),
    ("structure.clear_logit_bias", []),
    ("branch.enabled", []),
    ("branch.checkpoint_now", []),
    ("branch.request_verify", []),
    ("steering.enabled", []),
    ("steering.clear_cvec", []),
]
PREVIOUS_ACTION_NUMERIC_FIELDS = [
    "sampling.temperature",
    "sampling.top_k",
    "sampling.top_p",
    "sampling.min_p",
    "sampling.typical_p",
    "sampling.top_n_sigma",
    "sampling.min_keep",
    "repetition.repeat_penalty",
    "repetition.frequency_penalty",
    "repetition.presence_penalty",
    "repetition.penalty_last_n",
    "branch.checkpoint_slot",
    "branch.restore_slot",
    "branch.branch_sample_count",
]
ACTION_BOOL_FIELDS = [
    "valid",
    "sampling.enabled",
    "sampling.has_temperature",
    "sampling.has_top_k",
    "sampling.has_top_p",
    "sampling.has_min_p",
    "sampling.has_typical_p",
    "sampling.has_top_n_sigma",
    "repetition.enabled",
    "repetition.has_repeat_penalty",
    "repetition.has_frequency_penalty",
    "repetition.has_presence_penalty",
    "repetition.has_penalty_last_n",
    "structure.enabled",
    "structure.clear_grammar",
    "structure.clear_logit_bias",
    "branch.enabled",
    "branch.checkpoint_now",
    "branch.request_verify",
    "steering.enabled",
    "steering.clear_cvec",
]
ACTION_NUMERIC_FIELDS = [
    "sampling.temperature",
    "sampling.top_k",
    "sampling.top_p",
    "sampling.min_p",
    "sampling.typical_p",
    "sampling.top_n_sigma",
    "sampling.min_keep",
    "repetition.repeat_penalty",
    "repetition.frequency_penalty",
    "repetition.presence_penalty",
    "repetition.penalty_last_n",
    "branch.checkpoint_slot",
    "branch.restore_slot",
    "branch.branch_sample_count",
]
ACTION_PROFILE_FIELDS = [
    "structure.grammar_profile_id",
    "structure.logit_bias_profile_id",
    "steering.cvec_profile_id",
]
ACTION_NUMERIC_RANGES = {
    "sampling.temperature": [0.0, 2.0],
    "sampling.top_k": [0.0, 200.0],
    "sampling.top_p": [0.0, 1.0],
    "sampling.min_p": [0.0, 1.0],
    "sampling.typical_p": [0.0, 1.0],
    "sampling.top_n_sigma": [0.0, 8.0],
    "sampling.min_keep": [0.0, 16.0],
    "repetition.repeat_penalty": [0.0, 2.0],
    "repetition.frequency_penalty": [0.0, 2.0],
    "repetition.presence_penalty": [0.0, 2.0],
    "repetition.penalty_last_n": [0.0, 512.0],
    "branch.checkpoint_slot": [0.0, 8.0],
    "branch.restore_slot": [0.0, 8.0],
    "branch.branch_sample_count": [0.0, 8.0],
}
ACTION_NUMERIC_REQUIREMENTS = {
    "sampling.temperature": "sampling.has_temperature",
    "sampling.top_k": "sampling.has_top_k",
    "sampling.top_p": "sampling.has_top_p",
    "sampling.min_p": "sampling.has_min_p",
    "sampling.typical_p": "sampling.has_typical_p",
    "sampling.top_n_sigma": "sampling.has_top_n_sigma",
    "sampling.min_keep": "sampling.enabled",
    "repetition.repeat_penalty": "repetition.has_repeat_penalty",
    "repetition.frequency_penalty": "repetition.has_frequency_penalty",
    "repetition.presence_penalty": "repetition.has_presence_penalty",
    "repetition.penalty_last_n": "repetition.has_penalty_last_n",
    "branch.checkpoint_slot": "branch.enabled",
    "branch.restore_slot": "branch.enabled",
    "branch.branch_sample_count": "branch.enabled",
}
ACTION_PROFILE_REQUIREMENTS = {
    "structure.grammar_profile_id": "structure.enabled",
    "structure.logit_bias_profile_id": "structure.enabled",
    "steering.cvec_profile_id": "steering.enabled",
}


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


def _path_get(payload: dict[str, Any], dotted: str) -> Any:
    cursor: Any = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
        if cursor is None:
            return None
    return cursor


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_to_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def flatten_decode_step_input(step: dict[str, Any]) -> list[float]:
    features: list[float] = []
    moment = step.get("moment") or {}
    vad = step.get("vad") or {}
    runtime_signals = step.get("runtime_signals") or {}
    bundles = step.get("bundles") or {}
    decode_policy = step.get("decode_policy_config") or {}
    previous_action = step.get("previous_executed_action") or {}

    for field in MOMENT_FIELD_ORDER:
        features.append(_as_float(moment.get(field)))
    for field in VAD_FIELD_ORDER:
        features.append(_as_float(vad.get(field)))
    for field in RUNTIME_NUMERIC_FIELDS:
        features.append(_as_float(runtime_signals.get(field)))
    for field in RUNTIME_BOOL_FIELDS:
        features.append(_bool_to_float(runtime_signals.get(field)))
    for field in BUNDLE_FIELDS:
        features.append(_as_float(bundles.get(field)))

    for field in DECODE_POLICY_NUMERIC_FIELDS:
        features.append(_as_float(_path_get(decode_policy, field)))

    mask = {
        "allow_sampling": bool((((decode_policy.get("action_mask") or step.get("mask") or {}).get("allow_sampling", True)))),
        "allow_repetition": bool((((decode_policy.get("action_mask") or step.get("mask") or {}).get("allow_repetition", True)))),
        "allow_structure": bool((((decode_policy.get("action_mask") or step.get("mask") or {}).get("allow_structure", True)))),
        "allow_branch": bool((((decode_policy.get("action_mask") or step.get("mask") or {}).get("allow_branch", True)))),
        "allow_steering": bool((((decode_policy.get("action_mask") or step.get("mask") or {}).get("allow_steering", True)))),
        "max_branch_sample_count": int((((decode_policy.get("action_mask") or step.get("mask") or {}).get("max_branch_sample_count", 0)))),
    }
    for field in MASK_FIELDS:
        features.append(_as_float(mask.get(field)))

    features.append(_bool_to_float(step.get("previous_executed_action_available")))
    for field in PREVIOUS_ACTION_BOOL_FIELDS:
        features.append(_bool_to_float(_path_get(previous_action, field[0])))
    for field in PREVIOUS_ACTION_NUMERIC_FIELDS:
        features.append(_as_float(_path_get(previous_action, field)))
    return features


def _normalize_mask(step: dict[str, Any]) -> dict[str, Any]:
    decode_policy = step.get("decode_policy_config") or {}
    mask = (decode_policy.get("action_mask") or step.get("mask") or {})
    return {
        "allow_sampling": bool(mask.get("allow_sampling", True)),
        "allow_repetition": bool(mask.get("allow_repetition", True)),
        "allow_structure": bool(mask.get("allow_structure", True)),
        "allow_branch": bool(mask.get("allow_branch", True)),
        "allow_steering": bool(mask.get("allow_steering", True)),
        "max_branch_sample_count": int(mask.get("max_branch_sample_count", 0)),
    }


def _collect_profile_vocab(rows: list[dict[str, Any]], dotted: str) -> list[str]:
    values = {""}
    for row in rows:
        trace = row["decode_trace"]
        for step in trace.get("steps", []):
            teacher_action = step.get("teacher_action") or {}
            value = _path_get(teacher_action, dotted)
            if isinstance(value, str):
                values.add(value)
    return sorted(values)


def _action_target_masks(teacher_action: dict[str, Any]) -> dict[str, float]:
    masks: dict[str, float] = {}
    for numeric_field in ACTION_NUMERIC_FIELDS:
        requirement = ACTION_NUMERIC_REQUIREMENTS[numeric_field]
        masks[numeric_field] = _bool_to_float(_path_get(teacher_action, requirement))
    for profile_field in ACTION_PROFILE_FIELDS:
        requirement = ACTION_PROFILE_REQUIREMENTS[profile_field]
        masks[profile_field] = _bool_to_float(_path_get(teacher_action, requirement))
    return masks


def encode_teacher_action(
    teacher_action: dict[str, Any],
    *,
    grammar_profile_vocab: list[str],
    logit_bias_profile_vocab: list[str],
    cvec_profile_vocab: list[str],
) -> dict[str, Any]:
    bool_targets = {field: bool(_path_get(teacher_action, field)) for field in ACTION_BOOL_FIELDS}
    numeric_targets = {field: _as_float(_path_get(teacher_action, field)) for field in ACTION_NUMERIC_FIELDS}
    profile_targets = {
        "structure.grammar_profile_id": str(_path_get(teacher_action, "structure.grammar_profile_id") or ""),
        "structure.logit_bias_profile_id": str(_path_get(teacher_action, "structure.logit_bias_profile_id") or ""),
        "steering.cvec_profile_id": str(_path_get(teacher_action, "steering.cvec_profile_id") or ""),
    }
    for field, vocab in [
        ("structure.grammar_profile_id", grammar_profile_vocab),
        ("structure.logit_bias_profile_id", logit_bias_profile_vocab),
        ("steering.cvec_profile_id", cvec_profile_vocab),
    ]:
        if profile_targets[field] not in vocab:
            profile_targets[field] = ""
    return {
        "boolean": bool_targets,
        "numeric": numeric_targets,
        "numeric_masks": _action_target_masks(teacher_action),
        "profiles": profile_targets,
    }


def _step_outcome_weight(step: dict[str, Any]) -> float:
    next_outcome = step.get("next_outcome") or {}
    if not next_outcome.get("available"):
        return 1.0
    confidence_gain = max(0.0, _as_float(next_outcome.get("d_confidence")))
    entropy_drop = max(0.0, -_as_float(next_outcome.get("d_mean_entropy")))
    stall_drop = max(0.0, -_as_float(next_outcome.get("d_stall")))
    return min(3.0, 1.0 + confidence_gain + entropy_drop + stall_drop)


def build_decode_gru_training_corpus(
    dataset_dir: Path,
    *,
    sequence_length: int = 32,
    stride: int = 16,
) -> dict[str, Any]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    rows = load_decode_trace_rows(dataset_dir)
    grammar_profile_vocab = _collect_profile_vocab(rows, "structure.grammar_profile_id")
    logit_bias_profile_vocab = _collect_profile_vocab(rows, "structure.logit_bias_profile_id")
    cvec_profile_vocab = _collect_profile_vocab(rows, "steering.cvec_profile_id")

    records: list[dict[str, Any]] = []
    for row in rows:
        trace = row["decode_trace"]
        steps = trace.get("steps", [])
        for start in range(0, max(len(steps), 1), stride):
            window = steps[start : start + sequence_length]
            if not window:
                continue
            encoded_steps = []
            for step in window:
                encoded_steps.append(
                    {
                        "step_index": int(step.get("step_index", -1)),
                        "output_index": int(step.get("output_index", -1)),
                        "decode_policy_config": step.get("decode_policy_config") or {},
                        "input_features": flatten_decode_step_input(step),
                        "mask": _normalize_mask(step),
                        "teacher_target": encode_teacher_action(
                            step.get("teacher_action") or {},
                            grammar_profile_vocab=grammar_profile_vocab,
                            logit_bias_profile_vocab=logit_bias_profile_vocab,
                            cvec_profile_vocab=cvec_profile_vocab,
                        ),
                        "executed_action": step.get("executed_action") or {},
                        "candidate_metadata": step.get("candidate_metadata") or {},
                        "next_outcome": step.get("next_outcome") or {},
                        "weight": _step_outcome_weight(step),
                    }
                )
            records.append(
                {
                    "schema_version": DECODE_GRU_TRAINING_RECORD_SCHEMA_VERSION,
                    "contract_version": DECODE_GRU_TRAINING_CONTRACT_VERSION,
                    "dataset_id": manifest["dataset_id"],
                    "trace_export_key": row["export_key"],
                    "request_id": trace.get("request_id"),
                    "controller_mode": trace.get("controller_mode"),
                    "candidate_policy_version": trace.get("candidate_policy_version"),
                    "sequence_index": start // stride,
                    "steps": encoded_steps,
                }
            )

    training_dir = dataset_dir / manifest["training_dir"]
    training_records_path = training_dir / "decode_gru_training_records_v1.jsonl"
    training_manifest_path = training_dir / "decode_gru_training_manifest_v1.json"
    _write_jsonl(training_records_path, records)
    _write_json(
        training_manifest_path,
        {
            "schema_version": DECODE_GRU_TRAINING_MANIFEST_SCHEMA_VERSION,
            "contract_version": DECODE_GRU_TRAINING_CONTRACT_VERSION,
            "dataset_id": manifest["dataset_id"],
            "record_count": len(records),
            "input_dimension": len(records[0]["steps"][0]["input_features"]) if records else 0,
            "sequence_length": int(sequence_length),
            "stride": int(stride),
            "grammar_profile_vocab": grammar_profile_vocab,
            "logit_bias_profile_vocab": logit_bias_profile_vocab,
            "cvec_profile_vocab": cvec_profile_vocab,
            "action_boolean_fields": ACTION_BOOL_FIELDS,
            "action_numeric_fields": ACTION_NUMERIC_FIELDS,
            "action_numeric_ranges": ACTION_NUMERIC_RANGES,
            "action_profile_fields": ACTION_PROFILE_FIELDS,
            "training_records_path": str(training_records_path.relative_to(training_dir)),
        },
    )
    return {
        "dataset_id": manifest["dataset_id"],
        "record_count": len(records),
        "training_manifest_path": str(training_manifest_path),
        "training_records_path": str(training_records_path),
    }


def load_decode_gru_training_records(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    training_dir = dataset_dir / manifest["training_dir"]
    training_manifest_path = training_dir / "decode_gru_training_manifest_v1.json"
    training_records_path = training_dir / "decode_gru_training_records_v1.jsonl"
    if not training_manifest_path.exists() or not training_records_path.exists():
        raise FileNotFoundError(
            f"decode GRU training contract not found in {training_dir}; run build-decode-training-set first"
        )
    manifest_payload = _read_json(training_manifest_path)
    records: list[dict[str, Any]] = []
    with training_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                records.append(json.loads(payload))
    return manifest_payload, records
