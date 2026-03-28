#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from policy_dataset import load_manifest
from policy_training_contract import (
    BOOLEAN_HEADS,
    PREFIX_PROFILE_VOCAB,
    REASONING_DEPTH_VOCAB,
    REPETITION_PROFILE_VOCAB,
    SAMPLING_PROFILE_VOCAB,
    SELECTED_MODE_VOCAB,
    STOP_PROFILE_VOCAB,
    THINKING_MODE_VOCAB,
    TOKEN_BUDGET_BUCKET_VOCAB,
    TOOL_CHOICE_PROFILE_VOCAB,
    normalize_action_mask,
)


POLICY_ARTIFACT_SCHEMA_VERSION = "vicuna.policy_artifact.v1"
TRAINING_RUN_SCHEMA_VERSION = "vicuna.policy_training_run.v1"
TRAINER_ALGORITHM = "vicuna.masked_bc_tabular.v1"

FEATURE_SCHEMA = [
    "mode_label",
    "bridge_scoped",
    "cognitive_replay",
    "heuristic_matched",
    "heuristic_id",
    "available_tool_count_bucket",
    "parallel_tool_calls_requested",
    "input_message_count_bucket",
    "ongoing_task_due_bucket",
    "vad_valence_bucket",
    "vad_arousal_bucket",
    "vad_dominance_bucket",
]


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _bucket_signed(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if number <= -0.25:
        return "negative"
    if number >= 0.25:
        return "positive"
    return "neutral"


def _bucket_unit_interval(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if number < 0.33:
        return "low"
    if number < 0.66:
        return "medium"
    return "high"


def _bucket_tool_count(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "none"
    if count == 1:
        return "one"
    return "many"


def _bucket_input_count(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    return "many"


def _bucket_due(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if number <= 0.0:
        return "none"
    if number < 0.5:
        return "soon"
    return "active"


def extract_feature_values(observation: dict[str, Any]) -> dict[str, str]:
    vad = observation.get("vad") or {}
    heuristic_matched = bool(observation.get("heuristic_matched", False))
    return {
        "mode_label": str(observation.get("mode_label") or "unknown"),
        "bridge_scoped": _bool_token(observation.get("bridge_scoped", False)),
        "cognitive_replay": _bool_token(observation.get("cognitive_replay", False)),
        "heuristic_matched": _bool_token(heuristic_matched),
        "heuristic_id": str(observation.get("heuristic_id") or "none")
        if heuristic_matched
        else "none",
        "available_tool_count_bucket": _bucket_tool_count(
            observation.get("available_tool_count", 0)
        ),
        "parallel_tool_calls_requested": _bool_token(
            observation.get("parallel_tool_calls_requested", False)
        ),
        "input_message_count_bucket": _bucket_input_count(
            observation.get("input_message_count", 0)
        ),
        "ongoing_task_due_bucket": _bucket_due(observation.get("ongoing_task_due", 0.0)),
        "vad_valence_bucket": _bucket_signed(vad.get("valence")),
        "vad_arousal_bucket": _bucket_unit_interval(vad.get("arousal")),
        "vad_dominance_bucket": _bucket_unit_interval(vad.get("dominance")),
    }


def _feature_signature(feature_values: dict[str, str]) -> str:
    ordered = [[key, feature_values[key]] for key in FEATURE_SCHEMA]
    return _canonical_json(ordered)


def _record_weight(record: dict[str, Any]) -> float:
    reward_total = float(record.get("reward_total", 0.0))
    return min(3.0, max(0.1, 1.0 + reward_total))


def _empty_head_counts() -> dict[str, dict[str, float]]:
    return {
        "selected_mode": {},
        "reasoning_depth": {},
        "thinking_mode": {},
        "prefix_profile": {},
        "stop_profile": {},
        "sampling_profile": {},
        "repetition_profile": {},
        "tool_choice_profile": {},
        "token_budget_bucket": {},
        "tool_parallelism_cap": {},
        **{head: {} for head in BOOLEAN_HEADS},
    }


def _increment_head(
    container: dict[str, dict[str, float]],
    head: str,
    value: Any,
    weight: float,
) -> None:
    key = str(value).lower() if isinstance(value, bool) else str(value)
    head_counts = container.setdefault(head, {})
    head_counts[key] = float(head_counts.get(key, 0.0)) + weight


def _training_paths(dataset_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    training_dir = dataset_dir / manifest["training_dir"]
    training_manifest_path = training_dir / "policy_training_manifest_v1.json"
    training_records_path = training_dir / "policy_training_records_v1.jsonl"
    if not training_manifest_path.exists() or not training_records_path.exists():
        raise FileNotFoundError(
            f"training contract not found in {training_dir}; run build-training-set first"
        )
    return manifest, training_manifest_path, training_records_path


def load_policy_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_json(path)
    if artifact.get("schema_version") != POLICY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported policy artifact schema at {path}")
    return artifact


def _head_counts_for_signature(
    artifact: dict[str, Any],
    feature_signature: str,
    head: str,
) -> tuple[dict[str, float], dict[str, float]]:
    bucket = artifact.get("bucket_models", {}).get(feature_signature, {})
    bucket_counts = bucket.get("head_counts", {}).get(head, {})
    global_counts = artifact.get("global_action_priors", {}).get(head, {})
    return bucket_counts, global_counts


def _normalized_allowed_values(allowed_values: list[Any], preference_order: list[Any]) -> list[Any]:
    if allowed_values:
        return list(allowed_values)
    return list(preference_order)


def _select_value(
    allowed_values: list[Any],
    bucket_counts: dict[str, float],
    global_counts: dict[str, float],
    preference_order: list[Any],
) -> Any:
    allowed_values = _normalized_allowed_values(allowed_values, preference_order)

    best_value = allowed_values[0]
    best_score: tuple[float, float, int] | None = None
    for order_index, value in enumerate(preference_order):
        if value not in allowed_values:
            continue
        lookup_key = str(value).lower() if isinstance(value, bool) else str(value)
        score = (
            float(bucket_counts.get(lookup_key, 0.0)),
            float(global_counts.get(lookup_key, 0.0)),
            -order_index,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_value = value
    return best_value


def _confidence_for_value(
    selected_value: Any,
    allowed_values: list[Any],
    bucket_counts: dict[str, float],
    global_counts: dict[str, float],
    preference_order: list[Any],
) -> float:
    allowed = _normalized_allowed_values(allowed_values, preference_order)
    source_counts = bucket_counts if any(float(value) > 0.0 for value in bucket_counts.values()) else global_counts
    total = 0.0
    selected_score = 0.0
    for value in allowed:
        lookup_key = str(value).lower() if isinstance(value, bool) else str(value)
        score = float(source_counts.get(lookup_key, 0.0))
        total += score
        if value == selected_value:
            selected_score = score
    if total <= 0.0:
        return 1.0 / float(len(allowed)) if allowed else 0.0
    return selected_score / total


def predict_action_with_confidence(
    artifact: dict[str, Any],
    observation: dict[str, Any],
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    normalized_mask = normalize_action_mask(action_mask)
    feature_values = extract_feature_values(observation)
    feature_signature = _feature_signature(feature_values)
    feature_signature_seen = feature_signature in artifact.get("bucket_models", {})

    selected_mode_bucket_counts, selected_mode_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "selected_mode"
    )
    selected_mode = _select_value(
        normalized_mask["allowed_modes"],
        selected_mode_bucket_counts,
        selected_mode_global_counts,
        preference_order=SELECTED_MODE_VOCAB,
    )
    reasoning_bucket_counts, reasoning_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "reasoning_depth"
    )
    reasoning_depth = _select_value(
        normalized_mask["allowed_reasoning_depths"],
        reasoning_bucket_counts,
        reasoning_global_counts,
        preference_order=REASONING_DEPTH_VOCAB,
    )
    thinking_bucket_counts, thinking_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "thinking_mode"
    )
    thinking_mode = _select_value(
        normalized_mask["allowed_thinking_modes"],
        thinking_bucket_counts,
        thinking_global_counts,
        preference_order=THINKING_MODE_VOCAB,
    )
    prefix_bucket_counts, prefix_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "prefix_profile"
    )
    prefix_profile = _select_value(
        normalized_mask["allowed_prefix_profiles"],
        prefix_bucket_counts,
        prefix_global_counts,
        preference_order=PREFIX_PROFILE_VOCAB,
    )
    stop_bucket_counts, stop_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "stop_profile"
    )
    stop_profile = _select_value(
        normalized_mask["allowed_stop_profiles"],
        stop_bucket_counts,
        stop_global_counts,
        preference_order=STOP_PROFILE_VOCAB,
    )
    sampling_bucket_counts, sampling_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "sampling_profile"
    )
    sampling_profile = _select_value(
        normalized_mask["allowed_sampling_profiles"],
        sampling_bucket_counts,
        sampling_global_counts,
        preference_order=SAMPLING_PROFILE_VOCAB,
    )
    repetition_bucket_counts, repetition_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "repetition_profile"
    )
    repetition_profile = _select_value(
        normalized_mask["allowed_repetition_profiles"],
        repetition_bucket_counts,
        repetition_global_counts,
        preference_order=REPETITION_PROFILE_VOCAB,
    )
    tool_choice_bucket_counts, tool_choice_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "tool_choice_profile"
    )
    tool_choice_profile = _select_value(
        normalized_mask["allowed_tool_choice_profiles"],
        tool_choice_bucket_counts,
        tool_choice_global_counts,
        preference_order=TOOL_CHOICE_PROFILE_VOCAB,
    )
    token_bucket_counts, token_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "token_budget_bucket"
    )
    token_budget_bucket = int(
        _select_value(
            list(TOKEN_BUDGET_BUCKET_VOCAB),
            token_bucket_counts,
            token_global_counts,
            preference_order=list(TOKEN_BUDGET_BUCKET_VOCAB),
        )
    )
    parallelism_bucket_counts, parallelism_global_counts = _head_counts_for_signature(
        artifact, feature_signature, "tool_parallelism_cap"
    )
    tool_parallelism_cap = int(
        _select_value(
            normalized_mask["allowed_tool_parallelism_caps"],
            parallelism_bucket_counts,
            parallelism_global_counts,
            preference_order=normalized_mask["allowed_tool_parallelism_caps"],
        )
    )

    boolean_permissions = {
        "interrupt_allowed": normalized_mask["allow_interrupt"],
        "replan_required": normalized_mask["allow_replan"],
        "early_stop_ok": normalized_mask["allow_early_stop"],
        "force_synthesis": normalized_mask["allow_force_synthesis"],
    }
    boolean_values: dict[str, bool] = {}
    head_confidence: dict[str, float] = {
        "selected_mode": _confidence_for_value(
            selected_mode,
            normalized_mask["allowed_modes"],
            selected_mode_bucket_counts,
            selected_mode_global_counts,
            SELECTED_MODE_VOCAB,
        ),
        "reasoning_depth": _confidence_for_value(
            reasoning_depth,
            normalized_mask["allowed_reasoning_depths"],
            reasoning_bucket_counts,
            reasoning_global_counts,
            REASONING_DEPTH_VOCAB,
        ),
        "thinking_mode": _confidence_for_value(
            thinking_mode,
            normalized_mask["allowed_thinking_modes"],
            thinking_bucket_counts,
            thinking_global_counts,
            THINKING_MODE_VOCAB,
        ),
        "prefix_profile": _confidence_for_value(
            prefix_profile,
            normalized_mask["allowed_prefix_profiles"],
            prefix_bucket_counts,
            prefix_global_counts,
            PREFIX_PROFILE_VOCAB,
        ),
        "stop_profile": _confidence_for_value(
            stop_profile,
            normalized_mask["allowed_stop_profiles"],
            stop_bucket_counts,
            stop_global_counts,
            STOP_PROFILE_VOCAB,
        ),
        "sampling_profile": _confidence_for_value(
            sampling_profile,
            normalized_mask["allowed_sampling_profiles"],
            sampling_bucket_counts,
            sampling_global_counts,
            SAMPLING_PROFILE_VOCAB,
        ),
        "repetition_profile": _confidence_for_value(
            repetition_profile,
            normalized_mask["allowed_repetition_profiles"],
            repetition_bucket_counts,
            repetition_global_counts,
            REPETITION_PROFILE_VOCAB,
        ),
        "tool_choice_profile": _confidence_for_value(
            tool_choice_profile,
            normalized_mask["allowed_tool_choice_profiles"],
            tool_choice_bucket_counts,
            tool_choice_global_counts,
            TOOL_CHOICE_PROFILE_VOCAB,
        ),
        "token_budget_bucket": _confidence_for_value(
            token_budget_bucket,
            list(TOKEN_BUDGET_BUCKET_VOCAB),
            token_bucket_counts,
            token_global_counts,
            list(TOKEN_BUDGET_BUCKET_VOCAB),
        ),
        "tool_parallelism_cap": _confidence_for_value(
            tool_parallelism_cap,
            normalized_mask["allowed_tool_parallelism_caps"],
            parallelism_bucket_counts,
            parallelism_global_counts,
            normalized_mask["allowed_tool_parallelism_caps"],
        ),
    }
    for head in BOOLEAN_HEADS:
        if not boolean_permissions[head]:
            boolean_values[head] = False
            head_confidence[head] = 1.0
            continue
        bucket_counts, global_counts = _head_counts_for_signature(artifact, feature_signature, head)
        selected = _select_value(
            [False, True],
            bucket_counts,
            global_counts,
            preference_order=[False, True],
        )
        boolean_values[head] = bool(selected)
        head_confidence[head] = _confidence_for_value(
            bool(selected),
            [False, True],
            bucket_counts,
            global_counts,
            [False, True],
        )

    action = {
        "schema_version": "policy_action_v2",
        "policy_version": artifact.get("policy_version", "unversioned"),
        "selected_mode": selected_mode,
        "reasoning_depth": reasoning_depth,
        "thinking_mode": thinking_mode,
        "prefix_profile": prefix_profile,
        "stop_profile": stop_profile,
        "sampling_profile": sampling_profile,
        "repetition_profile": repetition_profile,
        "tool_choice_profile": tool_choice_profile,
        "token_budget_bucket": token_budget_bucket,
        "tool_parallelism_cap": tool_parallelism_cap,
        **boolean_values,
        "proposal_source": "registry_artifact",
    }
    return {
        "action": action,
        "confidence": {
            "overall": min(head_confidence.values()) if head_confidence else 0.0,
            "by_head": head_confidence,
            "feature_signature_seen": feature_signature_seen,
        },
        "feature_signature": feature_signature,
        "feature_signature_seen": feature_signature_seen,
    }


def predict_action(
    artifact: dict[str, Any],
    observation: dict[str, Any],
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    return predict_action_with_confidence(artifact, observation, action_mask)["action"]


def _compare_action(candidate_action: dict[str, Any], targets: dict[str, Any]) -> bool:
    scalar_heads = [
        "selected_mode",
        "reasoning_depth",
        "thinking_mode",
        "prefix_profile",
        "stop_profile",
        "sampling_profile",
        "repetition_profile",
        "tool_choice_profile",
        "token_budget_bucket",
        "tool_parallelism_cap",
    ]
    for head in scalar_heads + BOOLEAN_HEADS:
        if candidate_action.get(head) != targets.get(head):
            return False
    return True


def _compute_training_metrics(
    artifact: dict[str, Any],
    training_records: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_match_count = 0
    reward_total_sum = 0.0
    for record in training_records:
        prediction = predict_action(artifact, record["observation"], record["action_mask"])
        if _compare_action(prediction, record["targets"]):
            exact_match_count += 1
        reward_total_sum += float(record.get("reward_total", 0.0))

    record_count = len(training_records)
    return {
        "record_count": record_count,
        "unique_feature_signature_count": len(artifact.get("bucket_models", {})),
        "exact_match_rate": 0.0 if record_count == 0 else exact_match_count / record_count,
        "reward_total_mean": 0.0 if record_count == 0 else reward_total_sum / record_count,
    }


def _default_run_root(dataset_dir: Path, registry_dir: Path | None) -> Path:
    if registry_dir is not None:
        return registry_dir.parent / "policy-runs"
    cache_root = dataset_dir.parent.parent if dataset_dir.parent.parent != dataset_dir.parent else dataset_dir.parent
    return cache_root / "policy-runs"


def train_policy(
    dataset_dir: Path,
    *,
    model_name: str,
    run_root: Path | None = None,
    registry_dir: Path | None = None,
) -> dict[str, Any]:
    dataset_manifest, training_manifest_path, training_records_path = _training_paths(dataset_dir)
    training_manifest = _read_json(training_manifest_path)
    training_records = _read_jsonl(training_records_path)
    if not training_records:
        raise ValueError("training contract contains no records")

    global_head_counts = _empty_head_counts()
    bucket_models: dict[str, dict[str, Any]] = {}

    for record in training_records:
        weight = _record_weight(record)
        feature_values = extract_feature_values(record["observation"])
        feature_signature = _feature_signature(feature_values)
        bucket = bucket_models.setdefault(
            feature_signature,
            {
                "feature_values": feature_values,
                "record_count": 0,
                "weight_total": 0.0,
                "head_counts": _empty_head_counts(),
            },
        )
        bucket["record_count"] += 1
        bucket["weight_total"] = float(bucket["weight_total"]) + weight
        targets = record["targets"]
        for head in [
            "selected_mode",
            "reasoning_depth",
            "thinking_mode",
            "prefix_profile",
            "stop_profile",
            "sampling_profile",
            "repetition_profile",
            "tool_choice_profile",
            "token_budget_bucket",
            "tool_parallelism_cap",
        ]:
            _increment_head(bucket["head_counts"], head, targets[head], weight)
            _increment_head(global_head_counts, head, targets[head], weight)
        for head in BOOLEAN_HEADS:
            _increment_head(bucket["head_counts"], head, targets[head], weight)
            _increment_head(global_head_counts, head, targets[head], weight)

    artifact = {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "algorithm": TRAINER_ALGORITHM,
        "contract_version": training_manifest["contract_version"],
        "dataset_id": dataset_manifest["dataset_id"],
        "reward_model_version": dataset_manifest.get("source_reward_model_version"),
        "model_name": model_name,
        "feature_schema": FEATURE_SCHEMA,
        "global_action_priors": global_head_counts,
        "bucket_models": bucket_models,
    }
    artifact["training_metrics"] = _compute_training_metrics(artifact, training_records)
    version_hash = hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()
    artifact["policy_version"] = f"{model_name}-{version_hash[:12]}"

    created_at_ms = int(time.time() * 1000)
    training_run_id = f"training-{created_at_ms}-{version_hash[:8]}"
    resolved_run_root = run_root or _default_run_root(dataset_dir, registry_dir)
    run_dir = resolved_run_root / training_run_id
    artifact_path = run_dir / "artifact.json"
    training_run_manifest_path = run_dir / "training_run_manifest.json"
    _write_json(artifact_path, artifact)

    training_run_manifest = {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "training_run_id": training_run_id,
        "created_at_ms": created_at_ms,
        "algorithm": TRAINER_ALGORITHM,
        "model_name": model_name,
        "policy_version": artifact["policy_version"],
        "dataset_id": dataset_manifest["dataset_id"],
        "reward_model_version": dataset_manifest.get("source_reward_model_version"),
        "dataset_manifest_path": str((dataset_dir / "manifest.json").resolve()),
        "training_manifest_path": str(training_manifest_path.resolve()),
        "artifact_path": str(artifact_path.resolve()),
        "record_count": len(training_records),
        "feature_schema": FEATURE_SCHEMA,
        "metrics": artifact["training_metrics"],
    }
    _write_json(training_run_manifest_path, training_run_manifest)
    return {
        "training_run_id": training_run_id,
        "training_run_manifest_path": str(training_run_manifest_path),
        "artifact_path": str(artifact_path),
        "record_count": len(training_records),
        "algorithm": TRAINER_ALGORITHM,
        "policy_version": artifact["policy_version"],
        "metrics": artifact["training_metrics"],
    }
