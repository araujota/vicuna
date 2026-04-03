#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAINING_RECORD_SCHEMA_VERSION = "vicuna.policy_training_record.v1"
TRAINING_MANIFEST_SCHEMA_VERSION = "vicuna.policy_training_manifest.v1"
TRAINING_CONTRACT_VERSION = "vicuna.governance_masked_heads.v3"

SELECTED_MODE_VOCAB = [
    "direct",
    "reflective",
    "tool_light",
    "tool_heavy",
    "background_defer",
]
REASONING_DEPTH_VOCAB = ["none", "short", "medium", "deep"]
TOKEN_BUDGET_BUCKET_VOCAB = [256, 512, 1024, 2048]
REASONING_BUDGET_BUCKET_VOCAB = [0, 64, 128, 256, 512, 1024]
THINKING_MODE_VOCAB = ["enabled", "disabled"]
PREFIX_PROFILE_VOCAB = ["none", "bounded_answer", "replan_outline", "json_object_open"]
STOP_PROFILE_VOCAB = ["none", "concise_answer", "markdown_code_fence"]
SAMPLING_PROFILE_VOCAB = ["provider_default", "deterministic", "balanced", "creative"]
REPETITION_PROFILE_VOCAB = ["none", "anti_stall_soft", "anti_stall_hard", "novelty_soft"]
TOOL_CHOICE_PROFILE_VOCAB = ["caller_default", "none", "auto", "required"]
BOOLEAN_HEADS = [
    "interrupt_allowed",
    "replan_required",
    "early_stop_ok",
    "force_synthesis",
]


def _build_index(vocab: list[Any]) -> dict[Any, int]:
    return {value: idx for idx, value in enumerate(vocab)}


SELECTED_MODE_INDEX = _build_index(SELECTED_MODE_VOCAB)
REASONING_DEPTH_INDEX = _build_index(REASONING_DEPTH_VOCAB)
TOKEN_BUDGET_BUCKET_INDEX = _build_index(TOKEN_BUDGET_BUCKET_VOCAB)
REASONING_BUDGET_BUCKET_INDEX = _build_index(REASONING_BUDGET_BUCKET_VOCAB)
THINKING_MODE_INDEX = _build_index(THINKING_MODE_VOCAB)
PREFIX_PROFILE_INDEX = _build_index(PREFIX_PROFILE_VOCAB)
STOP_PROFILE_INDEX = _build_index(STOP_PROFILE_VOCAB)
SAMPLING_PROFILE_INDEX = _build_index(SAMPLING_PROFILE_VOCAB)
REPETITION_PROFILE_INDEX = _build_index(REPETITION_PROFILE_VOCAB)
TOOL_CHOICE_PROFILE_INDEX = _build_index(TOOL_CHOICE_PROFILE_VOCAB)


def normalize_action_mask(action_mask: dict[str, Any]) -> dict[str, Any]:
    allowed_modes = action_mask.get("allowed_modes", [])
    allowed_reasoning_depths = action_mask.get("allowed_reasoning_depths", [])
    allowed_thinking_modes = action_mask.get("allowed_thinking_modes", THINKING_MODE_VOCAB)
    allowed_prefix_profiles = action_mask.get("allowed_prefix_profiles", PREFIX_PROFILE_VOCAB)
    allowed_stop_profiles = action_mask.get("allowed_stop_profiles", STOP_PROFILE_VOCAB)
    allowed_sampling_profiles = action_mask.get("allowed_sampling_profiles", SAMPLING_PROFILE_VOCAB)
    allowed_repetition_profiles = action_mask.get("allowed_repetition_profiles", REPETITION_PROFILE_VOCAB)
    allowed_tool_choice_profiles = action_mask.get("allowed_tool_choice_profiles", TOOL_CHOICE_PROFILE_VOCAB)
    allowed_response_budget_buckets = action_mask.get(
        "allowed_response_budget_buckets",
        TOKEN_BUDGET_BUCKET_VOCAB,
    )
    allowed_reasoning_budget_buckets = action_mask.get(
        "allowed_reasoning_budget_buckets",
        REASONING_BUDGET_BUCKET_VOCAB,
    )
    max_tool_parallelism_cap = int(action_mask.get("max_tool_parallelism_cap", 0))

    for mode in allowed_modes:
        if mode not in SELECTED_MODE_INDEX:
            raise ValueError(f"unsupported selected_mode in action mask: {mode}")
    for depth in allowed_reasoning_depths:
        if depth not in REASONING_DEPTH_INDEX:
            raise ValueError(f"unsupported reasoning_depth in action mask: {depth}")
    for value in allowed_thinking_modes:
        if value not in THINKING_MODE_INDEX:
            raise ValueError(f"unsupported thinking_mode in action mask: {value}")
    for value in allowed_prefix_profiles:
        if value not in PREFIX_PROFILE_INDEX:
            raise ValueError(f"unsupported prefix_profile in action mask: {value}")
    for value in allowed_stop_profiles:
        if value not in STOP_PROFILE_INDEX:
            raise ValueError(f"unsupported stop_profile in action mask: {value}")
    for value in allowed_sampling_profiles:
        if value not in SAMPLING_PROFILE_INDEX:
            raise ValueError(f"unsupported sampling_profile in action mask: {value}")
    for value in allowed_repetition_profiles:
        if value not in REPETITION_PROFILE_INDEX:
            raise ValueError(f"unsupported repetition_profile in action mask: {value}")
    for value in allowed_tool_choice_profiles:
        if value not in TOOL_CHOICE_PROFILE_INDEX:
            raise ValueError(f"unsupported tool_choice_profile in action mask: {value}")
    for value in allowed_response_budget_buckets:
        if int(value) not in TOKEN_BUDGET_BUCKET_INDEX:
            raise ValueError(f"unsupported response_budget_bucket in action mask: {value}")
    for value in allowed_reasoning_budget_buckets:
        if int(value) not in REASONING_BUDGET_BUCKET_INDEX:
            raise ValueError(f"unsupported reasoning_budget_bucket in action mask: {value}")
    if max_tool_parallelism_cap < 0:
        raise ValueError("max_tool_parallelism_cap must be non-negative")

    return {
        "allowed_modes": allowed_modes,
        "allowed_mode_indices": [SELECTED_MODE_INDEX[value] for value in allowed_modes],
        "allowed_reasoning_depths": allowed_reasoning_depths,
        "allowed_reasoning_depth_indices": [
            REASONING_DEPTH_INDEX[value] for value in allowed_reasoning_depths
        ],
        "allowed_thinking_modes": allowed_thinking_modes,
        "allowed_thinking_mode_indices": [THINKING_MODE_INDEX[value] for value in allowed_thinking_modes],
        "allowed_prefix_profiles": allowed_prefix_profiles,
        "allowed_prefix_profile_indices": [PREFIX_PROFILE_INDEX[value] for value in allowed_prefix_profiles],
        "allowed_stop_profiles": allowed_stop_profiles,
        "allowed_stop_profile_indices": [STOP_PROFILE_INDEX[value] for value in allowed_stop_profiles],
        "allowed_sampling_profiles": allowed_sampling_profiles,
        "allowed_sampling_profile_indices": [SAMPLING_PROFILE_INDEX[value] for value in allowed_sampling_profiles],
        "allowed_repetition_profiles": allowed_repetition_profiles,
        "allowed_repetition_profile_indices": [
            REPETITION_PROFILE_INDEX[value] for value in allowed_repetition_profiles
        ],
        "allowed_tool_choice_profiles": allowed_tool_choice_profiles,
        "allowed_tool_choice_profile_indices": [
            TOOL_CHOICE_PROFILE_INDEX[value] for value in allowed_tool_choice_profiles
        ],
        "allowed_response_budget_buckets": [int(value) for value in allowed_response_budget_buckets],
        "allowed_response_budget_bucket_indices": [
            TOKEN_BUDGET_BUCKET_INDEX[int(value)] for value in allowed_response_budget_buckets
        ],
        "allowed_reasoning_budget_buckets": [int(value) for value in allowed_reasoning_budget_buckets],
        "allowed_reasoning_budget_bucket_indices": [
            REASONING_BUDGET_BUCKET_INDEX[int(value)] for value in allowed_reasoning_budget_buckets
        ],
        "max_tool_parallelism_cap": max_tool_parallelism_cap,
        "allowed_tool_parallelism_caps": list(range(max_tool_parallelism_cap + 1)),
        "allow_interrupt": bool(action_mask.get("allow_interrupt", True)),
        "allow_replan": bool(action_mask.get("allow_replan", True)),
        "allow_early_stop": bool(action_mask.get("allow_early_stop", True)),
        "allow_force_synthesis": bool(action_mask.get("allow_force_synthesis", True)),
    }


def normalize_action_targets(action: dict[str, Any], normalized_mask: dict[str, Any]) -> dict[str, Any]:
    selected_mode = action.get("selected_mode")
    reasoning_depth = action.get("reasoning_depth")
    response_budget_bucket = int(
        action.get("response_budget_bucket", action.get("token_budget_bucket", 0))
    )
    reasoning_budget_bucket = int(action.get("reasoning_budget_bucket", 0))
    tool_parallelism_cap = int(action.get("tool_parallelism_cap", 0))
    thinking_mode = action.get(
        "thinking_mode",
        "disabled" if reasoning_depth == "none" else "enabled",
    )
    prefix_profile = action.get("prefix_profile", "none")
    stop_profile = action.get("stop_profile", "none")
    sampling_profile = action.get("sampling_profile", "provider_default")
    repetition_profile = action.get("repetition_profile", "none")
    tool_choice_profile = action.get("tool_choice_profile", "caller_default")

    if selected_mode not in SELECTED_MODE_INDEX:
        raise ValueError(f"unsupported selected_mode target: {selected_mode}")
    if reasoning_depth not in REASONING_DEPTH_INDEX:
        raise ValueError(f"unsupported reasoning_depth target: {reasoning_depth}")
    if response_budget_bucket not in TOKEN_BUDGET_BUCKET_INDEX:
        raise ValueError(f"unsupported response_budget_bucket target: {response_budget_bucket}")
    if reasoning_budget_bucket not in REASONING_BUDGET_BUCKET_INDEX:
        raise ValueError(f"unsupported reasoning_budget_bucket target: {reasoning_budget_bucket}")
    if thinking_mode not in THINKING_MODE_INDEX:
        raise ValueError(f"unsupported thinking_mode target: {thinking_mode}")
    if prefix_profile not in PREFIX_PROFILE_INDEX:
        raise ValueError(f"unsupported prefix_profile target: {prefix_profile}")
    if stop_profile not in STOP_PROFILE_INDEX:
        raise ValueError(f"unsupported stop_profile target: {stop_profile}")
    if sampling_profile not in SAMPLING_PROFILE_INDEX:
        raise ValueError(f"unsupported sampling_profile target: {sampling_profile}")
    if repetition_profile not in REPETITION_PROFILE_INDEX:
        raise ValueError(f"unsupported repetition_profile target: {repetition_profile}")
    if tool_choice_profile not in TOOL_CHOICE_PROFILE_INDEX:
        raise ValueError(f"unsupported tool_choice_profile target: {tool_choice_profile}")

    if selected_mode not in normalized_mask["allowed_modes"]:
        raise ValueError(
            f"selected_mode target {selected_mode} violates action mask {normalized_mask['allowed_modes']}"
        )
    if reasoning_depth not in normalized_mask["allowed_reasoning_depths"]:
        raise ValueError(
            "reasoning_depth target "
            f"{reasoning_depth} violates action mask {normalized_mask['allowed_reasoning_depths']}"
        )
    if thinking_mode not in normalized_mask["allowed_thinking_modes"]:
        raise ValueError(
            f"thinking_mode target {thinking_mode} violates action mask {normalized_mask['allowed_thinking_modes']}"
        )
    if prefix_profile not in normalized_mask["allowed_prefix_profiles"]:
        raise ValueError(
            f"prefix_profile target {prefix_profile} violates action mask {normalized_mask['allowed_prefix_profiles']}"
        )
    if stop_profile not in normalized_mask["allowed_stop_profiles"]:
        raise ValueError(
            f"stop_profile target {stop_profile} violates action mask {normalized_mask['allowed_stop_profiles']}"
        )
    if sampling_profile not in normalized_mask["allowed_sampling_profiles"]:
        raise ValueError(
            "sampling_profile target "
            f"{sampling_profile} violates action mask {normalized_mask['allowed_sampling_profiles']}"
        )
    if repetition_profile not in normalized_mask["allowed_repetition_profiles"]:
        raise ValueError(
            "repetition_profile target "
            f"{repetition_profile} violates action mask {normalized_mask['allowed_repetition_profiles']}"
        )
    if tool_choice_profile not in normalized_mask["allowed_tool_choice_profiles"]:
        raise ValueError(
            "tool_choice_profile target "
            f"{tool_choice_profile} violates action mask {normalized_mask['allowed_tool_choice_profiles']}"
        )
    if response_budget_bucket not in normalized_mask["allowed_response_budget_buckets"]:
        raise ValueError(
            "response_budget_bucket target "
            f"{response_budget_bucket} violates action mask {normalized_mask['allowed_response_budget_buckets']}"
        )
    if reasoning_budget_bucket not in normalized_mask["allowed_reasoning_budget_buckets"]:
        raise ValueError(
            "reasoning_budget_bucket target "
            f"{reasoning_budget_bucket} violates action mask {normalized_mask['allowed_reasoning_budget_buckets']}"
        )
    if tool_parallelism_cap not in normalized_mask["allowed_tool_parallelism_caps"]:
        raise ValueError(
            "tool_parallelism_cap target "
            f"{tool_parallelism_cap} exceeds {normalized_mask['max_tool_parallelism_cap']}"
        )

    bool_targets = {}
    bool_rules = {
        "interrupt_allowed": normalized_mask["allow_interrupt"],
        "replan_required": normalized_mask["allow_replan"],
        "early_stop_ok": normalized_mask["allow_early_stop"],
        "force_synthesis": normalized_mask["allow_force_synthesis"],
    }
    for head in BOOLEAN_HEADS:
        value = bool(action.get(head, False))
        if value and not bool_rules[head]:
            raise ValueError(f"{head} target violates action mask")
        bool_targets[head] = value

    return {
        "selected_mode": selected_mode,
        "selected_mode_index": SELECTED_MODE_INDEX[selected_mode],
        "reasoning_depth": reasoning_depth,
        "reasoning_depth_index": REASONING_DEPTH_INDEX[reasoning_depth],
        "thinking_mode": thinking_mode,
        "thinking_mode_index": THINKING_MODE_INDEX[thinking_mode],
        "prefix_profile": prefix_profile,
        "prefix_profile_index": PREFIX_PROFILE_INDEX[prefix_profile],
        "stop_profile": stop_profile,
        "stop_profile_index": STOP_PROFILE_INDEX[stop_profile],
        "sampling_profile": sampling_profile,
        "sampling_profile_index": SAMPLING_PROFILE_INDEX[sampling_profile],
        "repetition_profile": repetition_profile,
        "repetition_profile_index": REPETITION_PROFILE_INDEX[repetition_profile],
        "tool_choice_profile": tool_choice_profile,
        "tool_choice_profile_index": TOOL_CHOICE_PROFILE_INDEX[tool_choice_profile],
        "response_budget_bucket": response_budget_bucket,
        "response_budget_bucket_index": TOKEN_BUDGET_BUCKET_INDEX[response_budget_bucket],
        "token_budget_bucket": response_budget_bucket,
        "token_budget_bucket_index": TOKEN_BUDGET_BUCKET_INDEX[response_budget_bucket],
        "reasoning_budget_bucket": reasoning_budget_bucket,
        "reasoning_budget_bucket_index": REASONING_BUDGET_BUCKET_INDEX[reasoning_budget_bucket],
        "tool_parallelism_cap": tool_parallelism_cap,
        **bool_targets,
    }


def build_training_record(dataset_id: str, dataset_row: dict[str, Any]) -> dict[str, Any]:
    transition = dataset_row["transition"]
    normalized_mask = normalize_action_mask(transition["action_mask"])
    targets = normalize_action_targets(transition["executed_action"], normalized_mask)
    return {
        "schema_version": TRAINING_RECORD_SCHEMA_VERSION,
        "contract_version": TRAINING_CONTRACT_VERSION,
        "dataset_id": dataset_id,
        "export_key": dataset_row["export_key"],
        "reward_model_version": (transition.get("reward_model") or {}).get("model_version"),
        "observation": transition["observation"],
        "action_mask": normalized_mask,
        "targets": targets,
        "reward_total": float(transition.get("reward_total", 0.0)),
        "reward_events": transition.get("reward_events", []),
        "terminated": bool(transition.get("terminated", False)),
        "termination_reason": transition.get("termination_reason"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_training_corpus(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transitions_path = dataset_dir / manifest["transitions_path"]
    dataset_rows = _read_jsonl(transitions_path)
    training_records = [
        build_training_record(manifest["dataset_id"], dataset_row)
        for dataset_row in dataset_rows
    ]

    training_dir = dataset_dir / manifest["training_dir"]
    training_manifest_path = training_dir / "policy_training_manifest_v1.json"
    training_records_path = training_dir / "policy_training_records_v1.jsonl"
    training_manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "contract_version": TRAINING_CONTRACT_VERSION,
        "dataset_id": manifest["dataset_id"],
        "record_count": len(training_records),
        "selected_mode_vocab": SELECTED_MODE_VOCAB,
        "reasoning_depth_vocab": REASONING_DEPTH_VOCAB,
        "thinking_mode_vocab": THINKING_MODE_VOCAB,
        "prefix_profile_vocab": PREFIX_PROFILE_VOCAB,
        "stop_profile_vocab": STOP_PROFILE_VOCAB,
        "sampling_profile_vocab": SAMPLING_PROFILE_VOCAB,
        "repetition_profile_vocab": REPETITION_PROFILE_VOCAB,
        "tool_choice_profile_vocab": TOOL_CHOICE_PROFILE_VOCAB,
        "response_budget_bucket_vocab": TOKEN_BUDGET_BUCKET_VOCAB,
        "token_budget_bucket_vocab": TOKEN_BUDGET_BUCKET_VOCAB,
        "reasoning_budget_bucket_vocab": REASONING_BUDGET_BUCKET_VOCAB,
        "boolean_heads": BOOLEAN_HEADS,
        "records_path": str(training_records_path.relative_to(dataset_dir)),
    }
    _write_json(training_manifest_path, training_manifest)
    _write_jsonl(training_records_path, training_records)
    return {
        "training_manifest_path": str(training_manifest_path),
        "training_records_path": str(training_records_path),
        "record_count": len(training_records),
        "contract_version": TRAINING_CONTRACT_VERSION,
    }
