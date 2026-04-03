#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from policy_training_contract import (
    BOOLEAN_HEADS,
    PREFIX_PROFILE_VOCAB,
    REASONING_DEPTH_VOCAB,
    REASONING_BUDGET_BUCKET_VOCAB,
    REPETITION_PROFILE_VOCAB,
    SAMPLING_PROFILE_VOCAB,
    SELECTED_MODE_VOCAB,
    STOP_PROFILE_VOCAB,
    THINKING_MODE_VOCAB,
    TOKEN_BUDGET_BUCKET_VOCAB,
    TOOL_CHOICE_PROFILE_VOCAB,
)


PPO_POLICY_ARTIFACT_SCHEMA_VERSION = "vicuna.ppo_policy_artifact.v1"
PPO_POLICY_RUN_SCHEMA_VERSION = "vicuna.ppo_policy_training_run.v1"
PPO_TRAINER_ALGORITHM = "vicuna.masked_ppo_mlp.v1"

MODE_LABEL_VOCAB = ["chat", "bridge", "replay", "background", "unknown"]
HEAD_SPECS: list[tuple[str, list[Any]]] = [
    ("selected_mode", SELECTED_MODE_VOCAB),
    ("reasoning_depth", REASONING_DEPTH_VOCAB),
    ("thinking_mode", THINKING_MODE_VOCAB),
    ("prefix_profile", PREFIX_PROFILE_VOCAB),
    ("stop_profile", STOP_PROFILE_VOCAB),
    ("sampling_profile", SAMPLING_PROFILE_VOCAB),
    ("repetition_profile", REPETITION_PROFILE_VOCAB),
    ("tool_choice_profile", TOOL_CHOICE_PROFILE_VOCAB),
    ("response_budget_bucket", TOKEN_BUDGET_BUCKET_VOCAB),
    ("reasoning_budget_bucket", REASONING_BUDGET_BUCKET_VOCAB),
]


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _bucket_mode_label(value: Any) -> list[float]:
    label = str(value or "unknown")
    if label not in MODE_LABEL_VOCAB:
        label = "unknown"
    return [1.0 if label == item else 0.0 for item in MODE_LABEL_VOCAB]


def flatten_policy_observation(observation: dict[str, Any]) -> list[float]:
    moment = observation.get("moment") or {}
    vad = observation.get("vad") or {}
    heuristic = observation.get("heuristic") or {}
    tool_context = observation.get("tool_context") or {}
    recent_runtime = observation.get("recent_runtime") or {}
    correctness = tool_context.get("correctness") or observation.get("tool_correctness") or {}

    features: list[float] = []
    features.extend(_bucket_mode_label(observation.get("mode_label")))
    features.extend(
        [
            _bool(observation.get("bridge_scoped")),
            _bool(observation.get("cognitive_replay")),
            _bool(heuristic.get("matched") if heuristic else observation.get("heuristic_matched")),
            float(tool_context.get("available_tool_count", observation.get("available_tool_count", 0))),
            _bool(tool_context.get("parallel_tool_calls_requested", observation.get("parallel_tool_calls_requested"))),
            float(recent_runtime.get("input_message_count", observation.get("input_message_count", 0))),
            _bool(correctness.get("available", False)),
            float(correctness.get("score", 0.0)),
            float(correctness.get("confidence", 0.0)),
        ]
    )
    features.extend(
        [
            float(moment.get("confidence", 0.0)),
            float(moment.get("curiosity", 0.0)),
            float(moment.get("frustration", 0.0)),
            float(moment.get("satisfaction", 0.0)),
            float(moment.get("momentum", 0.0)),
            float(moment.get("caution", 0.0)),
            float(moment.get("stall", 0.0)),
            float(moment.get("epistemic_pressure", 0.0)),
            float(moment.get("planning_clarity", 0.0)),
            float(moment.get("user_alignment", 0.0)),
            float(moment.get("semantic_novelty", 0.0)),
            float(moment.get("runtime_trust", 0.0)),
            float(moment.get("runtime_failure_pressure", 0.0)),
            float(moment.get("contradiction_pressure", 0.0)),
        ]
    )
    features.extend(
        [
            float(vad.get("valence", 0.0)),
            float(vad.get("arousal", 0.0)),
            float(vad.get("dominance", 0.0)),
        ]
    )
    return features


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp), 1e-12)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ppo_policy_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_json(path)
    if artifact.get("schema_version") != PPO_POLICY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported PPO policy artifact schema at {path}")
    return artifact


def _forward_shared(artifact: dict[str, Any], x: np.ndarray) -> np.ndarray:
    hidden = x
    for layer in artifact["shared_layers"]:
        weights = np.asarray(layer["weights"], dtype=np.float64)
        bias = np.asarray(layer["bias"], dtype=np.float64)
        hidden = np.tanh(hidden @ weights.T + bias)
    return hidden


def _head_logits(artifact: dict[str, Any], hidden: np.ndarray, head_name: str) -> np.ndarray:
    layer = artifact["policy_heads"][head_name]
    weights = np.asarray(layer["weights"], dtype=np.float64)
    bias = np.asarray(layer["bias"], dtype=np.float64)
    return hidden @ weights.T + bias


def _value_estimate(artifact: dict[str, Any], hidden: np.ndarray) -> float:
    layer = artifact["value_head"]
    weights = np.asarray(layer["weights"], dtype=np.float64)
    bias = np.asarray(layer["bias"], dtype=np.float64)
    return float((hidden @ weights.T + bias).reshape(-1)[0])


def _allowed_values_for_head(action_mask: dict[str, Any], head_name: str, vocab: list[Any]) -> list[Any]:
    if head_name == "selected_mode":
        return list(action_mask.get("allowed_modes") or vocab)
    if head_name == "reasoning_depth":
        return list(action_mask.get("allowed_reasoning_depths") or vocab)
    if head_name == "thinking_mode":
        return list(action_mask.get("allowed_thinking_modes") or vocab)
    if head_name == "prefix_profile":
        return list(action_mask.get("allowed_prefix_profiles") or vocab)
    if head_name == "stop_profile":
        return list(action_mask.get("allowed_stop_profiles") or vocab)
    if head_name == "sampling_profile":
        return list(action_mask.get("allowed_sampling_profiles") or vocab)
    if head_name == "repetition_profile":
        return list(action_mask.get("allowed_repetition_profiles") or vocab)
    if head_name == "tool_choice_profile":
        return list(action_mask.get("allowed_tool_choice_profiles") or vocab)
    if head_name == "response_budget_bucket":
        return list(action_mask.get("allowed_response_budget_buckets") or vocab)
    if head_name == "reasoning_budget_bucket":
        return list(action_mask.get("allowed_reasoning_budget_buckets") or vocab)
    raise ValueError(f"unsupported head {head_name}")


def _masked_distribution(logits: np.ndarray, vocab: list[Any], allowed_values: list[Any]) -> tuple[np.ndarray, int]:
    allowed = set(allowed_values)
    masked = logits.copy()
    for idx, value in enumerate(vocab):
        if value not in allowed:
            masked[idx] = -1e9
    probs = _softmax(masked)
    selected_index = int(np.argmax(probs))
    return probs, selected_index


def _boolean_distribution(logits: np.ndarray, allow_true: bool) -> tuple[np.ndarray, int]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.shape[0] != 2:
        raise ValueError("boolean PPO head must emit exactly two logits")
    if not allow_true:
        logits[1] = -1e9
    probs = _softmax(logits)
    selected_index = int(np.argmax(probs))
    return probs, selected_index


def predict_ppo_action_with_confidence(
    artifact: dict[str, Any],
    observation: dict[str, Any],
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    state = np.asarray(flatten_policy_observation(observation), dtype=np.float64)
    mean = np.asarray(artifact["feature_normalization"]["mean"], dtype=np.float64)
    std = np.asarray(artifact["feature_normalization"]["std"], dtype=np.float64)
    x = (state - mean) / np.maximum(std, 1e-6)
    hidden = _forward_shared(artifact, x)

    action: dict[str, Any] = {
        "schema_version": "policy_action_v2",
        "policy_version": artifact["policy_version"],
        "proposal_source": "ppo_artifact",
    }
    selected_log_probs: list[float] = []
    head_confidences: dict[str, float] = {}

    for head_name, vocab in HEAD_SPECS:
        logits = _head_logits(artifact, hidden, head_name)
        allowed_values = _allowed_values_for_head(action_mask, head_name, vocab)
        probs, selected_index = _masked_distribution(logits, vocab, allowed_values)
        selected_value = vocab[selected_index]
        action[head_name] = selected_value
        if head_name == "response_budget_bucket":
            action["token_budget_bucket"] = selected_value
        selected_log_probs.append(float(np.log(max(probs[selected_index], 1e-12))))
        head_confidences[head_name] = float(probs[selected_index])

    tool_parallelism_vocab = list(range(int(action_mask.get("max_tool_parallelism_cap", 0)) + 1))
    if not tool_parallelism_vocab:
        tool_parallelism_vocab = [0]
    probs, selected_index = _masked_distribution(
        _head_logits(artifact, hidden, "tool_parallelism_cap"),
        tool_parallelism_vocab,
        tool_parallelism_vocab,
    )
    action["tool_parallelism_cap"] = tool_parallelism_vocab[selected_index]
    selected_log_probs.append(float(np.log(max(probs[selected_index], 1e-12))))
    head_confidences["tool_parallelism_cap"] = float(probs[selected_index])

    bool_rules = {
        "interrupt_allowed": bool(action_mask.get("allow_interrupt", True)),
        "replan_required": bool(action_mask.get("allow_replan", True)),
        "early_stop_ok": bool(action_mask.get("allow_early_stop", True)),
        "force_synthesis": bool(action_mask.get("allow_force_synthesis", True)),
    }
    for head_name in BOOLEAN_HEADS:
        logits = _head_logits(artifact, hidden, head_name)
        probs, selected_index = _boolean_distribution(logits, bool_rules[head_name])
        action[head_name] = bool(selected_index == 1)
        selected_log_probs.append(float(np.log(max(probs[selected_index], 1e-12))))
        head_confidences[head_name] = float(probs[selected_index])

    overall_confidence = float(np.exp(np.mean(selected_log_probs)))
    entropy_terms = []
    for head_name, vocab in HEAD_SPECS:
        logits = _head_logits(artifact, hidden, head_name)
        probs, _ = _masked_distribution(logits, vocab, _allowed_values_for_head(action_mask, head_name, vocab))
        entropy_terms.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))))
    tool_probs, _ = _masked_distribution(
        _head_logits(artifact, hidden, "tool_parallelism_cap"),
        tool_parallelism_vocab,
        tool_parallelism_vocab,
    )
    entropy_terms.append(float(-np.sum(tool_probs * np.log(np.maximum(tool_probs, 1e-12)))))
    for head_name in BOOLEAN_HEADS:
        logits = _head_logits(artifact, hidden, head_name)
        probs, _ = _boolean_distribution(logits, bool_rules[head_name])
        entropy_terms.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))))
    value_estimate = _value_estimate(artifact, hidden)
    return {
        "action": action,
        "confidence": {
            "overall": overall_confidence,
            "per_head": head_confidences,
            "feature_signature_seen": True,
        },
        "rollout": {
            "available": True,
            "artifact_kind": "ppo_policy",
            "policy_version": artifact["policy_version"],
            "selected_log_prob": float(np.sum(selected_log_probs)),
            "value_estimate": value_estimate,
            "entropy": float(np.mean(entropy_terms)) if entropy_terms else 0.0,
        },
    }
