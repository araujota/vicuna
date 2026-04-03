#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

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
)
from ppo_policy import (
    HEAD_SPECS,
    PPO_POLICY_ARTIFACT_SCHEMA_VERSION,
    PPO_POLICY_RUN_SCHEMA_VERSION,
    PPO_TRAINER_ALGORITHM,
    MODE_LABEL_VOCAB,
    flatten_policy_observation,
    predict_ppo_action_with_confidence,
)
from ppo_training_contract import PPO_TRAINING_CONTRACT_VERSION, build_ppo_training_corpus, load_ppo_training_records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


HEAD_VOCABS: dict[str, list[Any]] = dict(HEAD_SPECS)
HEAD_VOCABS["tool_parallelism_cap"] = []
for head in BOOLEAN_HEADS:
    HEAD_VOCABS[head] = [False, True]


def _parse_hidden_dims(hidden_dims: str | list[int] | None) -> list[int]:
    if hidden_dims is None:
        return [64, 64]
    if isinstance(hidden_dims, list):
        return [int(value) for value in hidden_dims]
    return [int(value.strip()) for value in str(hidden_dims).split(",") if value.strip()]


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))


def _init_network(input_dim: int, hidden_dims: list[int], output_dims: dict[str, int]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    dims = [input_dim, *hidden_dims]
    shared = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        shared.append({"weights": _xavier(rng, fan_in, fan_out), "bias": np.zeros((fan_out,), dtype=np.float64)})
    hidden_dim = dims[-1]
    heads = {
        name: {"weights": _xavier(rng, hidden_dim, size), "bias": np.zeros((size,), dtype=np.float64)}
        for name, size in output_dims.items()
    }
    value_head = {"weights": _xavier(rng, hidden_dim, 1), "bias": np.zeros((1,), dtype=np.float64)}
    return {"shared_layers": shared, "policy_heads": heads, "value_head": value_head}


def _forward_shared(network: dict[str, Any], x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    activations = [x]
    pre = []
    hidden = x
    for layer in network["shared_layers"]:
        z = hidden @ layer["weights"].T + layer["bias"]
        pre.append(z)
        hidden = np.tanh(z)
        activations.append(hidden)
    return activations, pre, hidden


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _mask_logits(logits: np.ndarray, allowed_masks: np.ndarray) -> np.ndarray:
    masked = logits.copy()
    masked[~allowed_masks] = -1e9
    return masked


def _feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _bool_index(value: bool) -> int:
    return 1 if bool(value) else 0


def _prepare_arrays(records: list[dict[str, Any]]) -> dict[str, Any]:
    features = np.asarray([record["observation_features"] for record in records], dtype=np.float64)
    rewards = np.asarray([float(record["reward_total"]) for record in records], dtype=np.float64)
    live_rollout = np.asarray([
        bool(record.get("candidate_executed_live")) and bool((record.get("rollout") or {}).get("available"))
        for record in records
    ], dtype=bool)
    old_log_prob = np.asarray([
        float((record.get("rollout") or {}).get("selected_log_prob", 0.0) or 0.0)
        for record in records
    ], dtype=np.float64)
    old_value = np.asarray([
        float((record.get("rollout") or {}).get("value_estimate", 0.0) or 0.0)
        for record in records
    ], dtype=np.float64)

    target_indices: dict[str, np.ndarray] = {}
    allowed_masks: dict[str, np.ndarray] = {}
    for head_name, vocab in HEAD_SPECS:
        vocab_index = {value: idx for idx, value in enumerate(vocab)}
        target_indices[head_name] = np.asarray(
            [vocab_index[record["targets"][head_name]] for record in records],
            dtype=np.int64,
        )
        masks = []
        for record in records:
            allowed_values = record["action_mask"].get(f"allowed_{head_name}s")
            if head_name == "selected_mode":
                allowed_values = record["action_mask"]["allowed_modes"]
            elif head_name == "reasoning_depth":
                allowed_values = record["action_mask"]["allowed_reasoning_depths"]
            elif head_name == "thinking_mode":
                allowed_values = record["action_mask"]["allowed_thinking_modes"]
            elif head_name == "prefix_profile":
                allowed_values = record["action_mask"]["allowed_prefix_profiles"]
            elif head_name == "stop_profile":
                allowed_values = record["action_mask"]["allowed_stop_profiles"]
            elif head_name == "sampling_profile":
                allowed_values = record["action_mask"]["allowed_sampling_profiles"]
            elif head_name == "repetition_profile":
                allowed_values = record["action_mask"]["allowed_repetition_profiles"]
            elif head_name == "tool_choice_profile":
                allowed_values = record["action_mask"]["allowed_tool_choice_profiles"]
            elif head_name == "response_budget_bucket":
                allowed_values = record["action_mask"].get("allowed_response_budget_buckets", vocab)
            elif head_name == "reasoning_budget_bucket":
                allowed_values = record["action_mask"].get("allowed_reasoning_budget_buckets", vocab)
            allowed_set = set(allowed_values or vocab)
            masks.append([value in allowed_set for value in vocab])
        allowed_masks[head_name] = np.asarray(masks, dtype=bool)

    tool_parallelism_targets = np.asarray(
        [int(record["targets"]["tool_parallelism_cap"]) for record in records],
        dtype=np.int64,
    )
    max_cap = max(int(record["action_mask"]["max_tool_parallelism_cap"]) for record in records) if records else 0
    tool_parallelism_vocab = list(range(max_cap + 1))
    masks = []
    for record in records:
        max_allowed = int(record["action_mask"]["max_tool_parallelism_cap"])
        masks.append([value <= max_allowed for value in tool_parallelism_vocab])
    allowed_masks["tool_parallelism_cap"] = np.asarray(masks, dtype=bool)
    target_indices["tool_parallelism_cap"] = tool_parallelism_targets

    for head in BOOLEAN_HEADS:
        target_indices[head] = np.asarray([_bool_index(record["targets"][head]) for record in records], dtype=np.int64)
        allow_key = {
            "interrupt_allowed": "allow_interrupt",
            "replan_required": "allow_replan",
            "early_stop_ok": "allow_early_stop",
            "force_synthesis": "allow_force_synthesis",
        }[head]
        masks = []
        for record in records:
            allow_true = bool(record["action_mask"][allow_key])
            masks.append([True, allow_true])
        allowed_masks[head] = np.asarray(masks, dtype=bool)

    return {
        "features": features,
        "rewards": rewards,
        "live_rollout": live_rollout,
        "old_log_prob": old_log_prob,
        "old_value": old_value,
        "target_indices": target_indices,
        "allowed_masks": allowed_masks,
        "tool_parallelism_vocab": tool_parallelism_vocab,
    }


def _head_log_probs(logits: np.ndarray, masks: np.ndarray, target_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(_mask_logits(logits, masks))
    selected = probs[np.arange(len(target_indices)), target_indices]
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    return np.log(np.maximum(selected, 1e-12)), entropy


def train_ppo_policy(
    dataset_dir: Path,
    *,
    model_name: str,
    hidden_dims: str | list[int] | None = None,
    warmstart_epochs: int = 250,
    ppo_epochs: int = 200,
    learning_rate: float = 0.01,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    run_root: Path | None = None,
) -> dict[str, Any]:
    build_ppo_training_corpus(dataset_dir)
    training_manifest, records = load_ppo_training_records(dataset_dir)
    if not records:
        raise ValueError("no PPO training records found")
    arrays = _prepare_arrays(records)
    features = arrays["features"]
    input_mean, input_std = _feature_stats(features)
    x = (features - input_mean) / input_std
    hidden = _parse_hidden_dims(hidden_dims)
    output_dims = {name: len(vocab) for name, vocab in HEAD_SPECS}
    output_dims["tool_parallelism_cap"] = len(arrays["tool_parallelism_vocab"])
    for head in BOOLEAN_HEADS:
        output_dims[head] = 2
    network = _init_network(x.shape[1], hidden, output_dims)

    adam_state = {"shared_layers": [], "policy_heads": {}, "value_head": {}}
    for layer in network["shared_layers"]:
        adam_state["shared_layers"].append(
            {
                "m_w": np.zeros_like(layer["weights"]),
                "v_w": np.zeros_like(layer["weights"]),
                "m_b": np.zeros_like(layer["bias"]),
                "v_b": np.zeros_like(layer["bias"]),
            }
        )
    for head_name, layer in network["policy_heads"].items():
        adam_state["policy_heads"][head_name] = {
            "m_w": np.zeros_like(layer["weights"]),
            "v_w": np.zeros_like(layer["weights"]),
            "m_b": np.zeros_like(layer["bias"]),
            "v_b": np.zeros_like(layer["bias"]),
        }
    adam_state["value_head"] = {
        "m_w": np.zeros_like(network["value_head"]["weights"]),
        "v_w": np.zeros_like(network["value_head"]["weights"]),
        "m_b": np.zeros_like(network["value_head"]["bias"]),
        "v_b": np.zeros_like(network["value_head"]["bias"]),
    }

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def apply_adam(step: int, grad_shared, grad_heads, grad_value):
        for idx, layer in enumerate(network["shared_layers"]):
            state = adam_state["shared_layers"][idx]
            for key, grad, m_key, v_key in [
                ("weights", grad_shared[idx]["weights"], "m_w", "v_w"),
                ("bias", grad_shared[idx]["bias"], "m_b", "v_b"),
            ]:
                state[m_key] = beta1 * state[m_key] + (1.0 - beta1) * grad
                state[v_key] = beta2 * state[v_key] + (1.0 - beta2) * np.square(grad)
                m_hat = state[m_key] / (1.0 - beta1 ** step)
                v_hat = state[v_key] / (1.0 - beta2 ** step)
                layer[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        for head_name, layer in network["policy_heads"].items():
            state = adam_state["policy_heads"][head_name]
            for key, grad, m_key, v_key in [
                ("weights", grad_heads[head_name]["weights"], "m_w", "v_w"),
                ("bias", grad_heads[head_name]["bias"], "m_b", "v_b"),
            ]:
                state[m_key] = beta1 * state[m_key] + (1.0 - beta1) * grad
                state[v_key] = beta2 * state[v_key] + (1.0 - beta2) * np.square(grad)
                m_hat = state[m_key] / (1.0 - beta1 ** step)
                v_hat = state[v_key] / (1.0 - beta2 ** step)
                layer[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        for key, grad, m_key, v_key in [
            ("weights", grad_value["weights"], "m_w", "v_w"),
            ("bias", grad_value["bias"], "m_b", "v_b"),
        ]:
            state = adam_state["value_head"]
            state[m_key] = beta1 * state[m_key] + (1.0 - beta1) * grad
            state[v_key] = beta2 * state[v_key] + (1.0 - beta2) * np.square(grad)
            m_hat = state[m_key] / (1.0 - beta1 ** step)
            v_hat = state[v_key] / (1.0 - beta2 ** step)
            network["value_head"][key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def backward(activations, pre, grad_hidden, grad_heads, grad_value):
        grad_shared = [
            {"weights": np.zeros_like(layer["weights"]), "bias": np.zeros_like(layer["bias"])}
            for layer in network["shared_layers"]
        ]
        grad_policy_heads = {
            head_name: {"weights": np.zeros_like(layer["weights"]), "bias": np.zeros_like(layer["bias"])}
            for head_name, layer in network["policy_heads"].items()
        }
        for head_name, grad_logits in grad_heads.items():
            grad_policy_heads[head_name]["weights"] = grad_logits.T @ activations[-1]
            grad_policy_heads[head_name]["bias"] = np.sum(grad_logits, axis=0)
            grad_hidden += grad_logits @ network["policy_heads"][head_name]["weights"]
        grad_value_head = {
            "weights": grad_value.T @ activations[-1],
            "bias": np.sum(grad_value, axis=0),
        }
        grad_hidden += grad_value @ network["value_head"]["weights"]
        for idx in reversed(range(len(network["shared_layers"]))):
            local = grad_hidden * (1.0 - np.square(np.tanh(pre[idx])))
            grad_shared[idx]["weights"] = local.T @ activations[idx]
            grad_shared[idx]["bias"] = np.sum(local, axis=0)
            if idx > 0:
                grad_hidden = local @ network["shared_layers"][idx]["weights"]
        return grad_shared, grad_policy_heads, grad_value_head

    step = 0
    head_names = list(HEAD_SPECS) + [("tool_parallelism_cap", arrays["tool_parallelism_vocab"])] + [(head, [False, True]) for head in BOOLEAN_HEADS]
    for _ in range(warmstart_epochs):
        step += 1
        activations, pre, hidden_out = _forward_shared(network, x)
        grad_heads = {}
        total_entropy = 0.0
        for head_name, vocab in head_names:
            logits = hidden_out @ network["policy_heads"][head_name]["weights"].T + network["policy_heads"][head_name]["bias"]
            probs = _softmax(_mask_logits(logits, arrays["allowed_masks"][head_name]))
            indices = arrays["target_indices"][head_name]
            grad_logits = probs
            grad_logits[np.arange(len(indices)), indices] -= 1.0
            grad_logits /= float(len(indices))
            grad_heads[head_name] = grad_logits
            total_entropy += float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)))
        value_pred = (hidden_out @ network["value_head"]["weights"].T + network["value_head"]["bias"]).reshape(-1)
        grad_value = (2.0 / max(len(records), 1)) * (value_pred - arrays["rewards"])[:, None]
        grad_hidden = np.zeros_like(hidden_out)
        grad_shared, grad_policy_heads, grad_value_head = backward(activations, pre, grad_hidden, grad_heads, grad_value)
        apply_adam(step, grad_shared, grad_policy_heads, grad_value_head)

    for _ in range(ppo_epochs):
        if not np.any(arrays["live_rollout"]):
            break
        step += 1
        activations, pre, hidden_out = _forward_shared(network, x)
        log_prob_sum = np.zeros((len(records),), dtype=np.float64)
        entropy_sum = np.zeros((len(records),), dtype=np.float64)
        cached_probs = {}
        for head_name, vocab in head_names:
            logits = hidden_out @ network["policy_heads"][head_name]["weights"].T + network["policy_heads"][head_name]["bias"]
            probs = _softmax(_mask_logits(logits, arrays["allowed_masks"][head_name]))
            cached_probs[head_name] = probs
            indices = arrays["target_indices"][head_name]
            log_prob_sum += np.log(np.maximum(probs[np.arange(len(indices)), indices], 1e-12))
            entropy_sum += -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        value_pred = (hidden_out @ network["value_head"]["weights"].T + network["value_head"]["bias"]).reshape(-1)
        advantages = arrays["rewards"] - np.where(arrays["live_rollout"], arrays["old_value"], value_pred)
        advantages = (advantages - np.mean(advantages)) / np.maximum(np.std(advantages), 1e-6)
        ratios = np.exp(log_prob_sum - arrays["old_log_prob"])
        clipped = np.clip(ratios, 1.0 - clip_coef, 1.0 + clip_coef)
        coeff = np.where(
            arrays["live_rollout"],
            np.where(ratios * advantages <= clipped * advantages, -advantages * ratios, -advantages * clipped),
            0.0,
        )
        grad_heads = {}
        for head_name, _ in head_names:
            probs = cached_probs[head_name]
            indices = arrays["target_indices"][head_name]
            grad_logits = probs
            grad_logits[np.arange(len(indices)), indices] -= 1.0
            grad_logits *= coeff[:, None] / max(len(records), 1)
            grad_logits -= (ent_coef / max(len(records), 1)) * (-np.log(np.maximum(probs, 1e-12)) - 1.0) * probs
            grad_heads[head_name] = grad_logits
        value_target = arrays["rewards"]
        grad_value = (2.0 * vf_coef / max(len(records), 1)) * (value_pred - value_target)[:, None]
        grad_hidden = np.zeros_like(hidden_out)
        grad_shared, grad_policy_heads, grad_value_head = backward(activations, pre, grad_hidden, grad_heads, grad_value)
        apply_adam(step, grad_shared, grad_policy_heads, grad_value_head)

    artifact = {
        "schema_version": PPO_POLICY_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "ppo_policy",
        "algorithm": PPO_TRAINER_ALGORITHM,
        "policy_version": "",
        "model_name": model_name,
        "feature_schema": {
            "mode_label_vocab": MODE_LABEL_VOCAB,
            "feature_dimension": int(x.shape[1]),
        },
        "feature_normalization": {
            "mean": [float(v) for v in input_mean.tolist()],
            "std": [float(v) for v in input_std.tolist()],
        },
        "architecture": {
            "hidden_dims": hidden,
            "activation": "tanh",
        },
        "shared_layers": [
            {"weights": layer["weights"].tolist(), "bias": layer["bias"].tolist()}
            for layer in network["shared_layers"]
        ],
        "policy_heads": {
            name: {"weights": layer["weights"].tolist(), "bias": layer["bias"].tolist()}
            for name, layer in network["policy_heads"].items()
        },
        "value_head": {
            "weights": network["value_head"]["weights"].tolist(),
            "bias": network["value_head"]["bias"].tolist(),
        },
        "training_metrics": {
            "record_count": len(records),
            "rollout_record_count": int(np.sum(arrays["live_rollout"])),
            "warmstart_epochs": int(warmstart_epochs),
            "ppo_epochs": int(ppo_epochs),
            "clip_coef": float(clip_coef),
            "ent_coef": float(ent_coef),
            "vf_coef": float(vf_coef),
        },
        "reward_model_version": training_manifest.get("reward_model_version"),
    }
    version_hash = hashlib.sha256(_canonical_json(artifact).encode("utf-8")).hexdigest()[:16]
    policy_version = f"{model_name}-{version_hash}"
    artifact["policy_version"] = policy_version
    run_root = run_root or dataset_dir / "ppo_runs"
    run_dir = run_root / policy_version
    artifact_path = run_dir / "artifact.json"
    training_run_manifest_path = run_dir / "training_run_manifest.json"
    _write_json(artifact_path, artifact)
    _write_json(
        training_run_manifest_path,
        {
            "schema_version": PPO_POLICY_RUN_SCHEMA_VERSION,
            "policy_version": policy_version,
            "dataset_id": training_manifest["dataset_id"],
            "record_count": len(records),
            "rollout_record_count": int(np.sum(arrays["live_rollout"])),
            "created_at_ms": int(time.time() * 1000),
            "contract_version": PPO_TRAINING_CONTRACT_VERSION,
            "reward_model_version": training_manifest.get("reward_model_version"),
            "training_metrics": artifact["training_metrics"],
        },
    )
    return {
        "policy_version": policy_version,
        "artifact_path": str(artifact_path),
        "training_run_manifest_path": str(training_run_manifest_path),
        "metrics": artifact["training_metrics"],
    }
