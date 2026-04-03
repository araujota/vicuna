#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "cvec generator tooling requires numpy; use a numpy-enabled Python "
        "interpreter such as python3.10 or install numpy into the active environment"
    ) from exc

from cvec_generator import (
    CVEC_GENERATOR_ALGORITHM,
    CVEC_GENERATOR_ARTIFACT_SCHEMA_VERSION,
    CVEC_GENERATOR_RUN_SCHEMA_VERSION,
    infer_cvec_from_artifact,
)
from cvec_generator_contract import (
    CVEC_TRAINING_CONTRACT_VERSION,
    MOMENT_FIELD_ORDER,
    VAD_FIELD_ORDER,
    build_cvec_training_corpus,
    load_cvec_training_records,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _parse_hidden_dims(hidden_dims: str | list[int] | None) -> list[int]:
    if hidden_dims is None:
        return [64, 64]
    if isinstance(hidden_dims, list):
        values = [int(value) for value in hidden_dims]
    else:
        values = [int(value.strip()) for value in str(hidden_dims).split(",") if value.strip()]
    if not values:
        raise ValueError("hidden_dims must not be empty")
    for value in values:
        if value <= 0:
            raise ValueError("hidden_dims must be positive")
    return values


def _xavier_matrix(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / float(fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))


def _init_parameters(input_dim: int, hidden_dims: list[int], output_dim: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[dict[str, np.ndarray]] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        layers.append(
            {
                "weights": _xavier_matrix(rng, fan_in, fan_out),
                "bias": np.zeros((fan_out,), dtype=np.float64),
            }
        )
    return layers


def _forward(
    layers: list[dict[str, np.ndarray]],
    x: np.ndarray,
    *,
    activation: str,
    output_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    activations = [x]
    pre_activations: list[np.ndarray] = []
    hidden = x
    for index, layer in enumerate(layers):
        z = hidden @ layer["weights"].T + layer["bias"]
        pre_activations.append(z)
        if index == len(layers) - 1:
            hidden = np.tanh(z) if output_mode == "tanh" else z
        else:
            if activation == "tanh":
                hidden = np.tanh(z)
            elif activation == "relu":
                hidden = np.maximum(z, 0.0)
            else:
                raise ValueError(f"unsupported activation {activation}")
        activations.append(hidden)
    return activations, pre_activations, hidden


def _backward(
    layers: list[dict[str, np.ndarray]],
    activations: list[np.ndarray],
    pre_activations: list[np.ndarray],
    grad_output: np.ndarray,
    *,
    activation: str,
    output_mode: str,
) -> list[dict[str, np.ndarray]]:
    grads: list[dict[str, np.ndarray]] = [
        {"weights": np.zeros_like(layer["weights"]), "bias": np.zeros_like(layer["bias"])}
        for layer in layers
    ]

    grad = grad_output
    if output_mode == "tanh":
        grad = grad * (1.0 - np.square(np.tanh(pre_activations[-1])))

    for index in reversed(range(len(layers))):
        grads[index]["weights"] = grad.T @ activations[index]
        grads[index]["bias"] = np.sum(grad, axis=0)
        if index == 0:
            break
        grad = grad @ layers[index]["weights"]
        if activation == "tanh":
            grad = grad * (1.0 - np.square(np.tanh(pre_activations[index - 1])))
        elif activation == "relu":
            grad = grad * (pre_activations[index - 1] > 0.0)
    return grads


def _load_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([record["state_input"] for record in records], dtype=np.float64)
    y = np.asarray([record["target"]["vector"] for record in records], dtype=np.float64)
    weights = np.asarray([float(record.get("weight", 1.0)) for record in records], dtype=np.float64)
    return x, y, weights


def _feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _apply_output_norm_cap(pred: np.ndarray, cap: float) -> np.ndarray:
    if cap <= 0.0:
        return pred
    norms = np.linalg.norm(pred, axis=1, keepdims=True)
    factors = np.minimum(1.0, cap / np.maximum(norms, 1e-12))
    return pred * factors


def _weighted_metrics(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    normalized_weights = weights / np.maximum(np.sum(weights), 1e-12)
    diff = pred - target
    mse = float(np.sum(normalized_weights[:, None] * np.square(diff)))
    dot = np.sum(pred * target, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = dot / np.maximum(pred_norm * target_norm, 1e-12)
    weighted_cosine = float(np.sum(normalized_weights * cosine))
    return {
        "weighted_mse": mse,
        "weighted_cosine": weighted_cosine,
        "mean_target_norm": float(np.mean(target_norm)),
        "mean_pred_norm": float(np.mean(pred_norm)),
    }


def train_cvec_generator(
    dataset_dir: Path,
    *,
    model_name: str,
    target_embedding_dim: int,
    target_layer_start: int = 0,
    target_layer_end: int = -1,
    vector_library_path: Path | None = None,
    hidden_dims: str | list[int] | None = None,
    epochs: int = 600,
    learning_rate: float = 0.01,
    reward_weight_power: float = 1.0,
    output_norm_cap: float = 8.0,
    output_mode: str = "none",
    activation: str = "tanh",
    run_root: Path | None = None,
) -> dict[str, Any]:
    if target_embedding_dim <= 0:
        raise ValueError("target_embedding_dim must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if output_mode not in {"none", "tanh"}:
        raise ValueError("output_mode must be 'none' or 'tanh'")

    build_cvec_training_corpus(
        dataset_dir,
        target_embedding_dim=target_embedding_dim,
        vector_library_path=vector_library_path,
    )
    training_manifest, records = load_cvec_training_records(dataset_dir)
    if not records:
        raise ValueError("no cvec training records found; provide target vectors or a vector library")

    x_raw, y_raw, sample_weights = _load_arrays(records)
    sample_weights = np.power(sample_weights, reward_weight_power)
    input_mean, input_std = _feature_stats(x_raw)
    x = (x_raw - input_mean) / input_std

    hidden = _parse_hidden_dims(hidden_dims)
    layers = _init_parameters(x.shape[1], hidden, target_embedding_dim)
    adam_m = [
        {"weights": np.zeros_like(layer["weights"]), "bias": np.zeros_like(layer["bias"])}
        for layer in layers
    ]
    adam_v = [
        {"weights": np.zeros_like(layer["weights"]), "bias": np.zeros_like(layer["bias"])}
        for layer in layers
    ]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    weights_sum = float(np.sum(sample_weights))
    for step in range(1, epochs + 1):
        activations, pre_activations, pred = _forward(
            layers,
            x,
            activation=activation,
            output_mode=output_mode,
        )
        pred = _apply_output_norm_cap(pred, output_norm_cap)
        grad_output = (2.0 / max(weights_sum, 1e-12)) * (pred - y_raw) * sample_weights[:, None]
        grads = _backward(
            layers,
            activations,
            pre_activations,
            grad_output,
            activation=activation,
            output_mode=output_mode,
        )
        for idx, layer in enumerate(layers):
            for key in ["weights", "bias"]:
                adam_m[idx][key] = beta1 * adam_m[idx][key] + (1.0 - beta1) * grads[idx][key]
                adam_v[idx][key] = beta2 * adam_v[idx][key] + (1.0 - beta2) * np.square(grads[idx][key])
                m_hat = adam_m[idx][key] / (1.0 - beta1 ** step)
                v_hat = adam_v[idx][key] / (1.0 - beta2 ** step)
                layer[key] = layer[key] - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    _, _, pred = _forward(
        layers,
        x,
        activation=activation,
        output_mode=output_mode,
    )
    pred = _apply_output_norm_cap(pred, output_norm_cap)
    metrics = _weighted_metrics(pred, y_raw, sample_weights)

    artifact_payload = {
        "schema_version": CVEC_GENERATOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "cvec_generator",
        "algorithm": CVEC_GENERATOR_ALGORITHM,
        "generator_version": "",
        "model_name": model_name,
        "target_embedding_dim": int(target_embedding_dim),
        "target_layer_start": int(target_layer_start),
        "target_layer_end": int(target_layer_end),
        "input_schema": {
            "moment_fields": list(MOMENT_FIELD_ORDER),
            "vad_fields": list(VAD_FIELD_ORDER),
            "input_dimension": int(x.shape[1]),
        },
        "normalization": {
            "input_mean": [float(value) for value in input_mean.tolist()],
            "input_std": [float(value) for value in input_std.tolist()],
            "output_mode": output_mode,
            "output_norm_cap": float(output_norm_cap),
        },
        "architecture": {
            "hidden_dims": hidden,
            "activation": activation,
        },
        "layers": [
            {
                "weights": layer["weights"].tolist(),
                "bias": layer["bias"].tolist(),
            }
            for layer in layers
        ],
        "training_metrics": {
            **metrics,
            "record_count": len(records),
            "reward_weight_power": float(reward_weight_power),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
        },
        "reward_model_version": training_manifest.get("reward_model_version"),
        "vector_library_summary": {
            "provided": vector_library_path is not None,
            "path": str(vector_library_path) if vector_library_path else None,
        },
    }
    version_hash = hashlib.sha256(_canonical_json(artifact_payload).encode("utf-8")).hexdigest()[:16]
    generator_version = f"emvad_cvec_{version_hash}"
    artifact_payload["generator_version"] = generator_version

    timestamp_ms = int(time.time() * 1000)
    run_root = run_root or dataset_dir / "cvec_runs"
    run_dir = run_root / generator_version
    artifact_path = run_dir / "artifact.json"
    training_run_manifest_path = run_dir / "training_run_manifest.json"
    _write_json(artifact_path, artifact_payload)
    _write_json(
        training_run_manifest_path,
        {
            "schema_version": CVEC_GENERATOR_RUN_SCHEMA_VERSION,
            "generator_version": generator_version,
            "dataset_id": training_manifest["dataset_id"],
            "record_count": len(records),
            "target_embedding_dim": int(target_embedding_dim),
            "target_layer_start": int(target_layer_start),
            "target_layer_end": int(target_layer_end),
            "created_at_ms": timestamp_ms,
            "contract_version": CVEC_TRAINING_CONTRACT_VERSION,
            "training_metrics": artifact_payload["training_metrics"],
            "reward_model_version": training_manifest.get("reward_model_version"),
        },
    )
    return {
        "generator_version": generator_version,
        "artifact_path": str(artifact_path),
        "training_run_manifest_path": str(training_run_manifest_path),
        "metrics": metrics,
    }
