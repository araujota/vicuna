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

from cvec_generator_contract import MOMENT_FIELD_ORDER, VAD_FIELD_ORDER, flatten_state_input


CVEC_GENERATOR_ARTIFACT_SCHEMA_VERSION = "vicuna.cvec_generator_artifact.v1"
CVEC_GENERATOR_RUN_SCHEMA_VERSION = "vicuna.cvec_generator_training_run.v1"
CVEC_GENERATOR_ALGORITHM = "vicuna.emvad_mlp_generator.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cvec_generator_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_json(path)
    if artifact.get("schema_version") != CVEC_GENERATOR_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported cvec generator artifact schema at {path}")
    return artifact


def _as_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def build_cvec_state_input(observation: dict[str, Any]) -> list[float]:
    return flatten_state_input(observation)


def _forward_layers(artifact: dict[str, Any], x: np.ndarray) -> np.ndarray:
    hidden = x
    for index, layer in enumerate(artifact["layers"]):
        weights = _as_array(layer["weights"])
        bias = _as_array(layer["bias"])
        hidden = hidden @ weights.T + bias
        if index != len(artifact["layers"]) - 1:
            activation = artifact["architecture"]["activation"]
            if activation == "tanh":
                hidden = np.tanh(hidden)
            elif activation == "relu":
                hidden = np.maximum(hidden, 0.0)
            else:
                raise ValueError(f"unsupported activation {activation}")
    return hidden


def infer_cvec_from_artifact(
    artifact: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    state_input = np.asarray(build_cvec_state_input(observation), dtype=np.float64)
    input_mean = np.asarray(artifact["normalization"]["input_mean"], dtype=np.float64)
    input_std = np.asarray(artifact["normalization"]["input_std"], dtype=np.float64)
    normalized = (state_input - input_mean) / np.maximum(input_std, 1e-6)
    output = _forward_layers(artifact, normalized)

    output_mode = artifact["normalization"].get("output_mode", "none")
    if output_mode == "tanh":
        output = np.tanh(output)

    norm = float(np.linalg.norm(output))
    norm_cap = float(artifact["normalization"].get("output_norm_cap", 0.0) or 0.0)
    clipped = False
    if norm_cap > 0.0 and norm > norm_cap:
        output = output * (norm_cap / max(norm, 1e-12))
        clipped = True
        norm = float(np.linalg.norm(output))

    return {
        "generator_version": artifact["generator_version"],
        "artifact_kind": artifact.get("artifact_kind", "cvec_generator"),
        "vector": [float(value) for value in output.tolist()],
        "n_embd": int(artifact["target_embedding_dim"]),
        "il_start": int(artifact["target_layer_start"]),
        "il_end": int(artifact["target_layer_end"]),
        "norm": norm,
        "clipped": clipped,
        "input_schema": {
            "moment_fields": list(MOMENT_FIELD_ORDER),
            "vad_fields": list(VAD_FIELD_ORDER),
        },
    }
