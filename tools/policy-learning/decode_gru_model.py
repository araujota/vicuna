#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DECODE_CONTROLLER_ARTIFACT_SCHEMA_VERSION = "vicuna.decode_controller_artifact.v1"
DECODE_CONTROLLER_RUN_SCHEMA_VERSION = "vicuna.decode_controller_training_run.v1"
DECODE_CONTROLLER_ALGORITHM = "vicuna.decode_gru_distilled.v1"


class DecodeGRUController(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        bool_head_names: list[str],
        numeric_head_names: list[str],
        profile_vocabs: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bool_head_names = list(bool_head_names)
        self.numeric_head_names = list(numeric_head_names)
        self.profile_vocabs = {key: list(value) for key, value in profile_vocabs.items()}
        self._profile_key_map = {name: name.replace(".", "__") for name in self.profile_vocabs}
        self._profile_reverse_key_map = {value: key for key, value in self._profile_key_map.items()}

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.bool_head = nn.Linear(self.hidden_dim, len(self.bool_head_names))
        self.numeric_head = nn.Linear(self.hidden_dim, len(self.numeric_head_names))
        self.profile_heads = nn.ModuleDict(
            {
                safe_name: nn.Linear(self.hidden_dim, len(self.profile_vocabs[name]))
                for name, safe_name in self._profile_key_map.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden, _ = self.gru(x)
        return {
            "bool_logits": self.bool_head(hidden),
            "numeric": self.numeric_head(hidden),
            "profiles": {
                self._profile_reverse_key_map[name]: head(hidden)
                for name, head in self.profile_heads.items()
            },
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_decode_controller_artifact(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != DECODE_CONTROLLER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported decode controller artifact schema at {path}")
    return payload


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828459045, -value))


def infer_decode_controller_action(
    artifact: dict[str, Any],
    sequence_features: list[list[float]],
    *,
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    mean = np.asarray(artifact["normalization"]["input_mean"], dtype=np.float32)
    std = np.asarray(artifact["normalization"]["input_std"], dtype=np.float32)
    x = np.asarray(sequence_features, dtype=np.float32)
    x = (x - mean) / np.maximum(std, 1e-6)
    xt = torch.tensor(x[None, :, :], dtype=torch.float32)

    input_dim = int(artifact["input_schema"]["input_dimension"])
    hidden_dim = int(artifact["architecture"]["hidden_dim"])
    bool_head_names = list(artifact["action_schema"]["boolean_fields"])
    numeric_head_names = list(artifact["action_schema"]["numeric_fields"])
    profile_vocabs = {
        key: list(value)
        for key, value in artifact["action_schema"]["profile_vocabs"].items()
    }
    model = DecodeGRUController(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bool_head_names=bool_head_names,
        numeric_head_names=numeric_head_names,
        profile_vocabs=profile_vocabs,
    )
    state_dict = {}
    for name, value in artifact["state_dict"].items():
        state_dict[name] = torch.tensor(value, dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        outputs = model(xt)

    bool_logits = outputs["bool_logits"][0, -1].tolist()
    numeric = outputs["numeric"][0, -1].tolist()
    profile_logits = {name: tensor[0, -1].tolist() for name, tensor in outputs["profiles"].items()}

    bool_values = {}
    bool_confidences = {}
    for idx, name in enumerate(bool_head_names):
        prob = _sigmoid(float(bool_logits[idx]))
        bool_values[name] = prob >= 0.5
        bool_confidences[name] = prob if bool_values[name] else 1.0 - prob

    numeric_values = {name: float(numeric[idx]) for idx, name in enumerate(numeric_head_names)}
    ranges = artifact["action_schema"]["numeric_ranges"]
    for name, bounds in ranges.items():
        low, high = float(bounds[0]), float(bounds[1])
        numeric_values[name] = min(high, max(low, numeric_values[name]))

    profile_values = {}
    profile_confidences = {}
    for name, vocab in profile_vocabs.items():
        logits = np.asarray(profile_logits[name], dtype=np.float64)
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.maximum(np.sum(np.exp(logits)), 1e-12)
        index = int(np.argmax(probs))
        profile_values[name] = vocab[index]
        profile_confidences[name] = float(probs[index])

    overall = float(sum(bool_confidences.values()) / max(len(bool_confidences), 1))
    return {
        "boolean": bool_values,
        "numeric": numeric_values,
        "profiles": profile_values,
        "confidence": {
            "overall": overall,
            "bool": bool_confidences,
            "profiles": profile_confidences,
        },
    }
