#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from decode_gru_contract import (
    ACTION_BOOL_FIELDS,
    ACTION_NUMERIC_FIELDS,
    ACTION_PROFILE_FIELDS,
    DECODE_GRU_TRAINING_CONTRACT_VERSION,
    build_decode_gru_training_corpus,
    load_decode_gru_training_records,
)
from decode_gru_model import (
    DECODE_CONTROLLER_ALGORITHM,
    DECODE_CONTROLLER_ARTIFACT_SCHEMA_VERSION,
    DECODE_CONTROLLER_RUN_SCHEMA_VERSION,
    DecodeGRUController,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class DecodeSequenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], training_manifest: dict[str, Any]):
        self.records = records
        self.training_manifest = training_manifest
        self.bool_fields = list(training_manifest["action_boolean_fields"])
        self.numeric_fields = list(training_manifest["action_numeric_fields"])
        self.profile_fields = list(training_manifest["action_profile_fields"])
        self.profile_vocabs = {
            "structure.grammar_profile_id": list(training_manifest["grammar_profile_vocab"]),
            "structure.logit_bias_profile_id": list(training_manifest["logit_bias_profile_vocab"]),
            "steering.cvec_profile_id": list(training_manifest["cvec_profile_vocab"]),
        }
        self.profile_index = {
            key: {value: idx for idx, value in enumerate(vocab)}
            for key, vocab in self.profile_vocabs.items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        steps = record["steps"]
        inputs = torch.tensor(
            [step["input_features"] for step in steps],
            dtype=torch.float32,
        )
        bool_targets = torch.tensor(
            [
                [1.0 if step["teacher_target"]["boolean"][field] else 0.0 for field in self.bool_fields]
                for step in steps
            ],
            dtype=torch.float32,
        )
        numeric_targets = torch.tensor(
            [
                [float(step["teacher_target"]["numeric"][field]) for field in self.numeric_fields]
                for step in steps
            ],
            dtype=torch.float32,
        )
        numeric_masks = torch.tensor(
            [
                [float(step["teacher_target"]["numeric_masks"][field]) for field in self.numeric_fields]
                for step in steps
            ],
            dtype=torch.float32,
        )
        profile_targets = {
            field: torch.tensor(
                [self.profile_index[field][step["teacher_target"]["profiles"][field]] for step in steps],
                dtype=torch.long,
            )
            for field in self.profile_fields
        }
        profile_masks = {
            field: torch.tensor(
                [float(step["teacher_target"]["numeric_masks"][field]) for step in steps],
                dtype=torch.float32,
            )
            for field in self.profile_fields
        }
        step_weights = torch.tensor([float(step.get("weight", 1.0)) for step in steps], dtype=torch.float32)
        return {
            "inputs": inputs,
            "bool_targets": bool_targets,
            "numeric_targets": numeric_targets,
            "numeric_masks": numeric_masks,
            "profile_targets": profile_targets,
            "profile_masks": profile_masks,
            "step_weights": step_weights,
            "length": len(steps),
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    padded_inputs = pad_sequence([item["inputs"] for item in batch], batch_first=True)
    padded_bool_targets = pad_sequence([item["bool_targets"] for item in batch], batch_first=True)
    padded_numeric_targets = pad_sequence([item["numeric_targets"] for item in batch], batch_first=True)
    padded_numeric_masks = pad_sequence([item["numeric_masks"] for item in batch], batch_first=True)
    padded_step_weights = pad_sequence([item["step_weights"] for item in batch], batch_first=True)

    profile_targets = {}
    profile_masks = {}
    for field in batch[0]["profile_targets"]:
        profile_targets[field] = pad_sequence(
            [item["profile_targets"][field] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        profile_masks[field] = pad_sequence(
            [item["profile_masks"][field] for item in batch],
            batch_first=True,
        )

    max_len = int(lengths.max().item()) if len(batch) else 0
    valid_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "inputs": padded_inputs,
        "bool_targets": padded_bool_targets,
        "numeric_targets": padded_numeric_targets,
        "numeric_masks": padded_numeric_masks,
        "profile_targets": profile_targets,
        "profile_masks": profile_masks,
        "step_weights": padded_step_weights,
        "valid_mask": valid_mask.float(),
    }


def _feature_stats(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    all_rows = np.asarray(
        [step["input_features"] for record in records for step in record["steps"]],
        dtype=np.float32,
    )
    mean = np.mean(all_rows, axis=0)
    std = np.std(all_rows, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _normalize_records(records: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(records))
    for record in normalized:
        for step in record["steps"]:
            values = np.asarray(step["input_features"], dtype=np.float32)
            step["input_features"] = ((values - mean) / std).tolist()
    return normalized


def train_decode_gru_controller(
    dataset_dir: Path,
    *,
    model_name: str,
    hidden_dim: int = 96,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    teacher_forcing_weight: float = 1.0,
    run_root: Path | None = None,
) -> dict[str, Any]:
    build_decode_gru_training_corpus(dataset_dir)
    training_manifest, records = load_decode_gru_training_records(dataset_dir)
    if not records:
        raise ValueError("no decode GRU training records found")

    input_mean, input_std = _feature_stats(records)
    normalized_records = _normalize_records(records, input_mean, input_std)
    dataset = DecodeSequenceDataset(normalized_records, training_manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch)

    device = torch.device("cpu")
    model = DecodeGRUController(
        input_dim=int(training_manifest["input_dimension"]),
        hidden_dim=int(hidden_dim),
        bool_head_names=list(training_manifest["action_boolean_fields"]),
        numeric_head_names=list(training_manifest["action_numeric_fields"]),
        profile_vocabs={
            "structure.grammar_profile_id": list(training_manifest["grammar_profile_vocab"]),
            "structure.logit_bias_profile_id": list(training_manifest["logit_bias_profile_vocab"]),
            "steering.cvec_profile_id": list(training_manifest["cvec_profile_vocab"]),
        },
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_losses: dict[str, float] = {}
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch["inputs"].to(device))
            valid_mask = batch["valid_mask"].to(device)
            step_weights = batch["step_weights"].to(device) * valid_mask

            bool_loss = F.binary_cross_entropy_with_logits(
                outputs["bool_logits"],
                batch["bool_targets"].to(device),
                reduction="none",
            )
            bool_loss = torch.sum(bool_loss * step_weights.unsqueeze(-1)) / torch.clamp(step_weights.sum(), min=1.0)

            numeric_error = torch.square(outputs["numeric"] - batch["numeric_targets"].to(device))
            numeric_weight = batch["numeric_masks"].to(device) * step_weights.unsqueeze(-1)
            numeric_loss = torch.sum(numeric_error * numeric_weight) / torch.clamp(numeric_weight.sum(), min=1.0)

            profile_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            profile_terms = 0
            for field, logits in outputs["profiles"].items():
                target = batch["profile_targets"][field].to(device)
                per_step = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    reduction="none",
                ).reshape(target.shape)
                weight = batch["profile_masks"][field].to(device) * step_weights
                profile_loss = profile_loss + (torch.sum(per_step * weight) / torch.clamp(weight.sum(), min=1.0))
                profile_terms += 1
            if profile_terms:
                profile_loss = profile_loss / float(profile_terms)

            loss = bool_loss + numeric_loss + teacher_forcing_weight * profile_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            final_losses = {
                "total": float(loss.detach().cpu().item()),
                "bool": float(bool_loss.detach().cpu().item()),
                "numeric": float(numeric_loss.detach().cpu().item()),
                "profile": float(profile_loss.detach().cpu().item()),
            }

    state_dict = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    artifact_payload = {
        "schema_version": DECODE_CONTROLLER_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "decode_controller",
        "algorithm": DECODE_CONTROLLER_ALGORITHM,
        "controller_version": "",
        "model_name": model_name,
        "input_schema": {
            "input_dimension": int(training_manifest["input_dimension"]),
        },
        "normalization": {
            "input_mean": [float(value) for value in input_mean.tolist()],
            "input_std": [float(value) for value in input_std.tolist()],
        },
        "architecture": {
            "hidden_dim": int(hidden_dim),
        },
        "action_schema": {
            "boolean_fields": list(training_manifest["action_boolean_fields"]),
            "numeric_fields": list(training_manifest["action_numeric_fields"]),
            "numeric_ranges": training_manifest["action_numeric_ranges"],
            "profile_fields": list(training_manifest["action_profile_fields"]),
            "profile_vocabs": {
                "structure.grammar_profile_id": list(training_manifest["grammar_profile_vocab"]),
                "structure.logit_bias_profile_id": list(training_manifest["logit_bias_profile_vocab"]),
                "steering.cvec_profile_id": list(training_manifest["cvec_profile_vocab"]),
            },
        },
        "state_dict": state_dict,
        "training_metrics": {
            "record_count": len(records),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "teacher_forcing_weight": float(teacher_forcing_weight),
            **final_losses,
        },
    }
    version_hash = hashlib.sha256(_canonical_json(artifact_payload).encode("utf-8")).hexdigest()[:16]
    controller_version = f"decode_gru_{version_hash}"
    artifact_payload["controller_version"] = controller_version

    timestamp_ms = int(time.time() * 1000)
    run_root = run_root or dataset_dir / "decode_gru_runs"
    run_dir = run_root / controller_version
    artifact_path = run_dir / "artifact.json"
    training_run_manifest_path = run_dir / "training_run_manifest.json"
    _write_json(artifact_path, artifact_payload)
    _write_json(
        training_run_manifest_path,
        {
            "schema_version": DECODE_CONTROLLER_RUN_SCHEMA_VERSION,
            "controller_version": controller_version,
            "dataset_id": training_manifest["dataset_id"],
            "record_count": len(records),
            "created_at_ms": timestamp_ms,
            "contract_version": DECODE_GRU_TRAINING_CONTRACT_VERSION,
            "training_metrics": artifact_payload["training_metrics"],
        },
    )
    return {
        "controller_version": controller_version,
        "artifact_path": str(artifact_path),
        "training_run_manifest_path": str(training_run_manifest_path),
        "metrics": artifact_payload["training_metrics"],
    }
