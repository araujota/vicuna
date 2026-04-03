#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DATASET_MANIFEST_SCHEMA_VERSION = "vicuna.policy_dataset_manifest.v1"
DATASET_ROW_SCHEMA_VERSION = "vicuna.policy_dataset_row.v1"
DATASET_DECODE_TRACE_ROW_SCHEMA_VERSION = "vicuna.policy_decode_trace_row.v1"


def runtime_now_epoch_ms() -> int:
    return int(time.time() * 1000)


def compute_export_key(transition: dict[str, Any]) -> str:
    transition_id = str(transition.get("transition_id", "missing-transition-id"))
    request_id = str(transition.get("request_id", "missing-request-id"))
    decision_id = str(transition.get("decision_id", "missing-decision-id"))
    created_at_ms = str(transition.get("created_at_ms", 0))
    return "|".join([transition_id, request_id, decision_id, created_at_ms])


def compute_decode_trace_export_key(trace: dict[str, Any]) -> str:
    request_id = str(trace.get("request_id", "missing-request-id"))
    emotive_trace_id = str(trace.get("emotive_trace_id", "missing-emotive-trace-id"))
    created_at_ms = str(trace.get("created_at_ms", 0))
    step_count = str(trace.get("step_count", len(trace.get("steps", []))))
    return "|".join([request_id, emotive_trace_id, created_at_ms, step_count])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    return _read_json(dataset_dir / "manifest.json")


def load_dataset_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(dataset_dir)
    if not manifest:
        return []
    transitions_path = dataset_dir / manifest["transitions_path"]
    return _read_jsonl(transitions_path)


def load_decode_trace_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(dataset_dir)
    if not manifest:
        return []
    decode_traces_path = dataset_dir / manifest["decode_traces_path"]
    return _read_jsonl(decode_traces_path)


def ensure_dataset_manifest(
    dataset_dir: Path,
    dataset_id: str,
    source_base_url: str,
    export_mode: str,
    status_payload: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    existing = _read_json(manifest_path)
    now_ms = runtime_now_epoch_ms()
    if existing:
        existing["updated_at_ms"] = now_ms
        existing["source_behavior_policy_version"] = status_payload.get(
            "behavior_policy_version"
        )
        existing["source_candidate_policy_version"] = status_payload.get(
            "candidate_policy_version"
        )
        reward_model = status_payload.get("reward_model") or {}
        existing["source_reward_model_version"] = reward_model.get("model_version")
        existing["source_reward_model"] = reward_model
        existing["export_mode"] = export_mode
        existing.setdefault("stored_transition_count", 0)
        existing.setdefault("stored_decode_trace_count", 0)
        existing.setdefault("transitions_path", "data/transitions.jsonl")
        existing.setdefault("decode_traces_path", "data/decode_traces.jsonl")
        existing.setdefault("reports_dir", "reports")
        existing.setdefault("training_dir", "training")
        _write_json(manifest_path, existing)
        return existing

    reward_model = status_payload.get("reward_model") or {}
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "source_base_url": source_base_url,
        "source_behavior_policy_version": status_payload.get("behavior_policy_version"),
        "source_candidate_policy_version": status_payload.get("candidate_policy_version"),
        "source_reward_model_version": reward_model.get("model_version"),
        "source_reward_model": reward_model,
        "export_mode": export_mode,
        "stored_transition_count": 0,
        "stored_decode_trace_count": 0,
        "transitions_path": "data/transitions.jsonl",
        "decode_traces_path": "data/decode_traces.jsonl",
        "reports_dir": "reports",
        "training_dir": "training",
    }
    _write_json(manifest_path, manifest)
    return manifest


def append_transitions(
    dataset_dir: Path,
    dataset_id: str,
    source_base_url: str,
    export_mode: str,
    status_payload: dict[str, Any],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = ensure_dataset_manifest(
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        source_base_url=source_base_url,
        export_mode=export_mode,
        status_payload=status_payload,
    )
    transitions_path = dataset_dir / manifest["transitions_path"]
    existing_rows = _read_jsonl(transitions_path)
    existing_keys = {row["export_key"] for row in existing_rows}

    appended_rows = []
    for transition in transitions:
        export_key = compute_export_key(transition)
        if export_key in existing_keys:
            continue
        existing_keys.add(export_key)
        appended_rows.append(
            {
                "schema_version": DATASET_ROW_SCHEMA_VERSION,
                "export_key": export_key,
                "captured_at_ms": runtime_now_epoch_ms(),
                "transition": transition,
            }
        )

    if appended_rows:
        _append_jsonl(transitions_path, appended_rows)

    manifest["updated_at_ms"] = runtime_now_epoch_ms()
    manifest["stored_transition_count"] = len(existing_rows) + len(appended_rows)
    manifest["source_behavior_policy_version"] = status_payload.get(
        "behavior_policy_version"
    )
    manifest["source_candidate_policy_version"] = status_payload.get(
        "candidate_policy_version"
    )
    reward_model = status_payload.get("reward_model") or {}
    manifest["source_reward_model_version"] = reward_model.get("model_version")
    manifest["source_reward_model"] = reward_model
    _write_json(dataset_dir / "manifest.json", manifest)
    return {
        "dataset_dir": str(dataset_dir),
        "appended_count": len(appended_rows),
        "stored_transition_count": manifest["stored_transition_count"],
        "manifest_path": str(dataset_dir / "manifest.json"),
        "transitions_path": str(transitions_path),
    }


def append_decode_traces(
    dataset_dir: Path,
    dataset_id: str,
    source_base_url: str,
    export_mode: str,
    status_payload: dict[str, Any],
    decode_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = ensure_dataset_manifest(
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        source_base_url=source_base_url,
        export_mode=export_mode,
        status_payload=status_payload,
    )
    decode_traces_path = dataset_dir / manifest["decode_traces_path"]
    existing_rows = _read_jsonl(decode_traces_path)
    existing_keys = {row["export_key"] for row in existing_rows}

    appended_rows = []
    for trace in decode_traces:
        export_key = compute_decode_trace_export_key(trace)
        if export_key in existing_keys:
            continue
        existing_keys.add(export_key)
        appended_rows.append(
            {
                "schema_version": DATASET_DECODE_TRACE_ROW_SCHEMA_VERSION,
                "export_key": export_key,
                "captured_at_ms": runtime_now_epoch_ms(),
                "decode_trace": trace,
            }
        )

    if appended_rows:
        _append_jsonl(decode_traces_path, appended_rows)

    manifest["updated_at_ms"] = runtime_now_epoch_ms()
    manifest["stored_decode_trace_count"] = len(existing_rows) + len(appended_rows)
    manifest["source_behavior_policy_version"] = status_payload.get(
        "behavior_policy_version"
    )
    manifest["source_candidate_policy_version"] = status_payload.get(
        "candidate_policy_version"
    )
    reward_model = status_payload.get("reward_model") or {}
    manifest["source_reward_model_version"] = reward_model.get("model_version")
    manifest["source_reward_model"] = reward_model
    _write_json(dataset_dir / "manifest.json", manifest)
    return {
        "dataset_dir": str(dataset_dir),
        "appended_count": len(appended_rows),
        "stored_decode_trace_count": manifest["stored_decode_trace_count"],
        "manifest_path": str(dataset_dir / "manifest.json"),
        "decode_traces_path": str(decode_traces_path),
    }
