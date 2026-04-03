#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_VERSION = "vicuna.policy_registry.v1"
BATCH_RUN_SCHEMA_VERSION = "vicuna.policy_nightly_batch.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_dir(registry_dir: Path, model_name: str) -> Path:
    return registry_dir / model_name


def _registry_path(registry_dir: Path, model_name: str) -> Path:
    return _model_dir(registry_dir, model_name) / "registry.json"


def ensure_registry(registry_dir: Path, model_name: str) -> dict[str, Any]:
    registry_path = _registry_path(registry_dir, model_name)
    if registry_path.exists():
        payload = _read_json(registry_path)
        payload.setdefault("aliases", {})
        payload.setdefault("aliases_by_kind", {})
        payload.setdefault("promotion_history", [])
        return payload
    now_ms = int(time.time() * 1000)
    manifest = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "model_name": model_name,
        "created_at_ms": now_ms,
        "updated_at_ms": now_ms,
        "versions": [],
        "aliases": {},
        "aliases_by_kind": {},
        "promotion_history": [],
    }
    _write_json(registry_path, manifest)
    return manifest


def load_registry(registry_dir: Path, model_name: str) -> dict[str, Any]:
    registry_path = _registry_path(registry_dir, model_name)
    if not registry_path.exists():
        raise FileNotFoundError(f"registry not found for model {model_name} in {registry_dir}")
    return _read_json(registry_path)


def _next_version(registry: dict[str, Any]) -> int:
    versions = registry.get("versions", [])
    if not versions:
        return 1
    return max(int(version["version"]) for version in versions) + 1


def _copy_into_version_dir(source_path: Path, destination_path: Path) -> str:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return str(destination_path)


def _artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    schema_version = str(artifact.get("schema_version", ""))
    if schema_version == "vicuna.policy_artifact.v1":
        return {
            "artifact_kind": "policy",
            "artifact_version": artifact["policy_version"],
            "metrics_summary": {
                "exact_match_rate": artifact.get("training_metrics", {}).get("exact_match_rate"),
                "record_count": artifact.get("training_metrics", {}).get("record_count"),
            },
        }
    if schema_version == "vicuna.ppo_policy_artifact.v1":
        return {
            "artifact_kind": "ppo_policy",
            "artifact_version": artifact["policy_version"],
            "metrics_summary": {
                "record_count": artifact.get("training_metrics", {}).get("record_count"),
                "warmstart_loss": artifact.get("training_metrics", {}).get("warmstart_loss"),
                "ppo_loss": artifact.get("training_metrics", {}).get("ppo_loss"),
                "live_rollout_count": artifact.get("training_metrics", {}).get("live_rollout_count"),
            },
        }
    if schema_version == "vicuna.cvec_generator_artifact.v1":
        return {
            "artifact_kind": "cvec_generator",
            "artifact_version": artifact["generator_version"],
            "metrics_summary": {
                "weighted_mse": artifact.get("training_metrics", {}).get("weighted_mse"),
                "weighted_cosine": artifact.get("training_metrics", {}).get("weighted_cosine"),
                "record_count": artifact.get("training_metrics", {}).get("record_count"),
            },
        }
    if schema_version == "vicuna.decode_controller_artifact.v1":
        return {
            "artifact_kind": "decode_controller",
            "artifact_version": artifact["controller_version"],
            "metrics_summary": {
                "record_count": artifact.get("training_metrics", {}).get("record_count"),
                "loss_total": artifact.get("training_metrics", {}).get("total"),
            },
        }
    raise ValueError(f"unsupported artifact schema_version {schema_version}")


def register_artifact(
    registry_dir: Path,
    model_name: str,
    *,
    artifact_path: Path,
    training_run_manifest_path: Path,
    evaluation_report_path: Path,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    registry = ensure_registry(registry_dir, model_name)
    version = _next_version(registry)
    version_dir = _model_dir(registry_dir, model_name) / "versions" / str(version)

    copied_artifact_path = Path(
        _copy_into_version_dir(artifact_path, version_dir / "artifact.json")
    )
    copied_training_manifest_path = Path(
        _copy_into_version_dir(
            training_run_manifest_path, version_dir / "training_run_manifest.json"
        )
    )
    copied_evaluation_report_path = Path(
        _copy_into_version_dir(
            evaluation_report_path, version_dir / "offline_evaluation.json"
        )
    )

    artifact = _read_json(copied_artifact_path)
    training_manifest = _read_json(copied_training_manifest_path)
    evaluation_report = _read_json(copied_evaluation_report_path)
    now_ms = int(time.time() * 1000)

    artifact_meta = _artifact_metadata(artifact)
    registry["versions"].append(
        {
            "version": version,
            "artifact_kind": artifact_meta["artifact_kind"],
            "artifact_version": artifact_meta["artifact_version"],
            "registered_at_ms": now_ms,
            "artifact_path": str(copied_artifact_path.relative_to(_model_dir(registry_dir, model_name))),
            "training_run_manifest_path": str(
                copied_training_manifest_path.relative_to(_model_dir(registry_dir, model_name))
            ),
            "evaluation_report_path": str(
                copied_evaluation_report_path.relative_to(_model_dir(registry_dir, model_name))
            ),
            "dataset_id": training_manifest["dataset_id"],
            "metrics_summary": {
                **artifact_meta["metrics_summary"],
                "exact_match_rate": evaluation_report.get("exact_match_rate"),
                "invalid_action_rate": evaluation_report.get("invalid_action_rate"),
                "reward_total_mean_on_match": evaluation_report.get("reward_total_mean_on_match"),
            },
            "tags": tags or {},
        }
    )
    registry["updated_at_ms"] = now_ms
    _write_json(_registry_path(registry_dir, model_name), registry)
    result = {
        "model_name": model_name,
        "version": version,
        "registry_manifest_path": str(_registry_path(registry_dir, model_name)),
        "artifact_path": str(copied_artifact_path),
        "training_run_manifest_path": str(copied_training_manifest_path),
        "evaluation_report_path": str(copied_evaluation_report_path),
        "artifact_kind": artifact_meta["artifact_kind"],
        "artifact_version": artifact_meta["artifact_version"],
    }
    if artifact_meta["artifact_kind"] == "policy":
        result["policy_version"] = artifact_meta["artifact_version"]
    if artifact_meta["artifact_kind"] == "ppo_policy":
        result["policy_version"] = artifact_meta["artifact_version"]
    if artifact_meta["artifact_kind"] == "cvec_generator":
        result["generator_version"] = artifact_meta["artifact_version"]
    if artifact_meta["artifact_kind"] == "decode_controller":
        result["controller_version"] = artifact_meta["artifact_version"]
    return result


def get_alias_version(
    registry_dir: Path,
    model_name: str,
    alias: str,
    artifact_kind: str | None = None,
) -> int | None:
    registry = ensure_registry(registry_dir, model_name)
    if artifact_kind:
        value = (registry.get("aliases_by_kind", {}).get(artifact_kind) or {}).get(alias)
    else:
        value = registry.get("aliases", {}).get(alias)
    if value is None:
        return None
    return int(value)


def _version_entry(registry: dict[str, Any], version: int) -> dict[str, Any]:
    for entry in registry.get("versions", []):
        if int(entry["version"]) == int(version):
            return entry
    raise ValueError(f"version {version} not found in registry")


def promote_alias(
    registry_dir: Path,
    model_name: str,
    *,
    alias: str,
    version: int,
    reason: str,
    artifact_kind: str | None = None,
    thresholds: dict[str, Any] | None = None,
    decision: str = "promoted",
) -> dict[str, Any]:
    registry = ensure_registry(registry_dir, model_name)
    entry = _version_entry(registry, version)
    if artifact_kind is None:
        inferred_kind = entry.get("artifact_kind")
        artifact_kind = inferred_kind if inferred_kind in {"decode_controller", "cvec_generator"} else None
    if artifact_kind:
        aliases = registry.setdefault("aliases_by_kind", {}).setdefault(artifact_kind, {})
    else:
        aliases = registry.setdefault("aliases", {})
    previous_version = aliases.get(alias)
    if decision == "promoted":
        aliases[alias] = int(version)
    event = {
        "created_at_ms": int(time.time() * 1000),
        "alias": alias,
        "artifact_kind": artifact_kind,
        "from_version": previous_version,
        "to_version": int(version),
        "reason": reason,
        "thresholds": thresholds or {},
        "decision": decision,
    }
    registry.setdefault("promotion_history", []).append(event)
    registry["updated_at_ms"] = event["created_at_ms"]
    _write_json(_registry_path(registry_dir, model_name), registry)
    return {
        "model_name": model_name,
        "alias": alias,
        "previous_version": previous_version,
        "new_version": int(version),
        "decision": decision,
        "reason": reason,
    }


def registry_status(registry_dir: Path, model_name: str) -> dict[str, Any]:
    registry = ensure_registry(registry_dir, model_name)
    latest_version = None
    if registry["versions"]:
        latest_version = max(int(entry["version"]) for entry in registry["versions"])
    return {
        "model_name": model_name,
        "version_count": len(registry["versions"]),
        "latest_version": latest_version,
        "aliases": registry.get("aliases", {}),
        "aliases_by_kind": registry.get("aliases_by_kind", {}),
        "promotion_history": registry.get("promotion_history", []),
        "registry_manifest_path": str(_registry_path(registry_dir, model_name)),
    }


def resolve_artifact_path(
    registry_dir: Path,
    model_name: str,
    *,
    alias: str | None = None,
    version: int | None = None,
    artifact_kind: str | None = None,
) -> Path:
    registry = ensure_registry(registry_dir, model_name)
    resolved_version = version
    if resolved_version is None:
        if alias is None:
            raise ValueError("alias or version is required")
        if artifact_kind:
            alias_version = (registry.get("aliases_by_kind", {}).get(artifact_kind) or {}).get(alias)
        else:
            alias_version = registry.get("aliases", {}).get(alias)
        if alias_version is None:
            raise ValueError(f"alias {alias} is not assigned")
        resolved_version = int(alias_version)
    entry = _version_entry(registry, int(resolved_version))
    return _model_dir(registry_dir, model_name) / entry["artifact_path"]


def resolve_version_entry(
    registry_dir: Path,
    model_name: str,
    *,
    alias: str | None = None,
    version: int | None = None,
    artifact_kind: str | None = None,
) -> dict[str, Any]:
    registry = ensure_registry(registry_dir, model_name)
    resolved_version = version
    if resolved_version is None:
        if alias is None:
            raise ValueError("alias or version is required")
        if artifact_kind:
            alias_version = (registry.get("aliases_by_kind", {}).get(artifact_kind) or {}).get(alias)
        else:
            alias_version = registry.get("aliases", {}).get(alias)
        if alias_version is None:
            raise ValueError(f"alias {alias} is not assigned")
        resolved_version = int(alias_version)
    entry = dict(_version_entry(registry, int(resolved_version)))
    model_dir = _model_dir(registry_dir, model_name)
    entry["resolved_version"] = int(resolved_version)
    entry["resolved_alias"] = alias
    entry["artifact_path"] = str(model_dir / entry["artifact_path"])
    entry["training_run_manifest_path"] = str(model_dir / entry["training_run_manifest_path"])
    entry["evaluation_report_path"] = str(model_dir / entry["evaluation_report_path"])
    return entry


def write_batch_run(
    registry_dir: Path,
    model_name: str,
    *,
    batch_run_id: str,
    payload: dict[str, Any],
) -> Path:
    batch_run_path = _model_dir(registry_dir, model_name) / "batch-runs" / f"{batch_run_id}.json"
    _write_json(batch_run_path, payload)
    return batch_run_path
