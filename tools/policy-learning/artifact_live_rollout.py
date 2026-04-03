#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policy_live_rollout import _read_json, _write_json, _now_ms, emit_rollout_decision
from policy_registry import (
    get_alias_version,
    promote_alias,
    registry_status,
    resolve_version_entry,
)
from policy_runtime_client import PolicyRuntimeClient


ARTIFACT_ROLLOUT_STATE_SCHEMA_VERSION = "vicuna.artifact_live_rollout_state.v1"


@dataclass(frozen=True)
class ArtifactLiveRolloutConfig:
    server: str
    registry_dir: Path
    model_name: str
    artifact_kind: str
    state_path: Path
    shadow_min_comparisons: int = 64
    shadow_max_disagreement_rate: float = 0.25
    shadow_min_mean_cosine_similarity: float = 0.90
    shadow_max_mean_norm_delta: float = 0.30
    canary_min_samples: int = 8
    canary_max_disagreement_rate: float = 0.20
    canary_min_mean_cosine_similarity: float = 0.88
    canary_max_mean_norm_delta: float = 0.35


def load_artifact_rollout_state(state_path: Path, model_name: str, artifact_kind: str) -> dict[str, Any]:
    if state_path.exists():
        return _read_json(state_path)
    now_ms = _now_ms()
    return {
        "schema_version": ARTIFACT_ROLLOUT_STATE_SCHEMA_VERSION,
        "model_name": model_name,
        "artifact_kind": artifact_kind,
        "active_phase": "idle",
        "last_decision": "none",
        "last_decision_reason": "",
        "last_transition_at_ms": now_ms,
        "active_candidate_version": None,
    }


def write_artifact_rollout_state(state_path: Path, state: dict[str, Any]) -> None:
    _write_json(state_path, state)


def _load_registry_artifact_payload(
    registry_dir: Path,
    model_name: str,
    *,
    version: int,
    artifact_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = resolve_version_entry(
        registry_dir,
        model_name,
        version=version,
        artifact_kind=artifact_kind,
    )
    artifact_path = Path(entry["artifact_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return entry, payload


def _apply_artifact_slot(
    client: PolicyRuntimeClient,
    *,
    artifact_kind: str,
    slot: str,
    alias: str,
    version: int,
    artifact_version_label: str | None,
    artifact_payload: dict[str, Any],
    mode: str | None = None,
    current_rollout_step_index: int | None = None,
    reset_metrics: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "slot": slot,
        "artifact_alias": alias,
        "artifact_version": artifact_version_label or str(version),
        "artifact": artifact_payload,
        "reset_metrics": reset_metrics,
    }
    if mode is not None:
        payload["mode"] = mode
    if current_rollout_step_index is not None:
        payload["current_rollout_step_index"] = current_rollout_step_index
    return client.apply_runtime_artifact(payload)


def _clear_artifact_slot(
    client: PolicyRuntimeClient,
    *,
    artifact_kind: str,
    slot: str,
    mode: str | None = None,
    reset_metrics: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "slot": slot,
        "clear": True,
        "reset_metrics": reset_metrics,
    }
    if mode is not None:
        payload["mode"] = mode
    return client.apply_runtime_artifact(payload)


def _shadow_passes(config: ArtifactLiveRolloutConfig, runtime_item: dict[str, Any]) -> bool:
    window = runtime_item.get("current_window", {}) or {}
    if config.artifact_kind == "decode_controller":
        return float(window.get("disagreement_rate", 0.0) or 0.0) <= config.shadow_max_disagreement_rate
    return (
        float(window.get("mean_cosine_similarity", 0.0) or 0.0) >= config.shadow_min_mean_cosine_similarity
        and float(window.get("mean_norm_delta", 0.0) or 0.0) <= config.shadow_max_mean_norm_delta
    )


def _canary_passes(config: ArtifactLiveRolloutConfig, runtime_item: dict[str, Any]) -> bool:
    window = runtime_item.get("current_window", {}) or {}
    if config.artifact_kind == "decode_controller":
        return float(window.get("disagreement_rate", 0.0) or 0.0) <= config.canary_max_disagreement_rate
    return (
        float(window.get("mean_cosine_similarity", 0.0) or 0.0) >= config.canary_min_mean_cosine_similarity
        and float(window.get("mean_norm_delta", 0.0) or 0.0) <= config.canary_max_mean_norm_delta
    )


def advance_artifact_rollout(
    config: ArtifactLiveRolloutConfig,
    *,
    client: PolicyRuntimeClient | None = None,
) -> dict[str, Any]:
    client = client or PolicyRuntimeClient(config.server)
    state = load_artifact_rollout_state(config.state_path, config.model_name, config.artifact_kind)
    runtime_status = client.get_runtime_artifacts()
    runtime_item = ((runtime_status.get("items") or {}).get(config.artifact_kind) or {})
    registry_snapshot = registry_status(config.registry_dir, config.model_name)
    candidate_version = get_alias_version(
        config.registry_dir,
        config.model_name,
        "candidate",
        artifact_kind=config.artifact_kind,
    )
    champion_version = get_alias_version(
        config.registry_dir,
        config.model_name,
        "champion",
        artifact_kind=config.artifact_kind,
    )
    champion_entry = None
    champion_payload = None
    champion_artifact_version = None
    if champion_version is not None:
        champion_entry, champion_payload = _load_registry_artifact_payload(
            config.registry_dir,
            config.model_name,
            version=champion_version,
            artifact_kind=config.artifact_kind,
        )
        champion_artifact_version = champion_entry.get("artifact_version")
    candidate_entry = None
    candidate_payload = None
    candidate_artifact_version = None
    if candidate_version is not None:
        candidate_entry, candidate_payload = _load_registry_artifact_payload(
            config.registry_dir,
            config.model_name,
            version=candidate_version,
            artifact_kind=config.artifact_kind,
        )
        candidate_artifact_version = candidate_entry.get("artifact_version")

    decision: dict[str, Any] = {
        "ok": True,
        "model_name": config.model_name,
        "artifact_kind": config.artifact_kind,
        "candidate_version": candidate_version,
        "champion_version": champion_version,
        "runtime_mode_before": runtime_item.get("mode"),
        "action": "noop",
        "reason": "no_change",
        "created_at_ms": _now_ms(),
    }

    if champion_version is not None and runtime_item.get("active_version") != champion_artifact_version:
        _apply_artifact_slot(
            client,
            artifact_kind=config.artifact_kind,
            slot="active",
            alias="champion",
            version=champion_version,
            artifact_version_label=champion_artifact_version,
            artifact_payload=champion_payload or {},
            mode=str(runtime_item.get("mode") or "capture"),
        )
        runtime_status = client.get_runtime_artifacts()
        runtime_item = ((runtime_status.get("items") or {}).get(config.artifact_kind) or {})
        decision["champion_artifact_version"] = champion_entry.get("artifact_version")

    if candidate_version is None:
        state["active_phase"] = "idle"
        state["last_decision"] = decision["action"]
        state["last_decision_reason"] = "no_candidate_alias"
        state["last_transition_at_ms"] = decision["created_at_ms"]
        write_artifact_rollout_state(config.state_path, state)
        emit_rollout_decision(
            model_name=config.model_name,
            decision=decision,
            runtime_status=runtime_status,
            registry_snapshot=registry_snapshot,
        )
        return decision

    if runtime_item.get("candidate_version") != candidate_artifact_version:
        _apply_artifact_slot(
            client,
            artifact_kind=config.artifact_kind,
            slot="candidate",
            alias="candidate",
            version=candidate_version,
            artifact_version_label=candidate_artifact_version,
            artifact_payload=candidate_payload or {},
            mode="shadow",
            current_rollout_step_index=0,
            reset_metrics=True,
        )
        decision["action"] = "activate_shadow"
        decision["reason"] = "candidate_loaded"
        state["active_phase"] = "shadow"
        state["active_candidate_version"] = candidate_version
    else:
        current_mode = str(runtime_item.get("mode") or "capture")
        window = runtime_item.get("current_window", {}) or {}
        comparison_count = int(window.get("comparison_count", 0) or 0)
        sampled_request_count = int(window.get("sampled_request_count", 0) or 0)
        current_step_index = int(runtime_item.get("current_rollout_step_index", 0) or 0)
        canary_steps = list(runtime_item.get("canary_steps") or [10, 50, 100])
        if current_mode == "shadow" and comparison_count >= config.shadow_min_comparisons:
            if _shadow_passes(config, runtime_item):
                client.apply_runtime_artifact(
                    {
                        "artifact_kind": config.artifact_kind,
                        "slot": "candidate",
                        "mode": "canary_live",
                        "current_rollout_step_index": 0,
                        "reset_metrics": True,
                    }
                )
                decision["action"] = "activate_canary_live"
                decision["reason"] = "shadow_thresholds_passed"
                state["active_phase"] = "canary_live"
            else:
                if champion_version is not None:
                    promote_alias(
                        config.registry_dir,
                        config.model_name,
                        alias="candidate",
                        version=champion_version,
                        reason="shadow_rejected_reset_to_champion",
                        artifact_kind=config.artifact_kind,
                    )
                _clear_artifact_slot(
                    client,
                    artifact_kind=config.artifact_kind,
                    slot="candidate",
                    mode="capture",
                    reset_metrics=True,
                )
                decision["action"] = "reject_candidate"
                decision["reason"] = "shadow_thresholds_failed"
                state["active_phase"] = "idle"
        elif current_mode == "canary_live" and sampled_request_count >= config.canary_min_samples:
            if _canary_passes(config, runtime_item):
                if current_step_index + 1 < len(canary_steps):
                    client.apply_runtime_artifact(
                        {
                            "artifact_kind": config.artifact_kind,
                            "slot": "candidate",
                            "mode": "canary_live",
                            "current_rollout_step_index": current_step_index + 1,
                            "reset_metrics": True,
                        }
                    )
                    decision["action"] = "advance_canary_step"
                    decision["reason"] = "canary_step_passed"
                    state["active_phase"] = "canary_live"
                else:
                    promote_alias(
                        config.registry_dir,
                        config.model_name,
                        alias="champion",
                        version=candidate_version,
                        reason="candidate_promoted_after_canary",
                        artifact_kind=config.artifact_kind,
                    )
                    _apply_artifact_slot(
                        client,
                        artifact_kind=config.artifact_kind,
                        slot="active",
                        alias="champion",
                        version=candidate_version,
                        artifact_version_label=candidate_artifact_version,
                        artifact_payload=candidate_payload or {},
                        mode="capture",
                        reset_metrics=True,
                    )
                    _clear_artifact_slot(
                        client,
                        artifact_kind=config.artifact_kind,
                        slot="candidate",
                        mode="capture",
                        reset_metrics=False,
                    )
                    decision["action"] = "promote_champion"
                    decision["reason"] = "canary_completed"
                    state["active_phase"] = "idle"
            else:
                if champion_version is not None:
                    promote_alias(
                        config.registry_dir,
                        config.model_name,
                        alias="candidate",
                        version=champion_version,
                        reason="canary_rollback_reset_to_champion",
                        artifact_kind=config.artifact_kind,
                    )
                _clear_artifact_slot(
                    client,
                    artifact_kind=config.artifact_kind,
                    slot="candidate",
                    mode="shadow",
                    reset_metrics=True,
                )
                decision["action"] = "rollback_candidate"
                decision["reason"] = "canary_thresholds_failed"
                state["active_phase"] = "shadow"

    state["last_decision"] = decision["action"]
    state["last_decision_reason"] = decision["reason"]
    state["last_transition_at_ms"] = decision["created_at_ms"]
    write_artifact_rollout_state(config.state_path, state)
    emit_rollout_decision(
        model_name=config.model_name,
        decision=decision,
        runtime_status=runtime_status,
        registry_snapshot=registry_snapshot,
    )
    return decision
