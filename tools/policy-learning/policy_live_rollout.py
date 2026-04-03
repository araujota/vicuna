#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from policy_registry import get_alias_version, promote_alias, registry_status, resolve_version_entry
from policy_runtime_client import PolicyRuntimeClient


ROLLOUT_STATE_SCHEMA_VERSION = "vicuna.live_rollout_state.v1"
ROLLOUT_DECISION_SCHEMA_VERSION = "vicuna.live_rollout_decision.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def emit_log(event: str, **fields: Any) -> None:
    payload = {
        "schema_version": "vicuna.service_event.v1",
        "timestamp_ms": _now_ms(),
        "service": "policy-rollout",
        "event": event,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def load_rollout_state(state_path: Path, model_name: str) -> dict[str, Any]:
    if state_path.exists():
        return _read_json(state_path)
    now_ms = _now_ms()
    return {
        "schema_version": ROLLOUT_STATE_SCHEMA_VERSION,
        "model_name": model_name,
        "active_candidate_version": None,
        "active_candidate_policy_version": None,
        "active_phase": "idle",
        "last_decision": "none",
        "last_decision_reason": "",
        "last_transition_at_ms": now_ms,
        "last_shadow_activation_at_ms": None,
        "last_canary_activation_at_ms": None,
        "last_champion_promotion_at_ms": None,
        "last_rollback_at_ms": None,
    }


def write_rollout_state(state_path: Path, state: dict[str, Any]) -> None:
    _write_json(state_path, state)


def emit_rollout_decision(
    *,
    model_name: str,
    decision: dict[str, Any],
    runtime_status: dict[str, Any],
    registry_snapshot: dict[str, Any],
) -> None:
    created_at_ms = int(decision["created_at_ms"])
    emit_log(
        "rollout_decision",
        model_name=model_name,
        decision_id=f"rollout-{created_at_ms}",
        created_at_ms=created_at_ms,
        decision=decision,
        runtime_status=runtime_status,
        registry_status=registry_snapshot,
    )


def update_env_assignment(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def apply_runtime_mode_change(
    *,
    env_path: Path,
    runtime_service: str,
    new_mode: str,
    restart_runner: Callable[[list[str]], None] | None = None,
) -> None:
    update_env_assignment(env_path, "VICUNA_POLICY_MODE", new_mode)
    runner = restart_runner or (lambda argv: subprocess.run(argv, check=True))
    runner(["systemctl", "restart", runtime_service])


@dataclass(frozen=True)
class LiveRolloutConfig:
    server: str
    registry_dir: Path
    model_name: str
    runtime_env_file: Path
    runtime_service: str
    state_path: Path
    journal_dir: Path
    shadow_min_requests: int = 25
    shadow_max_disagreement_rate: float = 0.25
    shadow_max_candidate_failure_rate: float = 0.10


def _candidate_policy_version(
    registry_dir: Path,
    model_name: str,
    version: int | None,
) -> str | None:
    if version is None:
        return None
    entry = resolve_version_entry(registry_dir, model_name, version=version)
    return str(entry.get("policy_version")) if entry.get("policy_version") else None


def _build_decision(
    *,
    config: LiveRolloutConfig,
    candidate_version: int | None,
    champion_version: int | None,
    runtime_status: dict[str, Any],
    runtime_mode_after: str,
    action: str,
    reason: str,
) -> dict[str, Any]:
    shadow_requests = int(runtime_status.get("shadow_request_count", 0) or 0)
    shadow_disagreement_count = int(runtime_status.get("shadow_disagreement_count", 0) or 0)
    candidate_failure_count = int(runtime_status.get("candidate_failure_count", 0) or 0)
    current_window = runtime_status.get("current_window", {}) or {}
    return {
        "ok": True,
        "schema_version": ROLLOUT_DECISION_SCHEMA_VERSION,
        "model_name": config.model_name,
        "candidate_version": candidate_version,
        "champion_version": champion_version,
        "runtime_mode_before": runtime_status.get("mode"),
        "runtime_mode_after": runtime_mode_after,
        "rollout_state_before": runtime_status.get("rollout_state"),
        "action": action,
        "reason": reason,
        "shadow_metrics": {
            "request_count": shadow_requests,
            "disagreement_count": shadow_disagreement_count,
            "disagreement_rate": _safe_rate(shadow_disagreement_count, shadow_requests),
            "candidate_failure_count": candidate_failure_count,
            "candidate_failure_rate": _safe_rate(candidate_failure_count, shadow_requests),
        },
        "canary_metrics": {
            "sampled_request_count": int(runtime_status.get("sampled_request_count", 0) or 0),
            "live_candidate_execution_count": int(runtime_status.get("live_candidate_execution_count", 0) or 0),
            "live_fallback_count": int(runtime_status.get("live_fallback_count", 0) or 0),
            "rollback_count": int(runtime_status.get("rollback_count", 0) or 0),
            "last_rollback_reason": runtime_status.get("last_rollback_reason"),
            "current_window": current_window,
        },
        "created_at_ms": _now_ms(),
        "state_path": str(config.state_path),
    }


def advance_rollout(
    config: LiveRolloutConfig,
    *,
    client: PolicyRuntimeClient | None = None,
    runtime_mode_applier: Callable[[str], None] | None = None,
    now_ms: Callable[[], int] = _now_ms,
) -> dict[str, Any]:
    client = client or PolicyRuntimeClient(config.server)
    state = load_rollout_state(config.state_path, config.model_name)
    runtime_status = client.get_status()
    registry_snapshot = registry_status(config.registry_dir, config.model_name)
    candidate_version = get_alias_version(config.registry_dir, config.model_name, "candidate")
    champion_version = get_alias_version(config.registry_dir, config.model_name, "champion")
    current_mode = str(runtime_status.get("mode") or "disabled")
    current_rollout_state = str(runtime_status.get("rollout_state") or "disabled")

    def set_mode(new_mode: str) -> None:
        applier = runtime_mode_applier or (
            lambda mode: apply_runtime_mode_change(
                env_path=config.runtime_env_file,
                runtime_service=config.runtime_service,
                new_mode=mode,
            )
        )
        applier(new_mode)

    if candidate_version is None:
        decision = _build_decision(
            config=config,
            candidate_version=None,
            champion_version=champion_version,
            runtime_status=runtime_status,
            runtime_mode_after=current_mode,
            action="noop",
            reason="no_candidate_alias",
        )
        state["active_phase"] = "idle"
        state["last_decision"] = decision["action"]
        state["last_decision_reason"] = decision["reason"]
        state["last_transition_at_ms"] = now_ms()
        write_rollout_state(config.state_path, state)
        emit_rollout_decision(
            model_name=config.model_name,
            decision=decision,
            runtime_status=runtime_status,
            registry_snapshot=registry_snapshot,
        )
        return decision

    candidate_policy_version = _candidate_policy_version(
        config.registry_dir, config.model_name, candidate_version
    )

    if state.get("active_phase") == "canary_live":
        if current_rollout_state == "completed":
            promote_alias(
                registry_dir=config.registry_dir,
                model_name=config.model_name,
                alias="champion",
                version=candidate_version,
                reason="live canary completed",
            )
            set_mode("shadow")
            state["active_phase"] = "idle"
            state["last_champion_promotion_at_ms"] = now_ms()
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=candidate_version,
                runtime_status=runtime_status,
                runtime_mode_after="shadow",
                action="promote_champion",
                reason="canary_completed",
            )
            registry_snapshot = registry_status(config.registry_dir, config.model_name)
        elif current_rollout_state == "rolled_back":
            if champion_version is not None and champion_version != candidate_version:
                promote_alias(
                    registry_dir=config.registry_dir,
                    model_name=config.model_name,
                    alias="candidate",
                    version=champion_version,
                    reason="restore champion after canary rollback",
                )
                candidate_version = champion_version
                candidate_policy_version = _candidate_policy_version(
                    config.registry_dir, config.model_name, candidate_version
                )
                registry_snapshot = registry_status(config.registry_dir, config.model_name)
            set_mode("shadow")
            state["active_candidate_version"] = candidate_version
            state["active_candidate_policy_version"] = candidate_policy_version
            state["active_phase"] = "idle"
            state["last_rollback_at_ms"] = now_ms()
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after="shadow",
                action="reset_to_shadow",
                reason="canary_rolled_back",
            )
        else:
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after=current_mode,
                action="noop",
                reason="canary_in_progress",
            )
    elif champion_version is not None and candidate_version == champion_version:
        if current_mode != "shadow":
            set_mode("shadow")
            current_mode = "shadow"
            action = "reset_to_shadow"
            reason = "candidate_matches_champion"
        else:
            action = "noop"
            reason = "candidate_matches_champion"
        state["active_candidate_version"] = candidate_version
        state["active_candidate_policy_version"] = candidate_policy_version
        state["active_phase"] = "idle"
        decision = _build_decision(
            config=config,
            candidate_version=candidate_version,
            champion_version=champion_version,
            runtime_status=runtime_status,
            runtime_mode_after=current_mode,
            action=action,
            reason=reason,
        )
    # New nightly candidate or no active rollout state yet.
    elif state.get("active_candidate_version") != candidate_version:
        if current_mode != "shadow":
            set_mode("shadow")
            current_mode = "shadow"
        state["active_candidate_version"] = candidate_version
        state["active_candidate_policy_version"] = candidate_policy_version
        state["active_phase"] = "shadow"
        state["last_shadow_activation_at_ms"] = now_ms()
        decision = _build_decision(
            config=config,
            candidate_version=candidate_version,
            champion_version=champion_version,
            runtime_status=runtime_status,
            runtime_mode_after=current_mode,
            action="activate_shadow",
            reason="new_candidate_detected",
        )
    else:
        shadow_requests = int(runtime_status.get("shadow_request_count", 0) or 0)
        shadow_disagreement_count = int(runtime_status.get("shadow_disagreement_count", 0) or 0)
        candidate_failure_count = int(runtime_status.get("candidate_failure_count", 0) or 0)
        disagreement_rate = _safe_rate(shadow_disagreement_count, shadow_requests)
        candidate_failure_rate = _safe_rate(candidate_failure_count, shadow_requests)

        if current_mode != "shadow":
            set_mode("shadow")
            state["active_phase"] = "shadow"
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after="shadow",
                action="activate_shadow",
                reason="restore_shadow_mode",
            )
        elif shadow_requests < config.shadow_min_requests:
            state["active_phase"] = "shadow"
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after=current_mode,
                action="noop",
                reason="shadow_waiting_for_samples",
            )
        elif disagreement_rate > config.shadow_max_disagreement_rate:
            if champion_version is not None and champion_version != candidate_version:
                promote_alias(
                    registry_dir=config.registry_dir,
                    model_name=config.model_name,
                    alias="candidate",
                    version=champion_version,
                    reason="restore champion after shadow disagreement failure",
                )
                candidate_version = champion_version
                candidate_policy_version = _candidate_policy_version(
                    config.registry_dir, config.model_name, candidate_version
                )
                registry_snapshot = registry_status(config.registry_dir, config.model_name)
            state["active_candidate_version"] = candidate_version
            state["active_candidate_policy_version"] = candidate_policy_version
            state["active_phase"] = "idle"
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after=current_mode,
                action="reject_candidate",
                reason="shadow_disagreement_rate_exceeded",
            )
        elif candidate_failure_rate > config.shadow_max_candidate_failure_rate:
            if champion_version is not None and champion_version != candidate_version:
                promote_alias(
                    registry_dir=config.registry_dir,
                    model_name=config.model_name,
                    alias="candidate",
                    version=champion_version,
                    reason="restore champion after shadow candidate failure",
                )
                candidate_version = champion_version
                candidate_policy_version = _candidate_policy_version(
                    config.registry_dir, config.model_name, candidate_version
                )
                registry_snapshot = registry_status(config.registry_dir, config.model_name)
            state["active_candidate_version"] = candidate_version
            state["active_candidate_policy_version"] = candidate_policy_version
            state["active_phase"] = "idle"
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after=current_mode,
                action="reject_candidate",
                reason="shadow_candidate_failure_rate_exceeded",
            )
        else:
            set_mode("canary_live")
            state["active_phase"] = "canary_live"
            state["last_canary_activation_at_ms"] = now_ms()
            decision = _build_decision(
                config=config,
                candidate_version=candidate_version,
                champion_version=champion_version,
                runtime_status=runtime_status,
                runtime_mode_after="canary_live",
                action="activate_canary_live",
                reason="shadow_thresholds_passed",
            )

    state["last_decision"] = decision["action"]
    state["last_decision_reason"] = decision["reason"]
    state["last_transition_at_ms"] = int(decision["created_at_ms"])
    write_rollout_state(config.state_path, state)
    emit_rollout_decision(
        model_name=config.model_name,
        decision=decision,
        runtime_status=runtime_status,
        registry_snapshot=registry_snapshot,
    )
    return decision
