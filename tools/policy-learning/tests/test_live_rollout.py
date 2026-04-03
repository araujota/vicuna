#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_live_rollout import LiveRolloutConfig, advance_rollout
from policy_registry import register_artifact, promote_alias
from test_registry import bootstrap_artifact


class StubClient:
    def __init__(self, statuses: list[dict]):
        self._statuses = list(statuses)

    def get_status(self) -> dict:
        if len(self._statuses) == 1:
            return dict(self._statuses[0])
        return dict(self._statuses.pop(0))


def make_config(tmp_path: Path) -> LiveRolloutConfig:
    return LiveRolloutConfig(
        server="http://127.0.0.1:8080",
        registry_dir=tmp_path / "registry",
        model_name="vicuna-governance",
        runtime_env_file=tmp_path / "vicuna.env",
        runtime_service="vicuna-runtime.service",
        state_path=tmp_path / "policy-rollout" / "state.json",
        journal_dir=tmp_path / "policy-rollout" / "journal",
        shadow_min_requests=5,
        shadow_max_disagreement_rate=0.25,
        shadow_max_candidate_failure_rate=0.10,
    )


def seed_registry(tmp_path: Path) -> tuple[LiveRolloutConfig, int, int]:
    config = make_config(tmp_path)
    config.runtime_env_file.write_text("VICUNA_POLICY_MODE=shadow\n", encoding="utf-8")
    evaluation_report_path, trained = bootstrap_artifact(tmp_path)
    first = register_artifact(
        registry_dir=config.registry_dir,
        model_name=config.model_name,
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=evaluation_report_path,
        tags={"validation_status": "passed"},
    )
    second = register_artifact(
        registry_dir=config.registry_dir,
        model_name=config.model_name,
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=evaluation_report_path,
        tags={"validation_status": "passed"},
    )
    promote_alias(
        registry_dir=config.registry_dir,
        model_name=config.model_name,
        alias="champion",
        version=first["version"],
        reason="approved baseline",
    )
    promote_alias(
        registry_dir=config.registry_dir,
        model_name=config.model_name,
        alias="candidate",
        version=second["version"],
        reason="new nightly candidate",
    )
    return config, first["version"], second["version"]


def test_advance_rollout_tracks_new_candidate_in_shadow(tmp_path: Path):
    config, champion_version, candidate_version = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    result = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    assert result["action"] == "activate_shadow"
    assert result["candidate_version"] == candidate_version
    assert result["champion_version"] == champion_version
    assert runtime_modes == []
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_phase"] == "shadow"
    assert state["active_candidate_version"] == candidate_version


def test_advance_rollout_enters_canary_live_after_shadow_thresholds(tmp_path: Path):
    config, _, candidate_version = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    result = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 6,
                "shadow_disagreement_count": 1,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    assert result["action"] == "activate_canary_live"
    assert result["candidate_version"] == candidate_version
    assert runtime_modes == ["canary_live"]
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_phase"] == "canary_live"


def test_advance_rollout_rejects_bad_shadow_candidate_and_restores_champion(tmp_path: Path):
    config, champion_version, _ = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    result = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 8,
                "shadow_disagreement_count": 4,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    assert result["action"] == "reject_candidate"
    assert result["reason"] == "shadow_disagreement_rate_exceeded"
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_phase"] == "idle"
    registry = json.loads((config.registry_dir / config.model_name / "registry.json").read_text(encoding="utf-8"))
    assert registry["aliases"]["candidate"] == champion_version


def test_advance_rollout_promotes_completed_canary_to_champion_and_resets_shadow(tmp_path: Path):
    config, _, candidate_version = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 8,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    result = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "canary_live",
                "rollout_state": "completed",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
                "sampled_request_count": 10,
                "live_candidate_execution_count": 10,
                "live_fallback_count": 0,
                "rollback_count": 0,
                "last_rollback_reason": None,
                "current_window": {},
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    assert result["action"] == "promote_champion"
    assert result["reason"] == "canary_completed"
    registry = json.loads((config.registry_dir / config.model_name / "registry.json").read_text(encoding="utf-8"))
    assert registry["aliases"]["champion"] == candidate_version
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["active_phase"] == "idle"


def test_advance_rollout_emits_stdout_decision_log_and_skips_file_journal(tmp_path: Path, capsys):
    config, _, _ = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=lambda mode: None,
    )

    captured = capsys.readouterr()
    assert '"event": "rollout_decision"' in captured.out
    assert not config.journal_dir.exists()

    follow_up = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 12,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )
    assert follow_up["action"] == "activate_canary_live"
    assert follow_up["reason"] == "shadow_thresholds_passed"
    assert runtime_modes == ["canary_live"]


def test_advance_rollout_resets_shadow_after_canary_rollback(tmp_path: Path):
    config, champion_version, _ = seed_registry(tmp_path)
    runtime_modes: list[str] = []
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )
    advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "shadow",
                "rollout_state": "analysis_only",
                "shadow_request_count": 8,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    result = advance_rollout(
        config,
        client=StubClient([
            {
                "mode": "canary_live",
                "rollout_state": "rolled_back",
                "shadow_request_count": 0,
                "shadow_disagreement_count": 0,
                "candidate_failure_count": 0,
                "sampled_request_count": 10,
                "live_candidate_execution_count": 0,
                "live_fallback_count": 10,
                "rollback_count": 1,
                "last_rollback_reason": "fallback_rate_exceeded",
                "current_window": {},
            }
        ]),
        runtime_mode_applier=runtime_modes.append,
    )

    assert result["action"] == "reset_to_shadow"
    assert result["reason"] == "canary_rolled_back"
    registry = json.loads((config.registry_dir / config.model_name / "registry.json").read_text(encoding="utf-8"))
    assert registry["aliases"]["candidate"] == champion_version
    assert runtime_modes == ["canary_live", "shadow"]
