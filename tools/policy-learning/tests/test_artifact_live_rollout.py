#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from artifact_live_rollout import ArtifactLiveRolloutConfig, advance_artifact_rollout
from policy_registry import promote_alias, register_artifact


class StubClient:
    def __init__(self, runtime_items: list[dict]):
        self.runtime_items = list(runtime_items)
        self.applied: list[dict] = []

    def get_runtime_artifacts(self) -> dict:
        item = self.runtime_items[0] if len(self.runtime_items) == 1 else self.runtime_items.pop(0)
        return {"items": {item["artifact_kind"]: dict(item)}}

    def apply_runtime_artifact(self, payload: dict) -> dict:
        self.applied.append(dict(payload))
        return {"ok": True}


def _seed_registry(tmp_path: Path, artifact_kind: str) -> ArtifactLiveRolloutConfig:
    model_name = f"vicuna-{artifact_kind}"
    registry_dir = tmp_path / "registry"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "schema_version": "vicuna.decode_controller_artifact.v1" if artifact_kind == "decode_controller" else "vicuna.cvec_generator_artifact.v1",
        "controller_version": "decode-v1" if artifact_kind == "decode_controller" else None,
        "generator_version": "cvec-v1" if artifact_kind == "cvec_generator" else None,
        "training_metrics": {"record_count": 10, "loss_total": 0.1, "weighted_cosine": 0.99, "weighted_mse": 0.001},
    }
    first_path = runs_dir / f"{artifact_kind}-1.json"
    second_path = runs_dir / f"{artifact_kind}-2.json"
    first_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    if artifact_kind == "decode_controller":
        artifact_payload["controller_version"] = "decode-v2"
    else:
        artifact_payload["generator_version"] = "cvec-v2"
    second_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    manifest_path = runs_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps({"dataset_id": "dataset-1"}), encoding="utf-8")
    report_path = runs_dir / "eval.json"
    report_path.write_text(json.dumps({"exact_match_rate": 1.0, "invalid_action_rate": 0.0, "reward_total_mean_on_match": 1.0}), encoding="utf-8")

    first = register_artifact(
        registry_dir=registry_dir,
        model_name=model_name,
        artifact_path=first_path,
        training_run_manifest_path=manifest_path,
        evaluation_report_path=report_path,
    )
    second = register_artifact(
        registry_dir=registry_dir,
        model_name=model_name,
        artifact_path=second_path,
        training_run_manifest_path=manifest_path,
        evaluation_report_path=report_path,
    )
    promote_alias(
        registry_dir=registry_dir,
        model_name=model_name,
        alias="champion",
        version=first["version"],
        reason="baseline",
        artifact_kind=artifact_kind,
    )
    promote_alias(
        registry_dir=registry_dir,
        model_name=model_name,
        alias="candidate",
        version=second["version"],
        reason="candidate",
        artifact_kind=artifact_kind,
    )
    return ArtifactLiveRolloutConfig(
        server="http://127.0.0.1:8080",
        registry_dir=registry_dir,
        model_name=model_name,
        artifact_kind=artifact_kind,
        state_path=tmp_path / f"{artifact_kind}-rollout.json",
        shadow_min_comparisons=4,
        canary_min_samples=3,
    )


def test_decode_artifact_rollout_loads_candidate_and_enters_shadow(tmp_path: Path):
    config = _seed_registry(tmp_path, "decode_controller")
    client = StubClient(
        [
            {
                "artifact_kind": "decode_controller",
                "mode": "capture",
                "active_version": None,
                "candidate_version": None,
                "current_window": {"comparison_count": 0},
                "canary_steps": [10, 50, 100],
            }
        ]
    )
    result = advance_artifact_rollout(config, client=client)
    assert result["action"] == "activate_shadow"
    assert client.applied[0]["slot"] == "active"
    assert client.applied[1]["slot"] == "candidate"
    assert client.applied[1]["mode"] == "shadow"


def test_decode_artifact_rollout_advances_to_canary(tmp_path: Path):
    config = _seed_registry(tmp_path, "decode_controller")
    client = StubClient(
        [
            {
                "artifact_kind": "decode_controller",
                "mode": "shadow",
                "active_version": "decode-v1",
                "candidate_version": "decode-v2",
                "current_window": {"comparison_count": 5, "disagreement_rate": 0.0},
                "canary_steps": [10, 50, 100],
            }
        ]
    )
    result = advance_artifact_rollout(config, client=client)
    assert result["action"] == "activate_canary_live"
    assert client.applied[0]["mode"] == "canary_live"


def test_cvec_artifact_rollout_promotes_after_canary(tmp_path: Path):
    config = _seed_registry(tmp_path, "cvec_generator")
    client = StubClient(
        [
            {
                "artifact_kind": "cvec_generator",
                "mode": "canary_live",
                "active_version": "cvec-v1",
                "candidate_version": "cvec-v2",
                "current_rollout_step_index": 2,
                "canary_steps": [10, 50, 100],
                "current_window": {
                    "sampled_request_count": 4,
                    "mean_cosine_similarity": 0.95,
                    "mean_norm_delta": 0.05,
                },
            }
        ]
    )
    result = advance_artifact_rollout(config, client=client)
    assert result["action"] == "promote_champion"
    assert client.applied[0]["slot"] == "active"
    assert client.applied[1]["slot"] == "candidate"
    assert client.applied[1]["clear"] is True
