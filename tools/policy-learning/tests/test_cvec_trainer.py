#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from cvec_evaluator import evaluate_cvec_generator
from cvec_generator import infer_cvec_from_artifact, load_cvec_generator_artifact
from cvec_trainer import train_cvec_generator
from policy_dataset import append_transitions
from policy_registry import register_artifact


def _moment(scale: float) -> dict:
    return {
        "confidence": 0.2 * scale,
        "curiosity": 0.1 * scale,
        "frustration": -0.1 * scale,
        "satisfaction": 0.3 * scale,
        "momentum": 0.15 * scale,
        "caution": 0.05 * scale,
        "stall": -0.05 * scale,
        "epistemic_pressure": 0.1 * scale,
        "planning_clarity": 0.2 * scale,
        "user_alignment": 0.15 * scale,
        "semantic_novelty": 0.1 * scale,
        "runtime_trust": 0.25 * scale,
        "runtime_failure_pressure": -0.1 * scale,
        "contradiction_pressure": 0.05 * scale,
    }


def _transition(transition_id: str, scale: float, target: list[float]) -> dict:
    return {
        "transition_id": transition_id,
        "request_id": f"req-{transition_id}",
        "decision_id": f"dec-{transition_id}",
        "created_at_ms": 1700000000000,
        "observation": {
            "moment": _moment(scale),
            "vad": {"valence": 0.1 * scale, "arousal": 0.2 * scale, "dominance": 0.3 * scale},
        },
        "target_control_vector": target,
        "reward_total": 1.0 + scale,
    }


def _bootstrap_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={"behavior_policy_version": "control_surface_v2", "candidate_policy_version": None},
        transitions=[
            _transition("tr-1", 0.5, [0.10, 0.20, -0.10, 0.05]),
            _transition("tr-2", 1.0, [0.20, 0.40, -0.20, 0.10]),
            _transition("tr-3", 1.5, [0.30, 0.60, -0.30, 0.15]),
        ],
    )
    return dataset_dir


def test_train_cvec_generator_is_deterministic(tmp_path: Path):
    dataset_dir = _bootstrap_dataset(tmp_path)
    first = train_cvec_generator(
        dataset_dir,
        model_name="vicuna-cvec",
        target_embedding_dim=4,
        epochs=250,
        learning_rate=0.02,
        run_root=tmp_path / "runs-first",
    )
    second = train_cvec_generator(
        dataset_dir,
        model_name="vicuna-cvec",
        target_embedding_dim=4,
        epochs=250,
        learning_rate=0.02,
        run_root=tmp_path / "runs-second",
    )

    assert first["generator_version"] == second["generator_version"]
    assert Path(first["artifact_path"]).read_text(encoding="utf-8") == Path(second["artifact_path"]).read_text(encoding="utf-8")


def test_train_and_adapter_round_trip(tmp_path: Path):
    dataset_dir = _bootstrap_dataset(tmp_path)
    trained = train_cvec_generator(
        dataset_dir,
        model_name="vicuna-cvec",
        target_embedding_dim=4,
        epochs=400,
        learning_rate=0.02,
        output_norm_cap=4.0,
        run_root=tmp_path / "runs",
    )
    artifact = load_cvec_generator_artifact(Path(trained["artifact_path"]))
    prediction = infer_cvec_from_artifact(
        artifact,
        {
            "moment": _moment(1.0),
            "vad": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
        },
    )
    assert prediction["n_embd"] == 4
    assert len(prediction["vector"]) == 4
    assert prediction["norm"] <= 4.0001

    adapter = TOOLS_ROOT / "cvec_registry_adapter.py"
    result = subprocess.run(
        [sys.executable, str(adapter), "--artifact", trained["artifact_path"]],
        input=json.dumps(
            {
                "observation": {
                    "moment": _moment(1.0),
                    "vad": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
                }
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["generator_version"] == trained["generator_version"]
    assert payload["n_embd"] == 4


def test_cvec_evaluator_and_registry_registration(tmp_path: Path):
    dataset_dir = _bootstrap_dataset(tmp_path)
    trained = train_cvec_generator(
        dataset_dir,
        model_name="vicuna-cvec",
        target_embedding_dim=4,
        epochs=350,
        learning_rate=0.02,
        run_root=tmp_path / "runs",
    )
    report_path = tmp_path / "cvec_eval.json"
    report = evaluate_cvec_generator(
        dataset_dir,
        artifact_path=Path(trained["artifact_path"]),
        report_path=report_path,
    )
    assert report["weighted_cosine"] > 0.95

    registered = register_artifact(
        registry_dir=tmp_path / "registry",
        model_name="vicuna-cvec",
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=report_path,
        tags={"validation_status": "passed"},
    )
    assert registered["artifact_kind"] == "cvec_generator"
    assert registered["generator_version"] == trained["generator_version"]
