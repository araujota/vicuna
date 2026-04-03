#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from cvec_generator_contract import build_cvec_training_corpus
from policy_dataset import append_transitions


def test_build_cvec_training_corpus_resolves_profile_library(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    vector_library_path = tmp_path / "vectors.json"
    vector_library_path.write_text(
        json.dumps({"vectors": {"truthfulness": [0.5, -0.25, 0.75, -0.5]}}),
        encoding="utf-8",
    )
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={"behavior_policy_version": "control_surface_v2", "candidate_policy_version": None},
        transitions=[
            {
                "transition_id": "tr-1",
                "request_id": "req-1",
                "decision_id": "dec-1",
                "created_at_ms": 1700000000000,
                "observation": {
                    "moment": {
                        "confidence": 0.8,
                        "curiosity": 0.2,
                        "frustration": 0.0,
                        "satisfaction": 0.5,
                        "momentum": 0.6,
                        "caution": 0.4,
                        "stall": 0.1,
                        "epistemic_pressure": 0.2,
                        "planning_clarity": 0.7,
                        "user_alignment": 0.8,
                        "semantic_novelty": 0.5,
                        "runtime_trust": 0.9,
                        "runtime_failure_pressure": 0.0,
                        "contradiction_pressure": 0.1,
                    },
                    "vad": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
                },
                "target_control_profile_id": "truthfulness",
                "reward_total": 1.5,
            }
        ],
    )

    result = build_cvec_training_corpus(
        dataset_dir,
        target_embedding_dim=4,
        vector_library_path=vector_library_path,
    )

    records_path = Path(result["training_records_path"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert result["record_count"] == 1
    assert rows[0]["target"]["profile_id"] == "truthfulness"
    assert rows[0]["target"]["vector"] == [0.5, -0.25, 0.75, -0.5]
    assert rows[0]["weight"] > 1.0
