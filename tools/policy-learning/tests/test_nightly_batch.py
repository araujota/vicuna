#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from cli import nightly_batch


def make_transition(transition_id: str, tool_count: int) -> dict:
    selected_mode = "direct" if tool_count == 0 else "tool_light"
    return {
        "transition_id": transition_id,
        "request_id": f"req-{transition_id}",
        "decision_id": f"dec-{transition_id}",
        "created_at_ms": 1700000000000,
        "observation": {
            "mode_label": "chat",
            "bridge_scoped": False,
            "cognitive_replay": False,
            "heuristic_matched": False,
            "available_tool_count": tool_count,
            "parallel_tool_calls_requested": tool_count > 0,
            "input_message_count": 1,
            "vad": {"valence": 0.0, "arousal": 0.2, "dominance": 0.4},
        },
        "action_mask": {
            "allowed_modes": ["direct", "tool_light"],
            "allowed_reasoning_depths": ["short", "medium"],
            "max_tool_parallelism_cap": 1,
            "allow_interrupt": True,
            "allow_replan": True,
            "allow_early_stop": True,
            "allow_force_synthesis": True,
        },
        "executed_action": {
            "selected_mode": selected_mode,
            "reasoning_depth": "short",
            "token_budget_bucket": 512,
            "tool_parallelism_cap": 0,
            "interrupt_allowed": False,
            "replan_required": False,
            "early_stop_ok": True,
            "force_synthesis": False,
        },
        "reward_events": [],
        "reward_total": 1.0,
        "terminated": True,
        "termination_reason": "stop",
    }


class PolicyHandler(BaseHTTPRequestHandler):
    transitions = [make_transition("tr-1", 0), make_transition("tr-2", 1)]

    def do_GET(self):
        if self.path.startswith("/v1/policy/status"):
            payload = {
                "behavior_policy_version": "control_surface_v2",
                "candidate_policy_version": None,
            }
        elif self.path.startswith("/v1/policy/transitions"):
            payload = {"items": list(self.transitions)}
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):
        return


def with_server():
    server = HTTPServer(("127.0.0.1", 0), PolicyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_nightly_batch_promotes_candidate_alias(tmp_path: Path):
    server_cm = with_server()
    server = next(server_cm)
    try:
        args = type(
            "Args",
            (),
            {
                "server": f"http://127.0.0.1:{server.server_port}",
                "dataset_dir": str(tmp_path / "dataset"),
                "dataset_id": "vicuna-nightly-v1",
                "registry_dir": str(tmp_path / "registry"),
                "model_name": "vicuna-governance",
                "run_root": str(tmp_path / "runs"),
                "limit": 512,
                "timeout_ms": 5000,
                "min_record_count": 1,
                "min_exact_match_rate": 0.5,
                "max_invalid_action_rate": 0.0,
                "min_reward_delta": 0.0,
            },
        )()
        result = nightly_batch(args)
    finally:
        try:
            next(server_cm)
        except StopIteration:
            pass

    assert result["status"] == "passed"
    assert result["promotion_decision"]["decision"] == "promoted"
    assert result["candidate_version"] == 1
    assert Path(result["batch_run_manifest_path"]).exists()


def test_nightly_batch_rejects_alias_when_threshold_fails(tmp_path: Path):
    server_cm = with_server()
    server = next(server_cm)
    try:
        args = type(
            "Args",
            (),
            {
                "server": f"http://127.0.0.1:{server.server_port}",
                "dataset_dir": str(tmp_path / "dataset"),
                "dataset_id": "vicuna-nightly-v1",
                "registry_dir": str(tmp_path / "registry"),
                "model_name": "vicuna-governance",
                "run_root": str(tmp_path / "runs"),
                "limit": 512,
                "timeout_ms": 5000,
                "min_record_count": 1,
                "min_exact_match_rate": 1.1,
                "max_invalid_action_rate": 0.0,
                "min_reward_delta": 0.0,
            },
        )()
        result = nightly_batch(args)
    finally:
        try:
            next(server_cm)
        except StopIteration:
            pass

    assert result["status"] == "passed"
    assert result["promotion_decision"]["decision"] == "rejected"
    assert result["candidate_version"] == 1
