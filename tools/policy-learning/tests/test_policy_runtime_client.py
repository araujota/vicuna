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

from policy_runtime_client import PolicyRuntimeClient


class _Handler(BaseHTTPRequestHandler):
    last_post_payload: dict | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/policy/status"):
            payload = {"ok": True, "mode": "capture"}
        elif self.path.startswith("/v1/policy/transitions"):
            payload = {"object": "vicuna.policy.transitions", "count": 1}
        elif self.path.startswith("/v1/policy/decode-traces"):
            payload = {"object": "vicuna.policy.decode_traces", "count": 2}
        elif self.path.startswith("/v1/policy/runtime-artifacts"):
            payload = {"object": "vicuna.policy.runtime_artifacts", "items": {}}
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

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/policy/runtime-artifacts":
            self.send_response(404)
            self.end_headers()
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        _Handler.last_post_payload = json.loads(raw)
        body = json.dumps({"ok": True, "applied": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def test_policy_runtime_client_fetches_decode_traces() -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyRuntimeClient(f"http://127.0.0.1:{server.server_address[1]}")
        status = client.get_status()
        transitions = client.get_transitions(limit=5, request_id="req-1")
        decode_traces = client.get_decode_traces(limit=3, request_id="req-1")
        runtime_artifacts = client.get_runtime_artifacts()
        applied = client.apply_runtime_artifact({"artifact_kind": "decode_controller", "slot": "candidate", "clear": True})

        assert status["mode"] == "capture"
        assert transitions["count"] == 1
        assert decode_traces["object"] == "vicuna.policy.decode_traces"
        assert decode_traces["count"] == 2
        assert runtime_artifacts["object"] == "vicuna.policy.runtime_artifacts"
        assert applied["applied"] is True
        assert _Handler.last_post_payload == {"artifact_kind": "decode_controller", "slot": "candidate", "clear": True}
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
