import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import pytest

from utils import ServerProcess

server: ServerProcess


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_provider_server(base_url: str, extra_env: Optional[Dict[str, str]] = None) -> ServerProcess:
    process = ServerProcess()
    process.model_hf_repo = None
    process.model_hf_file = None
    process.model_alias = "deepseek-chat"
    process.no_webui = True
    process.server_port = reserve_port()
    process.extra_env = {
        "VICUNA_DEEPSEEK_API_KEY": "test-key",
        "VICUNA_DEEPSEEK_BASE_URL": base_url,
        "VICUNA_DEEPSEEK_MODEL": "deepseek-chat",
        "VICUNA_DEEPSEEK_TIMEOUT_MS": "5000",
    }
    if extra_env:
        process.extra_env.update(extra_env)
    return process


def make_runpod_host_server(base_url: str, auth_token: str, extra_env: Optional[Dict[str, str]] = None) -> ServerProcess:
    process = ServerProcess()
    process.model_hf_repo = None
    process.model_hf_file = None
    process.model_alias = "runpod-relay"
    process.no_webui = True
    process.server_port = reserve_port()
    process.extra_env = {
        "VICUNA_HOST_INFERENCE_MODE": "experimental",
        "VICUNA_RUNPOD_INFERENCE_ROLE": "host",
        "VICUNA_RUNPOD_INFERENCE_URL": base_url,
        "VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN": auth_token,
        "VICUNA_RUNPOD_INFERENCE_MODEL": "google/gemma-4-31B-it",
        "VICUNA_RUNPOD_SERVING_DTYPE": "q8_0",
        "VICUNA_RUNPOD_KV_PROFILE": "q8_0_64k",
        "VICUNA_RUNPOD_CONTEXT_LIMIT": "65536",
    }
    if extra_env:
        process.extra_env.update(extra_env)
    return process


@contextmanager
def run_mock_deepseek(response_factory=None, route_factory=None):
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            state["requests"].append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": {},
            })
            if not route_factory:
                self.send_response(404)
                self.end_headers()
                return
            route_response = route_factory(self.path, {}, state)
            if route_response is None:
                self.send_response(404)
                self.end_headers()
                return
            encoded = json.dumps(route_response.get("body", {})).encode("utf-8")
            self.send_response(route_response.get("status", 200))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            state["requests"].append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": payload,
            })

            if self.path not in ("/chat/completions", "/beta/chat/completions"):
                if not route_factory:
                    self.send_response(404)
                    self.end_headers()
                    return
                route_response = route_factory(self.path, payload, state)
                if route_response is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                encoded = json.dumps(route_response.get("body", {})).encode("utf-8")
                self.send_response(route_response.get("status", 200))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            response = response_factory(payload) if response_factory else {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello from DeepSeek.",
                        "reasoning_content": "Keep the reply brief.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 11,
                    "total_tokens": 16,
                },
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    finally:
        httpd.shutdown()
        thread.join()


def requests_for_path(state, path: str):
    return [request for request in state["requests"] if request["path"] == path]


def outbound_non_system_messages(payload):
    return [
        message
        for message in payload.get("messages", [])
        if message.get("role") != "system"
    ]


def minimal_emotive_trace(trace_id: str):
    return {
        "trace_id": trace_id,
        "model": "google/gemma-4-31B-it",
        "embedding_mode": "runtime_only",
        "estimator_version": "test",
        "provider_streamed": False,
        "retained_block_count": 0,
        "current_turn_block_count": 0,
        "cognitive_replay": False,
        "suppress_replay_admission": False,
        "mode": "foreground",
        "final_moment": {
            "epistemic_pressure": 0.2,
            "confidence": 0.8,
            "contradiction_pressure": 0.1,
            "planning_clarity": 0.8,
            "curiosity": 0.4,
            "caution": 0.3,
            "frustration": 0.1,
            "satisfaction": 0.8,
            "momentum": 0.6,
            "stall": 0.1,
            "semantic_novelty": 0.3,
            "user_alignment": 0.9,
            "runtime_trust": 0.8,
            "runtime_failure_pressure": 0.0,
        },
        "final_vad": {
            "valence": 0.5,
            "arousal": 0.1,
            "dominance": 0.7,
        },
        "final_policy": {
            "policy_version": "control_surface_v2",
            "selected_mode": "reflective",
            "reasoning_depth": "medium",
            "thinking_mode": "enabled",
            "response_budget_bucket": 512,
            "reasoning_budget_bucket": 512,
            "applied_provider_controls": {
                "thinking_enabled": True,
            },
        },
        "blocks": [],
    }


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerProcess()


def test_provider_mode_health_and_models_exclude_local_runtime_surfaces():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
        try:
            server.start()

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["provider"]["name"] == "deepseek"
            assert health.body["provider"]["model"] == "deepseek-chat"

            models = server.make_request("GET", "/v1/models")
            assert models.status_code == 200
            assert models.body["data"][0]["id"] == "deepseek-chat"
        finally:
            server.stop()


def test_runpod_host_mode_relays_inference_without_local_provider_credentials():
    global server

    def route_factory(path, payload, state):
        if path == "/health":
            return {"status": 200, "body": {"status": "ok", "state": "ready"}}
        if path != "/v1/chat/completions":
            return None
        return {
            "status": 200,
            "body": {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello from the RunPod relay.",
                        "reasoning_content": "Forwarding the request to the remote node.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 7,
                    "total_tokens": 16,
                },
                "vicuna_emotive_trace": {
                    "trace_id": "relay-trace-1",
                    "blocks": [],
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_runpod_host_server(base_url, "runpod-test-token", {
            "VICUNA_POLICY_MODE": "capture",
            "VICUNA_POLICY_MAX_TRANSITIONS": "32",
        })
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Say hello from relay mode."}],
            })
            assert response.status_code == 200
            assert response.body["choices"][0]["message"]["content"] == "Hello from the RunPod relay."

            relay_requests = requests_for_path(state, "/v1/chat/completions")
            assert len(relay_requests) == 1
            assert relay_requests[0]["headers"]["Authorization"] == "Bearer runpod-test-token"
            assert relay_requests[0]["body"]["max_tokens"] == 4096
            assert "reasoning_budget_tokens" not in relay_requests[0]["body"]
            assert "thinking" not in relay_requests[0]["body"]
            forwarded_messages = outbound_non_system_messages(relay_requests[0]["body"])
            assert forwarded_messages[0]["content"] == "Say hello from relay mode."
        finally:
            server.stop()


def test_runpod_host_mode_retries_reasoning_only_completion_with_thinking_disabled():
    global server

    def route_factory(path, payload, state):
        if path == "/health":
            return {"status": 200, "body": {"status": "ok"}}
        if path != "/v1/chat/completions":
            return None
        call_index = len([item for item in state["requests"] if item["path"] == "/v1/chat/completions"])
        if call_index == 1:
            assert payload["max_tokens"] == 4096
            assert payload.get("chat_template_kwargs", {}).get("enable_thinking", True) is True
            return {
                "status": 200,
                "body": {
                    "choices": [{
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Need to think longer before answering.",
                        },
                    }],
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 4096,
                        "total_tokens": 4138,
                    },
                },
            }
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        return {
            "status": 200,
            "body": {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Visible answer after retry.",
                        "reasoning_content": "",
                    },
                }],
                "usage": {
                    "prompt_tokens": 42,
                    "completion_tokens": 33,
                    "total_tokens": 75,
                },
                "vicuna_emotive_trace": minimal_emotive_trace("emo-retry"),
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_runpod_host_server(base_url, "relay-token")
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Give me a visible answer."}],
            })
            assert response.status_code == 200
            assert response.body["choices"][0]["message"]["content"] == "Visible answer after retry."

            relay_requests = requests_for_path(state, "/v1/chat/completions")
            assert len(relay_requests) == 2
            assert relay_requests[1]["body"]["chat_template_kwargs"]["enable_thinking"] is False
        finally:
            server.stop()


def test_runpod_host_mode_recovers_mistral_tool_calls_from_assistant_content():
    global server

    def route_factory(path, payload, state):
        if path == "/health":
            return {"status": 200, "body": {"status": "ok"}}
        if path != "/v1/chat/completions":
            return None
        return {
            "status": 200,
            "body": {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "[TOOL_CALLS]web_search[CALL_ID]abc123XYZ[ARGS]{\"query\":\"Mistral Small 3.2 llama.cpp guidance\"}",
                        "reasoning_content": "",
                    },
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 12,
                    "total_tokens": 23,
                },
                "vicuna_emotive_trace": minimal_emotive_trace("emo-mistral"),
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, _):
        server = make_runpod_host_server(base_url, "relay-token")
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Search for current guidance."}],
            })
            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            tool_call = choice["message"]["tool_calls"][0]
            assert tool_call["id"] == "abc123XYZ"
            assert tool_call["function"]["name"] == "web_search"
            assert json.loads(tool_call["function"]["arguments"]) == {
                "query": "Mistral Small 3.2 llama.cpp guidance",
            }
        finally:
            server.stop()


def test_runpod_host_mode_recovers_gemma_json_tool_calls_from_assistant_content():
    global server

    def route_factory(path, payload, state):
        if path == "/health":
            return {"status": 200, "body": {"status": "ok"}}
        if path != "/v1/chat/completions":
            return None
        return {
            "status": 200,
            "body": {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "```json\n{\"name\":\"web_search\",\"parameters\":{\"query\":\"Gemma 4 llama.cpp tool calling\"}}\n```",
                        "reasoning_content": "",
                    },
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 12,
                    "total_tokens": 23,
                },
                "vicuna_emotive_trace": minimal_emotive_trace("emo-gemma-json"),
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, _):
        server = make_runpod_host_server(base_url, "relay-token", {
            "VICUNA_RUNPOD_INFERENCE_MODEL": "google/gemma-4-31B-it",
        })
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Search for Gemma tool calling guidance."}],
            })
            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "web_search"
            assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
                "query": "Gemma 4 llama.cpp tool calling",
            }
        finally:
            server.stop()


def test_runpod_host_mode_recovers_gemma_python_tool_calls_from_assistant_content():
    global server

    def route_factory(path, payload, state):
        if path == "/health":
            return {"status": 200, "body": {"status": "ok"}}
        if path != "/v1/chat/completions":
            return None
        return {
            "status": 200,
            "body": {
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "[web_search(query=\"Gemma 4 llama.cpp tool calling\"), hard_memory_read(key='gemma-tooling')]",
                        "reasoning_content": "",
                    },
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 12,
                    "total_tokens": 23,
                },
                "vicuna_emotive_trace": minimal_emotive_trace("emo-gemma-python"),
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, _):
        server = make_runpod_host_server(base_url, "relay-token", {
            "VICUNA_RUNPOD_INFERENCE_MODEL": "google/gemma-4-31B-it",
        })
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Use tools if needed."}],
            })
            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert len(choice["message"]["tool_calls"]) == 2
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "web_search"
            assert choice["message"]["tool_calls"][1]["function"]["name"] == "hard_memory_read"
        finally:
            server.stop()


def test_standard_deepseek_mode_is_detached_from_experimental_capture():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url, {
            "VICUNA_HOST_INFERENCE_MODE": "standard",
            "VICUNA_POLICY_MODE": "capture",
            "VICUNA_POLICY_MAX_TRANSITIONS": "32",
        })
        try:
            server.start()

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["provider"]["name"] == "deepseek"
            assert health.body["provider"]["mode"] == "deepseek"
            assert health.body["runpod_inference"]["host_inference_mode"] == "standard"

            response = server.make_request("POST", "/v1/chat/completions", data={
                "messages": [{"role": "user", "content": "Reply briefly from standard mode."}],
            })
            assert response.status_code == 200
            assert response.body["choices"][0]["message"]["content"] == "Hello from DeepSeek."
            assert "vicuna_emotive_trace" not in response.body

            transitions = server.make_request("GET", "/v1/policy/transitions?request_id=&limit=5")
            assert transitions.status_code == 200
            assert transitions.body["count"] == 0
            assert transitions.body["items"] == []
        finally:
            server.stop()
