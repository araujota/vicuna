import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import pytest

from utils import ServerProcess

server: ServerProcess


def make_provider_server(base_url: str, extra_env: Optional[Dict[str, str]] = None) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_alias = "deepseek-chat"
    server.no_webui = True
    server.extra_env = {
        "VICUNA_DEEPSEEK_API_KEY": "test-key",
        "VICUNA_DEEPSEEK_BASE_URL": base_url,
        "VICUNA_DEEPSEEK_MODEL": "deepseek-chat",
        "VICUNA_DEEPSEEK_TIMEOUT_MS": "5000",
    }
    if extra_env:
        server.extra_env.update(extra_env)
    return server


def make_staged_provider_server(base_url: str, extra_env: Optional[Dict[str, str]] = None) -> ServerProcess:
    merged = {"VICUNA_ENABLE_STAGED_TOOL_FALLBACK": "1"}
    if extra_env:
        merged.update(extra_env)
    return make_provider_server(base_url, merged)


def wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@contextmanager
def run_mock_deepseek(response_factory=None, route_factory=None):
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            state["requests"].append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": payload,
                "client": {
                    "host": self.client_address[0],
                    "port": self.client_address[1],
                },
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
                        "reasoning_content": "I should greet the user and keep the reply brief.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 11,
                    "total_tokens": 16,
                },
            }
            if payload.get("stream") is True:
                message = response["choices"][0]["message"]
                finish_reason = response["choices"][0].get("finish_reason", "stop")
                events = []

                if message.get("reasoning_content"):
                    events.append({
                        "id": response.get("id", "chatcmpl-mock"),
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": message["reasoning_content"],
                            },
                        }],
                    })
                else:
                    events.append({
                        "id": response.get("id", "chatcmpl-mock"),
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                        }],
                    })

                if message.get("content"):
                    events.append({
                        "id": response.get("id", "chatcmpl-mock"),
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": message["content"],
                            },
                        }],
                    })

                for index, tool_call in enumerate(message.get("tool_calls", [])):
                    events.append({
                        "id": response.get("id", "chatcmpl-mock"),
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "id": tool_call["id"],
                                    "type": tool_call.get("type", "function"),
                                    "function": tool_call["function"],
                                }],
                            },
                        }],
                    })

                events.append({
                    "id": response.get("id", "chatcmpl-mock"),
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                    "usage": response.get("usage", {
                        "prompt_tokens": 5,
                        "completion_tokens": 11,
                        "total_tokens": 16,
                    }),
                })

                body_parts = [f"data: {json.dumps(event)}\n\n" for event in events]
                body_parts.append("data: [DONE]\n\n")
                encoded = "".join(body_parts).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

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


def outbound_system_messages(payload):
    return [
        message["content"]
        for message in payload.get("messages", [])
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ]


def outbound_non_system_messages(payload):
    return [
        message
        for message in payload.get("messages", [])
        if message.get("role") != "system"
    ]


def assert_has_skill_memory_indexes(payload, skill_name: Optional[str] = None, memory_name: Optional[str] = None):
    system_messages = outbound_system_messages(payload)
    index_message = next((message for message in system_messages if "SKILLS:\n" in message and "\n\nMEMORIES:\n" in message), None)
    assert index_message is not None
    assert "Read a skill or memory file explicitly when you need its contents." in index_message
    assert "skill_create may only be used when the user directly asks to create or update a skill in this conversation." in index_message
    if skill_name:
        assert skill_name in index_message
    if memory_name:
        assert memory_name in index_message


def assert_has_policy_guidance(payload):
    system_messages = outbound_system_messages(payload)
    assert any(message.startswith("Metacognitive control policy:") for message in system_messages)


def rich_plan_text(body: str, *, format: str = "plain_text", extra_meta: Optional[Dict[str, object]] = None) -> str:
    meta = {"format": format}
    if extra_meta:
        meta.update(extra_meta)
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", body])
    return "\n".join(lines)


def selector_tool_name(payload):
    tools = payload.get("tools", [])
    if not tools:
        return ""
    return tools[0]["function"]["name"]


def selector_tool_names(payload):
    return [tool["function"]["name"] for tool in payload.get("tools", [])]


def selector_choice_response(response_id: str, reasoning: str, tool_name: str, arguments: dict):
    return {
        "id": response_id,
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"{response_id}_call",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }],
                "reasoning_content": reasoning,
            },
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
        },
    }


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerProcess()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # Override the shared preset preloader because provider-mode tests do not
    # require a local model cache and intentionally start without a model.
    yield


def test_provider_mode_health_and_models_exclude_local_runtime_surfaces():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
        try:
            server.start()

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["status"] == "ok"
            assert health.body["state"] == "ready"
            assert health.body["provider"]["name"] == "deepseek"
            assert health.body["provider"]["model"] == "deepseek-chat"
            assert health.body["provider"]["transport"] == {
                "shared_client_cached": False,
                "client_builds": 0,
                "request_count": 0,
                "reused_request_count": 0,
                "base_url": None,
            }
            assert "runtime_persistence" not in health.body
            assert health.body["proactive_mailbox"]["live_stream_connected"] is False
            assert health.body["bridge_runtime"]["telegram_runtime_tool_cache"]["cached"] is False
            assert health.body["bridge_runtime"]["staged_prompt_cache"]["cached"] is False
            assert health.body["request_traces"] == {
                "stored_events": 0,
                "max_events": 512,
                "total_events": 0,
                "latest_request_id": None,
                "latest_event": None,
            }
            assert health.body["emotive_runtime"]["enabled"] is True
            assert health.body["emotive_runtime"]["embedding_backend"]["mode"] == "lexical_only"
            assert health.body["emotive_runtime"]["cognitive_replay"]["enabled"] is True

            models = server.make_request("GET", "/v1/models")
            assert models.status_code == 200
            assert models.body["data"][0]["id"] == "deepseek-chat"
            assert models.body["data"][0]["owned_by"] == "deepseek"

            lora = server.make_request("GET", "/lora-adapters")
            assert lora.status_code == 404
        finally:
            server.stop()


def test_provider_mode_chat_completion_exposes_reasoning_trace():
    global server
    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "temperature": 0.9,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            })
            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["message"]["content"] == "Hello from DeepSeek."
            assert choice["message"]["reasoning_content"] == "I should greet the user and keep the reply brief."
            assert choice["finish_reason"] == "stop"
            assert response.body["model"] == "deepseek-chat"
            trace = response.body["vicuna_emotive_trace"]
            assert trace["embedding_mode"] == "lexical_only"
            assert trace["live_generation_start_block_index"] == 1
            assert trace["final_policy"]["policy_version"] == "control_surface_v2"
            assert trace["heuristic_retrieval"]["matched"] is False
            assert [block["source"]["kind"] for block in trace["blocks"]] == [
                "user_message",
                "assistant_reasoning",
                "assistant_content",
                "runtime_event",
            ]
            assert trace["blocks"][0]["delta"]["negative_mass"] == 0.0
            assert trace["blocks"][1]["text"] == "I should greet the user and keep the reply brief."
            assert trace["blocks"][2]["text"] == "Hello from DeepSeek."
            vad = trace["blocks"][1]["vad"]
            assert "trend" in vad
            assert "labels" in vad and vad["labels"]
            assert "dominant_dimensions" in vad and vad["dominant_dimensions"]
            assert "style_guide" in vad
            assert vad["style_guide"]["tone_label"]
            assert vad["style_guide"]["prompt_hints"]
            assert 0.0 <= vad["style_guide"]["warmth"] <= 1.0
            assert 0.0 <= vad["style_guide"]["energy"] <= 1.0
            assert 0.0 <= vad["style_guide"]["assertiveness"] <= 1.0

            assert state["requests"]
            outbound = state["requests"][0]
            assert outbound["path"] == "/chat/completions"
            assert outbound["body"]["model"] == "deepseek-chat"
            assert outbound["body"]["thinking"] == {"type": "enabled"}
            assert "temperature" not in outbound["body"]
            assert outbound["body"]["stream"] is True
            assert_has_skill_memory_indexes(outbound["body"])
            assert outbound_non_system_messages(outbound["body"])[0] == {"role": "user", "content": "Hello"}
            assert_has_policy_guidance(outbound["body"])

            latest = server.make_request("GET", "/v1/emotive/trace/latest")
            assert latest.status_code == 200
            assert latest.body["trace"]["trace_id"] == trace["trace_id"]
            assert latest.body["trace"]["final_policy"]["policy_version"] == "control_surface_v2"
            assert latest.body["retained_turns"] == 1
        finally:
            server.stop()


def test_provider_mode_responses_route_emits_reasoning_item():
    global server
    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/responses", data={
                "model": "deepseek-reasoner",
                "max_output_tokens": 4096,
                "input": [
                    {"role": "user", "content": "Hello"},
                ],
            })
            assert response.status_code == 200
            assert response.body["object"] == "response"
            assert response.body["output"][0]["type"] == "reasoning"
            assert response.body["output"][0]["content"][0]["text"] == "I should greet the user and keep the reply brief."
            assert response.body["output"][1]["type"] == "message"
            assert response.body["output"][1]["content"][0]["text"] == "Hello from DeepSeek."
            assert response.body["vicuna_emotive_trace"]["blocks"][0]["source"]["kind"] == "user_message"
            assert response.body["vicuna_emotive_trace"]["final_policy"]["policy_version"] == "control_surface_v2"
            assert response.body["vicuna_emotive_trace"]["final_vad"]["style_guide"]["tone_label"]
            assert state["requests"][0]["body"]["max_tokens"] == 768
            assert "temperature" not in state["requests"][0]["body"]
            assert_has_policy_guidance(state["requests"][0]["body"])
        finally:
            server.stop()


def test_provider_mode_reuses_shared_deepseek_http_client():
    global server
    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            for prompt in ("First", "Second"):
                response = server.make_request("POST", "/v1/chat/completions", data={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                })
                assert response.status_code == 200

            health = server.make_request("GET", "/health")
            transport = health.body["provider"]["transport"]
            assert transport["shared_client_cached"] is True
            assert transport["client_builds"] == 1
            assert transport["request_count"] == 2
            assert transport["reused_request_count"] >= 1

            provider_requests = requests_for_path(state, "/chat/completions")
            assert len(provider_requests) == 2
            assert all(request["body"]["temperature"] == 0.2 for request in provider_requests)
            assert {request["client"]["port"] for request in provider_requests} == {
                provider_requests[0]["client"]["port"],
            }
        finally:
            server.stop()


def test_provider_mode_request_trace_endpoint_returns_correlated_foreground_events():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Trace this request."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_fg_1",
            })
            assert response.status_code == 200
            assert response.headers["x-client-request-id"] == "trace_fg_1"
            assert response.headers["x-request-id"]

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_fg_1&limit=20")
            assert traces.status_code == 200
            assert traces.body["object"] == "vicuna.request_traces"
            assert traces.body["request_id"] == "trace_fg_1"
            events = [item["event"] for item in traces.body["items"]]
            assert events[0] == "request_received"
            assert "policy_computed" in events
            assert "skill_memory_indexes_injected" in events
            assert "guidance_evaluated" in events
            assert "provider_controls_applied" in events
            assert "provider_request_started" in events
            assert "provider_request_finished" in events
            assert "request_completed" in events
            assert all(item["request_id"] == "trace_fg_1" for item in traces.body["items"])
            by_event = {item["event"]: item for item in traces.body["items"]}
            assert by_event["request_received"]["component"] == "runtime"
            assert by_event["policy_computed"]["component"] == "control_policy"
            assert by_event["policy_computed"]["data"]["policy_version"] == "control_surface_v2"
            assert by_event["guidance_evaluated"]["component"] == "runtime_guidance"
            assert by_event["guidance_evaluated"]["data"]["vad_skip_reason"] == "no_active_tool_continuation_span"
            assert by_event["guidance_evaluated"]["data"]["policy_injected"] is True
            assert by_event["skill_memory_indexes_injected"]["component"] == "prompt_context"
            assert by_event["provider_controls_applied"]["component"] == "policy_runtime"
            assert by_event["provider_controls_applied"]["data"]["applied_provider_controls"]["thinking_enabled"] is True
            assert by_event["provider_controls_applied"]["data"]["applied_provider_controls"]["temperature"] is None
            assert by_event["provider_request_started"]["component"] == "provider"
            assert by_event["provider_request_started"]["data"]["max_tokens"] == 512
            assert by_event["provider_request_started"]["data"]["temperature"] is None
            assert any(
                message.startswith("Metacognitive control policy:")
                for message in by_event["provider_request_started"]["data"]["system_messages"]
            )
            assert any(
                "SKILLS:\n" in message and "\n\nMEMORIES:\n" in message
                for message in by_event["provider_request_started"]["data"]["system_messages"]
            )
            assert by_event["provider_request_finished"]["data"]["finish_reason"] == "stop"
            assert by_event["provider_request_finished"]["data"]["reasoning_content"] == "I should greet the user and keep the reply brief."
            assert by_event["provider_request_finished"]["data"]["content"] == "Hello from DeepSeek."

            health = server.make_request("GET", "/health")
            assert health.body["request_traces"]["stored_events"] >= 4
            assert health.body["request_traces"]["latest_request_id"] == "trace_fg_1"
            assert health.body["request_traces"]["latest_event"] == "request_completed"
        finally:
            server.stop()


def test_provider_mode_policy_transition_capture_and_export():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "capture",
            "VICUNA_POLICY_MAX_TRANSITIONS": "8",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Capture this governance decision."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_capture_1",
            })
            assert response.status_code == 200

            status = server.make_request("GET", "/v1/policy/status")
            assert status.status_code == 200
            assert status.body["object"] == "vicuna.policy.status"
            assert status.body["enabled"] is True
            assert status.body["mode"] == "capture"
            assert status.body["stored_transitions"] == 1
            assert status.body["total_transitions"] == 1
            assert status.body["candidate_policy_version"] is None
            assert status.body["reward_model"]["model_version"] == "desired_state_reward_v1"
            assert status.body["reward_model"]["target_moment"]["confidence"] > 0.0
            assert status.body["reward_model"]["target_vad"]["valence"] > 0.0

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["policy_runtime"]["enabled"] is True
            assert health.body["policy_runtime"]["mode"] == "capture"
            assert health.body["policy_runtime"]["stored_transitions"] == 1
            assert (
                health.body["policy_runtime"]["reward_model"]["model_version"]
                == "desired_state_reward_v1"
            )

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_capture_1&limit=5",
            )
            assert exported.status_code == 200
            assert exported.body["object"] == "vicuna.policy.transitions"
            assert exported.body["request_id"] == "policy_capture_1"
            assert exported.body["count"] == 1
            item = exported.body["items"][0]
            assert item["observation"]["schema_version"] == "policy_observation_v1"
            assert item["executed_action"]["schema_version"] == "policy_action_v2"
            assert item["executed_action"]["proposal_source"] == "native"
            assert item["applied_provider_controls"]["thinking_enabled"] in (True, False)
            assert item["candidate_action"] is None
            assert item["reward_model"]["model_version"] == "desired_state_reward_v1"
            assert item["reward_breakdown"]["schema_version"] == "policy_reward_breakdown_v1"
            assert 0.0 <= item["reward_breakdown"]["before_score"] <= 1.0
            assert 0.0 <= item["reward_breakdown"]["after_score"] <= 1.0
            assert item["reward_events"]
            assert item["termination_reason"] == "stop"
            assert item["observation"]["request_id"] == "policy_capture_1"
            assert item["next_observation"]["trace_id"] == response.body["vicuna_emotive_trace"]["trace_id"]
        finally:
            server.stop()


def test_provider_mode_invalid_reward_config_fails_startup(tmp_path: Path):
    global server
    reward_config_path = tmp_path / "invalid-reward-config.json"
    reward_config_path.write_text(
        json.dumps(
            {
                "target_moment": {
                    "confidence": 1.2,
                }
            }
        ),
        encoding="utf-8",
    )

    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "capture",
            "VICUNA_POLICY_REWARD_CONFIG_PATH": str(reward_config_path),
        })
        server.api_surface = "openai"
        with pytest.raises(RuntimeError, match="return code 1"):
            server.start(timeout_seconds=3)


def test_provider_mode_shadow_policy_records_disagreement_without_overriding_native_action():
    global server

    def route_factory(path, payload, _state):
        if path != "/v1/policy/propose":
            return None
        assert payload["policy_mode"] == "shadow"
        return {
            "status": 200,
            "body": {
                "policy_version": "shadow_rule_v1",
                "action": {
                    "selected_mode": "tool_heavy",
                    "reasoning_depth": "deep",
                    "token_budget_bucket": 2048,
                    "tool_parallelism_cap": 2,
                    "interrupt_allowed": True,
                    "replan_required": True,
                    "early_stop_ok": False,
                    "force_synthesis": False,
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "shadow",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Shadow this policy decision."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_shadow_1",
            })
            assert response.status_code == 200

            status = server.make_request("GET", "/v1/policy/status")
            assert status.status_code == 200
            assert status.body["mode"] == "shadow"
            assert status.body["shadow_request_count"] == 1
            assert status.body["shadow_disagreement_count"] == 1
            assert status.body["candidate_failure_count"] == 0
            assert status.body["candidate_policy_version"] == "shadow_rule_v1"

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_shadow_1&limit=5",
            )
            assert exported.status_code == 200
            item = exported.body["items"][0]
            assert item["candidate_policy_version"] == "shadow_rule_v1"
            assert item["candidate_action"]["proposal_source"] == "shadow_candidate"
            assert item["executed_action"]["proposal_source"] == "native"
            assert item["candidate_action"]["selected_mode"] == "tool_heavy"
            assert item["executed_action"]["reasoning_depth"] != (
                item["candidate_action"]["reasoning_depth"]
            )
            assert item["executed_action"]["tool_parallelism_cap"] != (
                item["candidate_action"]["tool_parallelism_cap"]
            )

            policy_requests = requests_for_path(state, "/v1/policy/propose")
            assert len(policy_requests) == 1

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=policy_shadow_1&limit=20")
            policy_events = [
                item for item in traces.body["items"]
                if item["component"] == "policy_runtime" and item["event"] == "candidate_evaluated"
            ]
            assert len(policy_events) == 1
            assert policy_events[0]["data"]["candidate_failure"] is False
            assert policy_events[0]["data"]["disagrees_with_native"] is True
        finally:
            server.stop()


def test_provider_mode_shadow_policy_failure_falls_back_and_updates_status():
    global server

    def route_factory(path, _payload, _state):
        if path != "/v1/policy/propose":
            return None
        return {
            "status": 503,
            "body": {
                "error": "temporary failure",
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, _state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "shadow",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Force candidate failure."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_shadow_fail_1",
            })
            assert response.status_code == 200

            status = server.make_request("GET", "/v1/policy/status")
            assert status.status_code == 200
            assert status.body["mode"] == "shadow"
            assert status.body["shadow_request_count"] == 1
            assert status.body["candidate_failure_count"] == 1
            assert status.body["last_candidate_error"]

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_shadow_fail_1&limit=5",
            )
            assert exported.status_code == 200
            item = exported.body["items"][0]
            assert item["candidate_action"] is None
            assert item["executed_action"]["proposal_source"] == "native"
            assert item["safety_guard"]["candidate_present"] is False
            assert item["safety_guard"]["fallback_to_native"] is True
            assert item["safety_guard"]["reason"]
        finally:
            server.stop()


def test_provider_mode_canary_live_executes_candidate_action_and_updates_status():
    global server

    def route_factory(path, payload, _state):
        if path != "/v1/policy/propose":
            return None
        assert payload["policy_mode"] == "canary_live"
        return {
            "status": 200,
            "body": {
                "policy_version": "candidate_canary_v1",
                "policy_alias": "candidate",
                "confidence": {
                    "overall": 0.95,
                    "by_head": {
                        "selected_mode": 0.95,
                        "reasoning_depth": 0.90,
                    },
                    "feature_signature_seen": True,
                },
                "action": {
                    "selected_mode": "tool_heavy",
                    "reasoning_depth": "deep",
                    "token_budget_bucket": 2048,
                    "tool_parallelism_cap": 2,
                    "interrupt_allowed": True,
                    "replan_required": True,
                    "early_stop_ok": False,
                    "force_synthesis": False,
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "canary_live",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_CANARY_STEPS": "100",
            "VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP": "1",
            "VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD": "0.70",
            "VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE": "1.0",
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Use the learned canary policy."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_canary_live_1",
            })
            assert response.status_code == 200

            status = server.make_request("GET", "/v1/policy/status")
            assert status.status_code == 200
            assert status.body["mode"] == "canary_live"
            assert status.body["rollout_state"] == "completed"
            assert status.body["candidate_policy_version"] == "candidate_canary_v1"
            assert status.body["candidate_policy_alias"] == "candidate"
            assert status.body["sampled_request_count"] == 1
            assert status.body["live_candidate_execution_count"] == 1
            assert status.body["current_canary_share_percent"] == 100

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_canary_live_1&limit=5",
            )
            assert exported.status_code == 200
            item = exported.body["items"][0]
            assert item["rollout_mode"] == "canary_live"
            assert item["rollout_sampled"] is True
            assert item["candidate_executed_live"] is True
            assert item["candidate_policy_alias"] == "candidate"
            assert item["candidate_confidence"] == pytest.approx(0.95, rel=1e-3)
            assert item["candidate_confidence_passed"] is True
            assert item["rollout_decision_reason"] == "candidate_live"
            assert item["executed_action"]["proposal_source"] == "rollout_candidate"
            assert item["executed_action"]["selected_mode"] == "tool_heavy"
            assert item["executed_action"]["reasoning_depth"] == "deep"

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=policy_canary_live_1&limit=20")
            provider_events = [
                item for item in traces.body["items"]
                if item["component"] == "provider" and item["event"] == "provider_request_started"
            ]
            assert len(provider_events) == 1
            assert provider_events[0]["data"]["max_tokens"] == 2048
            assert any(
                message.startswith("Metacognitive control policy: mode=tool_heavy")
                for message in provider_events[0]["data"]["system_messages"]
            )

            policy_requests = requests_for_path(state, "/v1/policy/propose")
            assert len(policy_requests) == 1
        finally:
            server.stop()


def test_provider_mode_canary_live_rolls_back_after_low_confidence_fallback():
    global server

    def route_factory(path, payload, _state):
        if path != "/v1/policy/propose":
            return None
        assert payload["policy_mode"] == "canary_live"
        return {
            "status": 200,
            "body": {
                "policy_version": "candidate_low_conf_v1",
                "policy_alias": "candidate",
                "confidence": {
                    "overall": 0.20,
                    "by_head": {
                        "selected_mode": 0.20,
                    },
                    "feature_signature_seen": True,
                },
                "action": {
                    "selected_mode": "tool_heavy",
                    "reasoning_depth": "deep",
                    "token_budget_bucket": 2048,
                    "tool_parallelism_cap": 2,
                    "interrupt_allowed": True,
                    "replan_required": True,
                    "early_stop_ok": False,
                    "force_synthesis": False,
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "canary_live",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_CANARY_STEPS": "100",
            "VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP": "1",
            "VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD": "0.90",
            "VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE": "0.50",
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            first = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "First request should trigger rollback."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_canary_rb_1",
            })
            assert first.status_code == 200

            second = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Second request should stay native after rollback."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_canary_rb_2",
            })
            assert second.status_code == 200

            status = server.make_request("GET", "/v1/policy/status")
            assert status.status_code == 200
            assert status.body["mode"] == "canary_live"
            assert status.body["rollout_state"] == "rolled_back"
            assert status.body["rollback_count"] == 1
            assert status.body["last_rollback_reason"] == "fallback_rate_exceeded"
            assert status.body["low_confidence_count"] == 1
            assert status.body["live_candidate_execution_count"] == 0

            first_transition = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_canary_rb_1&limit=5",
            ).body["items"][0]
            assert first_transition["rollout_sampled"] is True
            assert first_transition["candidate_executed_live"] is False
            assert first_transition["candidate_confidence_passed"] is False
            assert first_transition["rollout_decision_reason"] == "low_confidence"
            assert first_transition["executed_action"]["proposal_source"] == "native"

            second_transition = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_canary_rb_2&limit=5",
            ).body["items"][0]
            assert second_transition["candidate_executed_live"] is False
            assert second_transition["rollout_decision_reason"] == "rollback_active"
            assert second_transition["executed_action"]["proposal_source"] == "native"

            policy_requests = requests_for_path(state, "/v1/policy/propose")
            assert len(policy_requests) == 1
        finally:
            server.stop()


def test_provider_mode_canary_live_applies_prefix_stop_and_repetition_profiles():
    global server

    def route_factory(path, payload, _state):
        if path != "/v1/policy/propose":
            return None
        assert payload["policy_mode"] == "canary_live"
        return {
            "status": 200,
            "body": {
                "policy_version": "candidate_profile_v1",
                "policy_alias": "candidate",
                "confidence": {"overall": 0.98},
                "action": {
                    "selected_mode": "direct",
                    "reasoning_depth": "short",
                    "thinking_mode": "disabled",
                    "prefix_profile": "bounded_answer",
                    "stop_profile": "concise_answer",
                    "sampling_profile": "deterministic",
                    "repetition_profile": "anti_stall_soft",
                    "tool_choice_profile": "caller_default",
                    "token_budget_bucket": 512,
                    "tool_parallelism_cap": 0,
                    "interrupt_allowed": False,
                    "replan_required": False,
                    "early_stop_ok": True,
                    "force_synthesis": True,
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "canary_live",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_CANARY_STEPS": "100",
            "VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP": "1",
            "VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD": "0.70",
            "VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE": "1.0",
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Give me the bounded answer now."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_canary_profiles_1",
            })
            assert response.status_code == 200

            beta_requests = requests_for_path(state, "/beta/chat/completions")
            assert len(beta_requests) == 1
            provider_body = beta_requests[0]["body"]
            assert provider_body["thinking"]["type"] == "disabled"
            assert provider_body["temperature"] == 0.0
            assert provider_body["frequency_penalty"] == pytest.approx(0.4, rel=1e-3)
            assert provider_body["presence_penalty"] == pytest.approx(0.1, rel=1e-3)
            assert provider_body["stop"] == ["\n\n"]
            assert provider_body["messages"][-1]["role"] == "assistant"
            assert provider_body["messages"][-1]["prefix"] is True
            assert provider_body["messages"][-1]["content"] == "Best bounded answer:\n"

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_canary_profiles_1&limit=5",
            )
            item = exported.body["items"][0]
            assert item["executed_action"]["thinking_mode"] == "disabled"
            assert item["executed_action"]["prefix_profile"] == "bounded_answer"
            assert item["applied_provider_controls"]["thinking_enabled"] is False
            assert item["applied_provider_controls"]["prefix_used"] is True
            assert item["applied_provider_controls"]["stop_sequences"] == ["\n\n"]
            assert item["applied_provider_controls"]["temperature"] == 0.0
        finally:
            server.stop()


def test_provider_mode_canary_live_suppresses_incompatible_profiles_when_thinking_enabled():
    global server

    def route_factory(path, payload, _state):
        if path != "/v1/policy/propose":
            return None
        assert payload["policy_mode"] == "canary_live"
        return {
            "status": 200,
            "body": {
                "policy_version": "candidate_profile_v2",
                "policy_alias": "candidate",
                "confidence": {"overall": 0.98},
                "action": {
                    "selected_mode": "reflective",
                    "reasoning_depth": "deep",
                    "thinking_mode": "enabled",
                    "prefix_profile": "bounded_answer",
                    "stop_profile": "concise_answer",
                    "sampling_profile": "creative",
                    "repetition_profile": "anti_stall_hard",
                    "tool_choice_profile": "caller_default",
                    "token_budget_bucket": 2048,
                    "tool_parallelism_cap": 0,
                    "interrupt_allowed": False,
                    "replan_required": False,
                    "early_stop_ok": False,
                    "force_synthesis": False,
                },
            },
        }

    with run_mock_deepseek(route_factory=route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_POLICY_MODE": "canary_live",
            "VICUNA_POLICY_CANDIDATE_URL": base_url,
            "VICUNA_POLICY_CANARY_STEPS": "100",
            "VICUNA_POLICY_CANARY_MIN_REQUESTS_PER_STEP": "1",
            "VICUNA_POLICY_LIVE_CONFIDENCE_THRESHOLD": "0.70",
            "VICUNA_POLICY_ROLLBACK_MAX_CANDIDATE_FAILURE_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_INVALID_ACTION_RATE": "1.0",
            "VICUNA_POLICY_ROLLBACK_MAX_FALLBACK_RATE": "1.0",
            "VICUNA_POLICY_TIMEOUT_MS": "500",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Think carefully before answering."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "policy_canary_profiles_2",
            })
            assert response.status_code == 200

            normal_requests = requests_for_path(state, "/chat/completions")
            assert len(normal_requests) == 1
            provider_body = normal_requests[0]["body"]
            assert provider_body["thinking"]["type"] == "enabled"
            assert "temperature" not in provider_body
            assert "top_p" not in provider_body
            assert "frequency_penalty" not in provider_body
            assert "presence_penalty" not in provider_body
            assert "stop" not in provider_body
            assert all(not message.get("prefix") for message in provider_body["messages"] if isinstance(message, dict))

            exported = server.make_request(
                "GET",
                "/v1/policy/transitions?request_id=policy_canary_profiles_2&limit=5",
            )
            item = exported.body["items"][0]
            assert item["executed_action"]["thinking_mode"] == "enabled"
            assert item["applied_provider_controls"]["thinking_enabled"] is True
            assert "sampling_profile" in item["applied_provider_controls"]["suppressed_fields"]
            assert "repetition_profile" in item["applied_provider_controls"]["suppressed_fields"]
            assert "prefix_profile" in item["applied_provider_controls"]["suppressed_fields"]
            assert "stop_profile" in item["applied_provider_controls"]["suppressed_fields"]
            assert item["applied_provider_controls"]["prefix_used"] is False
        finally:
            server.stop()


def test_provider_mode_chat_completion_round_trips_tools_and_tool_history():
    global server

    def tool_response(payload):
        if selector_tool_name(payload) == "ongoing_tasks_get_due":
            return {
                "id": "chatcmpl-direct-tool",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_direct_due",
                            "type": "function",
                            "function": {
                                "name": "ongoing_tasks_get_due",
                                "arguments": json.dumps({
                                    "task_id": "task_123",
                                }),
                            },
                        }],
                        "reasoning_content": "Constrain the due-task lookup to the one task that matters.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
            }
        return {
            "id": "chatcmpl-mock-tools-unexpected",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "unexpected",
                    "reasoning_content": "unexpected",
                },
            }],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "total_tokens": 26,
            },
        }

    with run_mock_deepseek(tool_response) as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            request = {
                "model": "deepseek-reasoner",
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": "Check the due ongoing tasks."},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should inspect the due recurring tasks before answering.",
                        "tool_calls": [{
                            "id": "call_due_1",
                            "type": "function",
                            "function": {
                                "name": "ongoing_tasks_get_due",
                                "arguments": "{}",
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_due_1",
                        "content": "{\"tasks\":[]}",
                    },
                    {"role": "user", "content": "If any are due, pick the highest-value one."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "ongoing_tasks_get_due",
                        "description": "Return ongoing tasks whose due time has passed.",
                        "parameters": {
                            "type": "object",
                            "description": "The due-task query payload.",
                            "properties": {
                                "task_id": {
                                    "type": "string",
                                    "description": "Optional one-task lookup."
                                }
                            }
                        }
                    }
                }],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "stream": False,
            }
            response = server.make_request("POST", "/v1/chat/completions", data=request)

            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["finish_reason"] in ("stop", "tool_calls")
            assert choice["message"]["content"] is None
            assert choice["message"]["reasoning_content"] == "Constrain the due-task lookup to the one task that matters."
            assert len(choice["message"]["tool_calls"]) == 1
            assert choice["message"]["tool_calls"] == [{
                "id": choice["message"]["tool_calls"][0]["id"],
                "type": "function",
                "function": {
                    "name": "ongoing_tasks_get_due",
                    "arguments": "{\"task_id\":\"task_123\"}",
                },
            }]

            provider_requests = [request for request in state["requests"] if request["path"] == "/chat/completions"]
            assert len(provider_requests) == 1
            provider_request = provider_requests[0]["body"]
            assert provider_request["tools"][0]["function"]["name"] == "ongoing_tasks_get_due"
            assert provider_request["tool_choice"] == "auto"
            assert provider_request["parallel_tool_calls"] is False
            assert provider_request["max_tokens"] == 768
            assert provider_request["thinking"] == {"type": "enabled"}
            assert any(
                message.get("reasoning_content") == request["messages"][1]["reasoning_content"]
                for message in provider_request["messages"]
            )
            assert any(
                message.get("tool_calls") == request["messages"][1]["tool_calls"]
                for message in provider_request["messages"]
            )
            assert any(
                message.get("tool_call_id") == "call_due_1" and message.get("content") == "{\"tasks\":[]}"
                for message in provider_request["messages"]
            )
            assert any(
                message.get("role") == "system" and message.get("content", "").startswith("Metacognitive control policy:")
                for message in provider_request["messages"]
            )
            assert any(
                message.get("role") == "system" and message.get("content", "").startswith("Current emotive guidance: valence=")
                for message in provider_request["messages"]
            )

            traced = server.make_request("GET", "/v1/debug/request-traces?limit=100")
            assert traced.status_code == 200
            guidance_events = [
                item for item in traced.body["items"]
                if item["component"] == "runtime_guidance" and item["event"] == "guidance_evaluated"
            ]
            assert guidance_events
            assert any(event["data"]["vad_injected"] is True for event in guidance_events)
            assert any(
                (event["data"]["vad_guidance"] or "").startswith("Current emotive guidance: valence=")
                for event in guidance_events
            )
            provider_finished = [
                item for item in traced.body["items"]
                if item["component"] == "provider" and item["event"] == "provider_request_finished"
            ]
            assert [item["data"]["reasoning_content"] for item in provider_finished[-1:]] == [
                "Constrain the due-task lookup to the one task that matters.",
            ]
        finally:
            server.stop()


def test_provider_mode_skips_stale_reasoning_replay_without_active_tool_continuation():
    global server
    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Summarize the project status."},
                    {
                        "role": "assistant",
                        "content": "The project is on track.",
                        "reasoning_content": "I already decided on the summary.",
                    },
                    {"role": "user", "content": "Now rewrite it more crisply."},
                ],
                "stream": False,
            })

            assert response.status_code == 200
            outbound = state["requests"][0]["body"]
            assert_has_skill_memory_indexes(outbound)
            assert outbound_non_system_messages(outbound)[0] == {"role": "user", "content": "Summarize the project status."}
            assert outbound_non_system_messages(outbound)[1] == {"role": "assistant", "content": "The project is on track."}
            assert outbound_non_system_messages(outbound)[2] == {"role": "user", "content": "Now rewrite it more crisply."}
            assert_has_policy_guidance(outbound)
        finally:
            server.stop()


def test_provider_mode_injects_skill_and_memory_indexes(tmp_path):
    global server
    skills_dir = tmp_path / "skills"
    memories_dir = tmp_path / "memories"
    skills_dir.mkdir()
    memories_dir.mkdir()
    (skills_dir / "deploy-worker.md").write_text("# Deploy Worker\n", encoding="utf-8")
    (memories_dir / "runtime-preference.md").write_text("---\nkind: \"preference\"\n---\nmanual reads only\n", encoding="utf-8")

    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_SKILLS_DIR": str(skills_dir),
            "VICUNA_HARD_MEMORY_DIR": str(memories_dir),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "What can you read?"},
                ],
                "stream": False,
            })

            assert response.status_code == 200
            outbound = state["requests"][0]["body"]
            assert_has_skill_memory_indexes(outbound, "deploy-worker.md", "runtime-preference.md")
            index_message = next(
                message for message in outbound_system_messages(outbound)
                if "SKILLS:\n" in message and "\n\nMEMORIES:\n" in message
            )
            assert "Current skill_create authorization: not allowed" in index_message
        finally:
            server.stop()


def test_provider_mode_host_shell_root_defaults_cover_skills_memories_and_heuristics(tmp_path):
    global server
    host_shell_root = tmp_path / "host-home"
    skills_dir = host_shell_root / "skills"
    memories_dir = host_shell_root / "memories"
    heuristics_dir = host_shell_root / "heuristics"
    skills_dir.mkdir(parents=True)
    memories_dir.mkdir(parents=True)
    (skills_dir / "deploy-worker.md").write_text("# Deploy Worker\n", encoding="utf-8")
    (memories_dir / "operator-preference.md").write_text(
        "---\nkind: \"preference\"\n---\nmanual reads only\n",
        encoding="utf-8",
    )

    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_HOST_SHELL_ROOT": str(host_shell_root),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the manual reading method for skills and memories. Speed doesn't matter.",
                    },
                ],
                "stream": False,
            })

            assert response.status_code == 200
            outbound = state["requests"][0]["body"]
            assert_has_skill_memory_indexes(outbound, "deploy-worker.md", "operator-preference.md")

            written_memories = sorted(path.name for path in memories_dir.glob("*.md"))
            assert len(written_memories) >= 2
            assert any(name != "operator-preference.md" for name in written_memories)

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["emotive_runtime"]["heuristic_memory"]["path"] == str(
                heuristics_dir / "vicuna-heuristic-memory.json"
            )
        finally:
            server.stop()


def test_provider_mode_marks_skill_create_allowed_only_on_direct_user_request(tmp_path):
    global server
    skills_dir = tmp_path / "skills"
    memories_dir = tmp_path / "memories"
    skills_dir.mkdir()
    memories_dir.mkdir()

    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_SKILLS_DIR": str(skills_dir),
            "VICUNA_HARD_MEMORY_DIR": str(memories_dir),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Create a skill for deployment handoff."},
                ],
                "stream": False,
            })

            assert response.status_code == 200
            outbound = state["requests"][0]["body"]
            index_message = next(
                message for message in outbound_system_messages(outbound)
                if "SKILLS:\n" in message and "\n\nMEMORIES:\n" in message
            )
            assert "Current skill_create authorization: allowed" in index_message
        finally:
            server.stop()


def test_provider_mode_post_response_memory_capture_writes_preference_memory(tmp_path):
    global server
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir()

    with run_mock_deepseek() as (base_url, _state):
        server = make_provider_server(base_url, {
            "VICUNA_HARD_MEMORY_DIR": str(memories_dir),
            "VICUNA_SKILLS_DIR": str(tmp_path / "skills"),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the manual reading method for skills and memories. Speed doesn't matter. Skill creation should only happen on direct user request.",
                    },
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "post_memory_write_1",
            })

            assert response.status_code == 200
            memory_files = list(memories_dir.glob("*.md"))
            assert memory_files
            content = memory_files[0].read_text(encoding="utf-8")
            assert 'kind: "preference"' in content
            assert "manual reading method" in content.lower()

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=post_memory_write_1&limit=20")
            assert traces.status_code == 200
            assert any(
                item["component"] == "post_response_memory" and item["event"] == "written"
                for item in traces.body["items"]
            )
        finally:
            server.stop()


def test_provider_mode_post_response_memory_capture_skips_non_durable_turn(tmp_path):
    global server
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir()

    with run_mock_deepseek() as (base_url, _state):
        server = make_provider_server(base_url, {
            "VICUNA_HARD_MEMORY_DIR": str(memories_dir),
            "VICUNA_SKILLS_DIR": str(tmp_path / "skills"),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "post_memory_skip_1",
            })

            assert response.status_code == 200
            assert list(memories_dir.glob("*.md")) == []

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=post_memory_skip_1&limit=20")
            assert traces.status_code == 200
            assert any(
                item["component"] == "post_response_memory"
                and item["event"] == "skipped"
                and item["data"]["reason"] == "no_durable_memory_match"
                for item in traces.body["items"]
            )
        finally:
            server.stop()


def test_provider_mode_interleaved_guidance_is_request_scoped_and_forwards_thinking():
    global server

    def tool_response(_payload):
        return {
            "id": "chatcmpl-mock-tools",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Handled.",
                    "reasoning_content": "I have enough context to respond.",
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
            },
        }

    with run_mock_deepseek(tool_response) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_DEEPSEEK_MODEL": "deepseek-chat",
        })
        server.api_surface = "openai"
        try:
            server.start()

            base_request = {
                "model": "deepseek-chat",
                "max_tokens": 2048,
                "max_completion_tokens": 3072,
                "max_output_tokens": 4096,
                "thinking": {"type": "enabled"},
                "messages": [
                    {"role": "user", "content": "Check the next operational step."},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should inspect the runtime state first.",
                        "tool_calls": [{
                            "id": "call_runtime_1",
                            "type": "function",
                            "function": {
                                "name": "runtime_status",
                                "arguments": "{}",
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_runtime_1",
                        "content": "{\"status\":\"ok\",\"note\":\"all clear\"}",
                    },
                ],
                "stream": False,
            }
            first = server.make_request("POST", "/v1/chat/completions", data=base_request)
            assert first.status_code == 200

            second_request = json.loads(json.dumps(base_request))
            second_request["messages"][0]["content"] = "Check the broken path next."
            second_request["messages"][1]["reasoning_content"] = "I should inspect the failure state first."
            second_request["messages"][2]["content"] = "{\"status\":\"error\",\"note\":\"timeout and failure detected\"}"
            second = server.make_request("POST", "/v1/chat/completions", data=second_request)
            assert second.status_code == 200

            first_outbound = state["requests"][0]["body"]
            second_outbound = state["requests"][1]["body"]
            assert first_outbound["max_tokens"] == 768
            assert second_outbound["max_tokens"] == 768
            assert first_outbound["thinking"]["type"] in ("enabled", "disabled")
            assert second_outbound["thinking"]["type"] in ("enabled", "disabled")
            assert_has_skill_memory_indexes(first_outbound)
            assert_has_skill_memory_indexes(second_outbound)
            first_guidance = [message for message in outbound_system_messages(first_outbound) if message.startswith("Current emotive guidance:")]
            second_guidance = [message for message in outbound_system_messages(second_outbound) if message.startswith("Current emotive guidance:")]
            assert first_guidance
            assert second_guidance
            assert first_guidance[0] != second_guidance[0]
        finally:
            server.stop()


def test_provider_mode_responses_route_emits_function_call_items():
    global server

    def tool_response(payload):
        if selector_tool_name(payload) == "ongoing_tasks_get_due":
            return selector_choice_response(
                "resp-direct-tool",
                "No arguments are required for the due-task lookup.",
                "ongoing_tasks_get_due",
                {},
            )
        return {
            "id": "resp-mock-tools-unexpected",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "unexpected",
                    "reasoning_content": "unexpected",
                },
            }],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        }

    with run_mock_deepseek(tool_response) as (base_url, _):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/responses", data={
                "model": "deepseek-reasoner",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "ongoing_tasks_get_due",
                        "description": "Return due recurring tasks.",
                        "parameters": {
                            "type": "object",
                            "description": "The due-task query payload.",
                            "properties": {}
                        }
                    }
                }],
                "input": [
                    {"role": "user", "content": "Check due tasks."},
                ],
            })
            assert response.status_code == 200
            assert response.body["object"] == "response"
            assert response.body["output"][0]["type"] == "reasoning"
            assert response.body["output"][1]["type"] == "function_call"
            assert response.body["output"][1]["name"] == "ongoing_tasks_get_due"
            assert response.body["output"][1]["arguments"] == "{}"
            assert response.body["output"][1]["call_id"]
        finally:
            server.stop()


def test_provider_mode_staged_tool_loop_supports_back_navigation_and_completion():
    global server
    stage_counter = {"family": 0}

    def staged_response(payload):
        if selector_tool_name(payload) == "select_family":
            stage_counter["family"] += 1
            family = "Ongoing Tasks" if stage_counter["family"] == 1 else "Telegram"
            return selector_choice_response(
                f"stage-family-{stage_counter['family']}",
                f"Choose the {family.lower()} family first.",
                "select_family",
                {"family": family},
            )
        if selector_tool_name(payload) == "select_method":
            stage_details = "\n".join(
                message["content"] for message in payload["messages"]
                if message.get("role") == "system"
            ).lower()
            if "chosen family: ongoing tasks" in stage_details:
                return selector_choice_response(
                    "stage-method-back",
                    "Back out and pick a better family.",
                    "select_method",
                    {"method": "back"},
                )
            if "chosen family: telegram" in stage_details:
                return selector_choice_response(
                    "stage-method-complete",
                    "The tool loop is complete.",
                    "select_method",
                    {"method": "complete"},
                )
        if any(
            message.get("role") == "system" and "tool loop is complete" in message.get("content", "").lower()
            for message in payload["messages"]
        ):
            return {
                "id": "stage-final-response",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "All set.",
                        "reasoning_content": "Now reply directly because the active loop is complete.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        return {
            "id": "stage-unexpected",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "unexpected",
                    "reasoning_content": "unexpected",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }

    with run_mock_deepseek(staged_response) as (base_url, state):
        server = make_staged_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "max_completion_tokens": 2048,
                "messages": [
                    {"role": "user", "content": "Check due tasks, then wrap up if nothing is due."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "ongoing_tasks_get_due",
                        "description": "Return due recurring tasks.",
                        "parameters": {
                            "type": "object",
                            "description": "The due-task query payload.",
                            "properties": {}
                        },
                        "x-vicuna-family-id": "ongoing_tasks",
                        "x-vicuna-family-name": "Ongoing Tasks",
                        "x-vicuna-family-description": "Inspect and manage recurring tasks.",
                    }
                }, {
                    "type": "function",
                    "function": {
                        "name": "telegram_relay",
                        "description": "Send a Telegram reply to the user.",
                        "parameters": {
                            "type": "object",
                            "description": "The telegram relay payload.",
                            "required": ["chat_scope", "text"],
                            "properties": {
                                "chat_scope": {
                                    "type": "string",
                                    "description": "The target chat scope."
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The message text to send."
                                }
                            }
                        },
                        "x-vicuna-family-id": "telegram",
                        "x-vicuna-family-name": "Telegram",
                        "x-vicuna-family-description": "Reply directly to the user through Telegram."
                    }
                }],
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "stop"
            assert response.body["choices"][0]["message"]["content"] == "All set."
            assert len(state["requests"]) == 5
            assert all(request["body"]["max_tokens"] == 768 for request in state["requests"])
            assert state["requests"][0]["body"]["tools"][0]["function"]["name"] == "select_family"
            assert state["requests"][1]["body"]["tools"][0]["function"]["name"] == "select_method"
            assert state["requests"][2]["body"]["tools"][0]["function"]["name"] == "select_family"
            assert state["requests"][3]["body"]["tools"][0]["function"]["name"] == "select_method"
            assert any(
                "tool loop is complete" in message.get("content", "").lower()
                for message in state["requests"][4]["body"]["messages"]
                if message.get("role") == "system"
            )
        finally:
            server.stop()


def test_provider_mode_retries_invalid_staged_family_selection_once():
    global server
    family_attempts = {"count": 0}

    def staged_response(payload):
        if selector_tool_name(payload) == "select_family":
            family_attempts["count"] += 1
            if family_attempts["count"] == 1:
                return {
                    "id": "retry-family-1",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Pick the ongoing task family once the JSON is valid.",
                        },
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
                }
            return selector_choice_response(
                "retry-family-2",
                "Pick the ongoing task family once the JSON is valid.",
                "select_family",
                {"family": "Ongoing Tasks"},
            )
        if selector_tool_name(payload) == "select_method":
            return selector_choice_response(
                "retry-family-method",
                "Use the due-task getter.",
                "select_method",
                {"method": "get_due"},
            )
        if "submit_payload" in selector_tool_names(payload):
            return selector_choice_response(
                "retry-family-payload",
                "No payload fields are required.",
                "submit_payload",
                {},
            )
        return {
            "id": "retry-family-unexpected",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "unexpected",
                    "reasoning_content": "unexpected",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }

    with run_mock_deepseek(staged_response) as (base_url, state):
        server = make_staged_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Check due tasks."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "ongoing_tasks_get_due",
                        "description": "Return due recurring tasks.",
                        "parameters": {
                            "type": "object",
                            "description": "The due-task query payload.",
                            "properties": {}
                        },
                        "x-vicuna-family-id": "ongoing_tasks",
                        "x-vicuna-family-name": "Ongoing Tasks",
                        "x-vicuna-family-description": "Inspect and manage recurring tasks.",
                    }
                }],
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ongoing_tasks_get_due"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert len(state["requests"]) == 4
            assert not any(
                "Previous response error:" in message.get("content", "")
                for message in state["requests"][0]["body"]["messages"]
                if message.get("role") == "system"
            )
            assert any(
                "Previous response error:" in message.get("content", "")
                for message in state["requests"][1]["body"]["messages"]
                if message.get("role") == "system"
            )
            assert any(
                "MANDATORY: call exactly one selector tool immediately" in message.get("content", "")
                for message in state["requests"][1]["body"]["messages"]
                if message.get("role") == "system"
            )
        finally:
            server.stop()


def test_provider_mode_retries_invalid_staged_method_selection_once():
    global server
    method_attempts = {"count": 0}

    def staged_response(payload):
        if selector_tool_name(payload) == "select_family":
            return selector_choice_response(
                "retry-method-family",
                "Choose the ongoing tasks family.",
                "select_family",
                {"family": "Ongoing Tasks"},
            )
        if selector_tool_name(payload) == "select_method":
            method_attempts["count"] += 1
            if method_attempts["count"] == 1:
                return {
                    "id": "retry-method-1",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Pick the due-task getter once the JSON is valid.",
                        },
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
                }
            return selector_choice_response(
                "retry-method-2",
                "Pick the due-task getter once the JSON is valid.",
                "select_method",
                {"method": "get_due"},
            )
        if "submit_payload" in selector_tool_names(payload):
            return selector_choice_response(
                "retry-method-payload",
                "No payload fields are required.",
                "submit_payload",
                {},
            )
        return {
            "id": "retry-method-unexpected",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "unexpected",
                    "reasoning_content": "unexpected",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }

    with run_mock_deepseek(staged_response) as (base_url, state):
        server = make_staged_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Check due tasks."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "ongoing_tasks_get_due",
                        "description": "Return due recurring tasks.",
                        "parameters": {
                            "type": "object",
                            "description": "The due-task query payload.",
                            "properties": {}
                        },
                        "x-vicuna-family-id": "ongoing_tasks",
                        "x-vicuna-family-name": "Ongoing Tasks",
                        "x-vicuna-family-description": "Inspect and manage recurring tasks.",
                    }
                }],
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ongoing_tasks_get_due"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert len(state["requests"]) == 4
            assert not any(
                "Previous response error:" in message.get("content", "")
                for message in state["requests"][1]["body"]["messages"]
                if message.get("role") == "system"
            )
            assert any(
                "Previous response error:" in message.get("content", "")
                for message in state["requests"][2]["body"]["messages"]
                if message.get("role") == "system"
            )
            assert any(
                "MANDATORY: call exactly one selector tool immediately" in message.get("content", "")
                for message in state["requests"][2]["body"]["messages"]
                if message.get("role") == "system"
            )
        finally:
            server.stop()


def test_provider_mode_rejects_tool_requests_without_staged_metadata():
    global server
    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Use the tool."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "opaque_tool",
                        "description": "Do something opaque.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                },
                            },
                        },
                    },
                }],
                "tool_choice": "auto",
                "stream": False,
            })

            assert response.status_code == 200
            assert state["requests"][0]["body"]["tools"][0]["function"]["name"] == "opaque_tool"
            assert_has_policy_guidance(state["requests"][0]["body"])
        finally:
            server.stop()


def test_provider_mode_retains_only_bounded_latest_traces():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url, {
            "VICUNA_EMOTIVE_MAX_TURN_HISTORY": "2",
        })
        server.api_surface = "openai"
        try:
            server.start()

            trace_ids = []
            for prompt in ["first", "second", "third"]:
                response = server.make_request("POST", "/v1/chat/completions", data={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                })
                assert response.status_code == 200
                trace_ids.append(response.body["vicuna_emotive_trace"]["trace_id"])

            latest = server.make_request("GET", "/v1/emotive/trace/latest")
            assert latest.status_code == 200
            assert latest.body["retained_turns"] == 2
            assert latest.body["trace"]["trace_id"] == trace_ids[-1]
        finally:
            server.stop()


def test_provider_mode_keeps_bridge_compatibility_endpoints():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"] == []
            assert outbox.body["last_sequence"] == 0

            enqueue = server.make_request("POST", "/v1/telegram/outbox", data={
                "kind": "message",
                "chat_scope": "12345",
                "text": "Follow-up ready.",
                "dedupe_key": "final-followup",
            })
            assert enqueue.status_code == 200
            assert enqueue.body["ok"] is True
            assert enqueue.body["queued"] is True
            assert enqueue.body["deduplicated"] is False
            assert enqueue.body["sequence_number"] == 1
            assert enqueue.body["chat_scope"] == "12345"

            deduped = server.make_request("POST", "/v1/telegram/outbox", data={
                "kind": "message",
                "chat_scope": "12345",
                "text": "Follow-up ready.",
                "dedupe_key": "final-followup",
            })
            assert deduped.status_code == 200
            assert deduped.body["ok"] is True
            assert deduped.body["queued"] is False
            assert deduped.body["deduplicated"] is True
            assert deduped.body["sequence_number"] == 1

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["last_sequence"] == 1
            assert outbox.body["stored_items"] == 1
            assert outbox.body["newest_sequence"] == 1
            assert outbox.body["items"] == [{
                "sequence_number": 1,
                "kind": "message",
                "chat_scope": "12345",
                "telegram_method": "sendMessage",
                "telegram_payload": {
                    "text": "Follow-up ready.",
                },
                "text": "Follow-up ready.",
                "dedupe_key": "final-followup",
            }]

            rich_enqueue = server.make_request("POST", "/v1/telegram/outbox", data={
                "kind": "message",
                "chat_scope": "12345",
                "telegram_method": "sendPhoto",
                "telegram_payload": {
                    "photo": "https://example.com/report.png",
                    "caption": "<b>Report ready</b>",
                    "parse_mode": "HTML",
                    "reply_markup": {
                        "inline_keyboard": [[{
                            "text": "Open report",
                            "url": "https://example.com/report",
                        }]],
                    },
                },
                "dedupe_key": "rich-followup",
            })
            assert rich_enqueue.status_code == 200
            assert rich_enqueue.body["ok"] is True
            assert rich_enqueue.body["queued"] is True
            assert rich_enqueue.body["deduplicated"] is False
            assert rich_enqueue.body["sequence_number"] == 2

            rich_outbox = server.make_request("GET", "/v1/telegram/outbox?after=1")
            assert rich_outbox.status_code == 200
            assert rich_outbox.body["items"] == [{
                "sequence_number": 2,
                "kind": "message",
                "chat_scope": "12345",
                "telegram_method": "sendPhoto",
                "telegram_payload": {
                    "photo": "https://example.com/report.png",
                    "caption": "<b>Report ready</b>",
                    "parse_mode": "HTML",
                    "reply_markup": {
                        "inline_keyboard": [[{
                            "text": "Open report",
                            "url": "https://example.com/report",
                        }]],
                    },
                },
                "text": "<b>Report ready</b>",
                "dedupe_key": "rich-followup",
            }]

            approval = server.make_request("POST", "/v1/telegram/approval", data={
                "approval_id": "appr_123",
                "decision": "allow",
            })
            assert approval.status_code == 200
            assert approval.body["ok"] is True
            assert approval.body["approval_id"] == "appr_123"

            interruption = server.make_request("POST", "/v1/telegram/interruption", data={
                "chat_scope": "12345",
            })
            assert interruption.status_code == 200
            assert interruption.body["ok"] is True
            assert interruption.body["cancelled_approval_ids"] == []
        finally:
            server.stop()


def test_provider_mode_requires_explicit_telegram_delivery_for_bridge_request():
    global server

    def rich_response(_payload):
        return {
            "id": "bridge-family-rich-plan",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("Hello from DeepSeek."),
                    "reasoning_content": "Return the bridge-scoped rich response directly.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(rich_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Earlier question."},
                    {"role": "assistant", "content": "Earlier answer."},
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "77",
                "X-Vicuna-Telegram-Deferred-Delivery": "1",
                "X-Vicuna-Telegram-Conversation-Id": "tc1",
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "stop"
            assert response.body["choices"][0]["message"]["content"] == ""
            assert response.body["vicuna_telegram_delivery"] == {
                "handled": True,
                "queued": True,
                "deduplicated": False,
                "sequence_number": 1,
                "chat_scope": "12345",
                "telegram_method": "sendMessage",
                "reply_to_message_id": 77,
                "source": "rich_plan",
            }
            assert response.body["vicuna_rich_response"]["body"] == "Hello from DeepSeek."
            assert state["requests"][0]["body"]["messages"][0]["content"].startswith("Bridge-scoped delivery contract:")
            assert_has_policy_guidance(state["requests"][0]["body"])

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert len(outbox.body["items"]) == 1
            item = outbox.body["items"][0]
            assert item["sequence_number"] == 1
            assert item["kind"] == "message"
            assert item["chat_scope"] == "12345"
            assert item["telegram_method"] == "sendMessage"
            assert item["telegram_payload"] == {
                "text": "Hello from DeepSeek.",
            }
            assert item["text"] == "Hello from DeepSeek."
            assert item["reply_to_message_id"] == 77
            assert item["dedupe_key"] == "bridge:tc1:77:sendMessage"
            animation = item["emotive_animation"]
            assert animation["bundle_version"] == 2
            assert animation["generation_start_block_index"] >= 3
            assert animation["seconds_per_keyframe"] == 0.5
            assert animation["fps"] == 30
            assert animation["raw_keyframe_count"] >= len(animation["keyframes"])
            assert animation["distinct_keyframe_count"] == len(animation["keyframes"])
            assert sum(frame["hold_keyframe_count"] for frame in animation["keyframes"]) == animation["raw_keyframe_count"]
            assert len(animation["dimensions"]) == 14
            assert animation["dimensions"][0]["label"] == "Epistemic Pressure"
            assert all(frame["trace_block_index"] >= 3 for frame in animation["keyframes"])
            assert all(frame["hold_keyframe_count"] >= 1 for frame in animation["keyframes"])
            assert animation["keyframes"][0]["source_kind"] == "assistant_reasoning"
        finally:
            server.stop()


def test_provider_mode_bridge_plain_assistant_text_defaults_to_plain_telegram_delivery():
    global server

    def plain_response(_payload):
        return {
            "id": "bridge-plain-text",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello! What's up?",
                    "reasoning_content": "Reply directly without extra formatting.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(plain_response) as (base_url, _):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "78",
                "X-Vicuna-Telegram-Conversation-Id": "tc-plain-default",
            })

            assert response.status_code == 200
            assert response.body["vicuna_telegram_delivery"]["source"] == "rich_plan"
            assert response.body["vicuna_rich_response"]["format"] == "plain_text"
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"][0]["telegram_payload"] == {
                "text": "Hello! What's up?",
            }
        finally:
            server.stop()


def test_provider_mode_bridge_markdown_rich_plan_normalizes_to_html_delivery():
    global server

    def markdown_response(_payload):
        return {
            "id": "bridge-markdown-text",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("*Hello!* What's up?", format="markdown"),
                    "reasoning_content": "Reply with markdown compatibility metadata.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(markdown_response) as (base_url, _):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_markdown_bridge_1",
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "79",
                "X-Vicuna-Telegram-Conversation-Id": "tc-markdown-default",
            })

            assert response.status_code == 200
            assert response.body["vicuna_telegram_delivery"]["source"] == "rich_plan"
            assert response.body["vicuna_rich_response"]["format"] == "markdown"
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"][0]["telegram_payload"] == {
                "text": "<i>Hello!</i> What's up?",
                "parse_mode": "HTML",
            }
            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_markdown_bridge_1&limit=50")
            assert traces.status_code == 200
            assert any(item["event"] == "rich_plan_markdown_normalized_html" for item in traces.body["items"])
        finally:
            server.stop()


def test_provider_mode_bridge_markdown_rich_plan_preserves_dividers_headings_and_code_blocks():
    global server

    body = "\n".join([
        "### Status",
        "",
        "**Ready** with `code`",
        "",
        "---",
        "",
        "> quoted line",
        "",
        "```python",
        "print('ok')",
        "```",
    ])

    def markdown_response(_payload):
        return {
            "id": "bridge-markdown-structured",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text(body, format="markdown"),
                    "reasoning_content": "Reply with structured markdown.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(markdown_response) as (base_url, _):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "80",
                "X-Vicuna-Telegram-Conversation-Id": "tc-markdown-structured",
            })

            assert response.status_code == 200
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            payload = outbox.body["items"][0]["telegram_payload"]
            assert payload["parse_mode"] == "HTML"
            assert payload["text"] == (
                "<b>Status</b>\n\n"
                "<b>Ready</b> with <code>code</code>\n\n"
                "━━━━━━━━━━━━\n\n"
                "<blockquote>quoted line</blockquote>\n\n"
                "<pre><code class=\"language-python\">print('ok')</code></pre>"
            )
        finally:
            server.stop()


def test_provider_mode_bridge_markdown_pipe_tables_render_as_preformatted_grids():
    global server

    body = "\n".join([
        "| Name | Score |",
        "| --- | --- |",
        "| Calm | 0.42 |",
        "| Trust | 0.80 |",
    ])

    def markdown_response(_payload):
        return {
            "id": "bridge-markdown-table",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text(body, format="markdown"),
                    "reasoning_content": "Reply with a compact markdown table.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(markdown_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply with a grid."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "81",
                "X-Vicuna-Telegram-Conversation-Id": "tc-markdown-table",
            })

            assert response.status_code == 200
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            payload = outbox.body["items"][0]["telegram_payload"]
            assert payload["parse_mode"] == "HTML"
            assert payload["text"] == (
                "<pre>+-------+-------+\n"
                "| Name  | Score |\n"
                "+-------+-------+\n"
                "| Calm  | 0.42  |\n"
                "| Trust | 0.80  |\n"
                "+-------+-------+</pre>"
            )

            system_prompt = state["requests"][0]["body"]["messages"][0]["content"]
            assert "preformatted grid" in system_prompt
            assert "raw markdown pipe tables" in system_prompt
        finally:
            server.stop()


def test_provider_mode_request_trace_endpoint_records_bridge_events():
    global server

    def rich_response(_payload):
        return {
            "id": "bridge-trace-rich-plan",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("Hello from DeepSeek."),
                    "reasoning_content": "Return the bridge-scoped rich response directly.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(rich_response) as (base_url, _):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_bridge_1",
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "77",
                "X-Vicuna-Telegram-Deferred-Delivery": "1",
                "X-Vicuna-Telegram-Conversation-Id": "tc-trace",
            })

            assert response.status_code == 200
            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_bridge_1&limit=50")
            assert traces.status_code == 200
            events = [item["event"] for item in traces.body["items"]]
            assert "request_received" in events
            assert "telegram_runtime_tools_loaded" in events
            assert "bridge_round_started" in events
            assert "policy_computed" in events
            assert "provider_request_started" in events
            assert "provider_request_finished" in events
            assert "telegram_outbox_enqueued" in events
            assert events[-1] == "request_completed"
            assert all(item["request_id"] == "trace_bridge_1" for item in traces.body["items"])
            assert all(item["bridge_scoped"] is True for item in traces.body["items"])
            assert any(item["component"] == "telegram_delivery" for item in traces.body["items"])
        finally:
            server.stop()


def test_provider_mode_executes_staged_telegram_tool_for_bridge_request():
    global server

    runtime_tools = [{
        "type": "function",
        "function": {
            "name": "radarr_list_downloaded_movies",
            "description": "List only the movies that are already fully downloaded in Radarr.",
            "parameters": {
                "type": "object",
                "properties": {},
                "description": "List only the movies that are already fully downloaded in Radarr.",
            },
            "x-vicuna-family-id": "radarr",
            "x-vicuna-family-name": "Radarr",
            "x-vicuna-family-description": "Inspect and manage the Radarr movie library on the media server.",
            "x-vicuna-method-name": "list_downloaded_movies",
            "x-vicuna-method-description": "List the movies already fully downloaded in Radarr.",
        },
    }, {
        "type": "function",
        "function": {
            "name": "telegram_relay",
            "description": "Legacy relay path that should be filtered from the staged Telegram family.",
            "parameters": {
                "type": "object",
                "description": "Legacy telegram relay payload.",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Plain text to send.",
                    },
                },
            },
            "x-vicuna-family-id": "telegram",
            "x-vicuna-family-name": "Telegram",
            "x-vicuna-family-description": "Legacy relay family description.",
            "x-vicuna-method-name": "relay",
            "x-vicuna-method-description": "Legacy relay method.",
        },
    }]

    def runtime_then_rich_response(payload):
        saw_runtime_observation = any(
            message.get("role") == "tool" and "downloaded_movie_count" in str(message.get("content", ""))
            for message in payload["messages"]
        )
        if not saw_runtime_observation:
            return selector_choice_response(
                "bridge-runtime-tool",
                "Inspect Radarr first.",
                "radarr_list_downloaded_movies",
                {},
            )
        return {
            "id": "bridge-runtime-rich-plan",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("<b>Ready</b>", format="html"),
                    "reasoning_content": "Return the final Telegram-ready rich text after the runtime lookup.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(runtime_then_rich_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": json.dumps(runtime_tools),
            "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON": json.dumps({
                "radarr_list_downloaded_movies": {
                    "ok": True,
                    "downloaded_movie_count": 51,
                    "movies": [{"title": "Arrival"}],
                },
            }),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Reply through Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "88",
                "X-Vicuna-Telegram-Conversation-Id": "tc2",
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "stop"
            assert response.body["choices"][0]["message"]["content"] == ""
            assert "tool_calls" not in response.body["choices"][0]["message"]
            assert response.body["vicuna_telegram_delivery"] == {
                "handled": True,
                "queued": True,
                "deduplicated": False,
                "sequence_number": 1,
                "chat_scope": "12345",
                "telegram_method": "sendMessage",
                "reply_to_message_id": 88,
                "source": "rich_plan",
            }
            assert response.body["vicuna_rich_response"]["format"] == "html"
            assert len(state["requests"]) == 2
            first_request = state["requests"][0]["body"]
            second_request = state["requests"][1]["body"]
            assert all(request["body"]["temperature"] == 0.2 for request in state["requests"])
            assert first_request["messages"][0]["role"] == "system"
            assert first_request["messages"][0]["content"].startswith("Bridge-scoped delivery contract:")
            assert "HTML is the canonical Telegram rich-text format" in first_request["messages"][0]["content"]
            assert_has_policy_guidance(first_request)
            assert any(
                tool["function"]["name"] == "radarr_list_downloaded_movies"
                for tool in first_request["tools"]
            )
            assert any(
                message.get("role") == "tool" and "downloaded_movie_count" in str(message.get("content", ""))
                for message in second_request["messages"]
            )
            assert_has_policy_guidance(second_request)

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"][0]["emotive_animation"]["generation_start_block_index"] >= 1
            assert outbox.body["items"][0]["emotive_animation"]["keyframes"]
            assert outbox.body["items"][0]["telegram_payload"]["parse_mode"] == "HTML"
        finally:
            server.stop()


def test_provider_mode_bridge_round_budget_forces_tool_free_synthesis_instead_of_500():
    global server

    runtime_tools = [{
        "type": "function",
        "function": {
            "name": "chaptarr_download_book",
            "description": "Search Chaptarr for a book and start a download when there is a match.",
            "parameters": {
                "type": "object",
                "required": ["term"],
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Book search term.",
                    },
                },
            },
            "x-vicuna-family-id": "chaptarr",
            "x-vicuna-family-name": "Chaptarr",
            "x-vicuna-family-description": "Inspect and manage book downloads.",
            "x-vicuna-method-name": "download_book",
            "x-vicuna-method-description": "Search for and download a book.",
        },
    }, {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web and return ranked results.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                },
            },
            "x-vicuna-family-id": "web",
            "x-vicuna-family-name": "Web Search",
            "x-vicuna-family-description": "Search the live web.",
            "x-vicuna-method-name": "search",
            "x-vicuna-method-description": "Search the live web.",
        },
    }]

    def looping_until_tool_free_synthesis(payload):
        if not payload.get("tools"):
            return {
                "id": "bridge-budget-synthesis",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "I couldn't confirm a book called \"By the Horns\" by Ali Hazelwood. "
                            "If you meant Ruby Dixon's \"By the Horns\", say so and I'll use that title instead."
                        ),
                        "reasoning_content": "The runtime tool budget is exhausted, so I should stop searching and ask the user to clarify the title or author.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            }

        tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
        if len(tool_messages) == 0:
            return selector_choice_response(
                "bridge-budget-1",
                "Try Chaptarr first.",
                "chaptarr_download_book",
                {"term": "By the Horns Ali Hazelwood"},
            )
        if len(tool_messages) == 1:
            return selector_choice_response(
                "bridge-budget-2",
                "Chaptarr found nothing, so verify the title on the web.",
                "web_search",
                {"query": "Ali Hazelwood By the Horns"},
            )
        raise AssertionError(f"expected tool-free synthesis request, got tools={selector_tool_names(payload)}")

    with run_mock_deepseek(looping_until_tool_free_synthesis) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_MAX_ROUNDS": "2",
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": json.dumps(runtime_tools),
            "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON": json.dumps({
                "chaptarr_download_book": {
                    "ok": False,
                    "error": {
                        "kind": "lookup_no_match",
                        "message": "lookup returned no candidates",
                    },
                },
                "web_search": {
                    "query": "Ali Hazelwood By the Horns",
                    "results": [{
                        "title": "Ali Hazelwood: Home",
                        "url": "https://alihazelwood.com/",
                        "excerpt": "No book titled By the Horns appears in the catalog.",
                    }],
                },
            }),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Download By the Horns by Ali Hazelwood."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_bridge_budget",
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "99",
                "X-Vicuna-Telegram-Conversation-Id": "tc-budget",
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "stop"
            assert response.body["choices"][0]["message"]["content"] == ""
            assert "tool_calls" not in response.body["choices"][0]["message"]
            assert response.body["vicuna_telegram_delivery"]["queued"] is True
            assert response.body["vicuna_rich_response"]["body"].startswith("I couldn't confirm a book called")

            assert len(state["requests"]) == 3
            assert selector_tool_names(state["requests"][0]["body"]) == [
                "chaptarr_download_book",
                "web_search",
            ]
            assert "tools" not in state["requests"][2]["body"]
            assert any(
                message["role"] == "system" and "runtime tool round budget (2 rounds) is exhausted" in message["content"]
                for message in state["requests"][2]["body"]["messages"]
            )

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_bridge_budget&limit=100")
            assert traces.status_code == 200
            events = [item["event"] for item in traces.body["items"]]
            assert "bridge_round_budget_synthesis_started" in events
            assert "bridge_round_budget_synthesis_completed" in events
            assert "request_failed" not in events

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"][0]["text"].startswith("I couldn't confirm a book called")
        finally:
            server.stop()


def test_provider_mode_recovers_dsml_tool_calls_from_assistant_content():
    global server

    runtime_tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web and return ranked results.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                },
            },
            "x-vicuna-family-id": "web",
            "x-vicuna-family-name": "Web Search",
            "x-vicuna-family-description": "Search the live web.",
            "x-vicuna-method-name": "search",
            "x-vicuna-method-description": "Search the live web.",
        },
    }]

    def dsml_tool_call_then_answer(payload):
        tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
        if len(tool_messages) == 0:
            return {
                "id": "bridge-dsml-1",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Let me check.\n"
                            "<｜DSML｜function_calls>\n"
                            "<｜DSML｜invoke name=\"web_search\">\n"
                            "<｜DSML｜parameter name=\"query\" string=\"true\">Penelope's Vegan Taqueria Logan Square</｜DSML｜parameter>\n"
                            "<｜DSML｜parameter name=\"max_results\" string=\"false\">4</｜DSML｜parameter>\n"
                            "</｜DSML｜invoke>\n"
                            "</｜DSML｜function_calls>"
                        ),
                        "reasoning_content": "I should search for the restaurant before replying.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        if len(tool_messages) == 1:
            assert tool_messages[0]["name"] == "web_search"
            return {
                "id": "bridge-dsml-2",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Penelope's Vegan Taqueria is in Logan Square and has strong recent reviews.",
                        "reasoning_content": "I have the search result, so I can answer directly.",
                    },
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
            }
        raise AssertionError(f"unexpected tool round count: {len(tool_messages)}")

    with run_mock_deepseek(dsml_tool_call_then_answer) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": json.dumps(runtime_tools),
            "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON": json.dumps({
                "web_search": {
                    "query": "Penelope's Vegan Taqueria Logan Square",
                    "results": [{
                        "title": "Penelope's Vegan Taqueria",
                        "url": "https://example.com/penelopes",
                        "excerpt": "Recent reviews praise the food and location in Logan Square.",
                    }],
                },
            }),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "What do you know about Penelope's Vegan Taqueria in Logan Square?"},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_bridge_dsml_recovery",
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "111",
                "X-Vicuna-Telegram-Conversation-Id": "tc-dsml-recovery",
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "stop"
            assert response.body["vicuna_telegram_delivery"]["queued"] is True
            assert response.body["vicuna_rich_response"]["body"].startswith("Penelope's Vegan Taqueria is in Logan Square")

            assert len(state["requests"]) == 2
            assistant_tool_replay = next(
                message
                for message in state["requests"][1]["body"]["messages"]
                if message.get("role") == "assistant" and message.get("tool_calls")
            )
            assert assistant_tool_replay["content"] == "Let me check."
            assert assistant_tool_replay["tool_calls"][0]["function"]["name"] == "web_search"
            assert json.loads(assistant_tool_replay["tool_calls"][0]["function"]["arguments"]) == {
                "query": "Penelope's Vegan Taqueria Logan Square",
                "max_results": 4,
            }

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_bridge_dsml_recovery&limit=100")
            assert traces.status_code == 200
            events = [item["event"] for item in traces.body["items"]]
            assert "provider_dsml_tool_calls_recovered" in events

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"][0]["text"].startswith("Penelope's Vegan Taqueria is in Logan Square")
            assert "<｜DSML｜function_calls>" not in outbox.body["items"][0]["text"]
        finally:
            server.stop()


def test_provider_mode_projects_web_search_payload_schema_to_required_subset():
    global server

    def staged_web_search_response(payload):
        if selector_tool_name(payload) == "select_family":
            return selector_choice_response(
                "web-family",
                "Web Search is the right family.",
                "select_family",
                {"family": "Web Search"},
            )
        if selector_tool_name(payload) == "select_method":
            return selector_choice_response(
                "web-method",
                "Use the search method.",
                "select_method",
                {"method": "search"},
            )
        if "submit_payload" in selector_tool_names(payload):
            return selector_choice_response(
                "web-payload",
                "Only the required query field is needed.",
                "submit_payload",
                {"query": "latest movie box office"},
            )
        raise AssertionError(f"unexpected staged prompt: {payload['messages'][-1]['content']}")

    with run_mock_deepseek(staged_web_search_response) as (base_url, state):
        server = make_staged_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Search the web."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the live web through Tavily and return ranked source evidence with URLs and excerpts.",
                        "parameters": {
                            "type": "object",
                            "description": "The Tavily search payload.",
                            "required": ["query"],
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The live web search query to run through Tavily.",
                                },
                                "topic": {
                                    "type": "string",
                                    "enum": ["general", "news", "finance"],
                                    "description": "The retrieval topic that best matches the query.",
                                },
                                "search_depth": {
                                    "type": "string",
                                    "enum": ["basic", "advanced"],
                                    "description": "How aggressively Tavily should retrieve and expand source evidence.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "minimum": 3,
                                    "maximum": 8,
                                    "description": "The maximum number of ranked sources to return.",
                                },
                                "time_range": {
                                    "type": "string",
                                    "enum": ["day", "week", "month", "year"],
                                    "description": "An optional recency window for the search.",
                                },
                                "include_domains": {
                                    "type": "array",
                                    "description": "Optional domains to include.",
                                    "items": {
                                        "type": "string",
                                        "description": "One domain to include.",
                                    },
                                },
                                "exclude_domains": {
                                    "type": "array",
                                    "description": "Optional domains to exclude.",
                                    "items": {
                                        "type": "string",
                                        "description": "One domain to exclude.",
                                    },
                                },
                                "country": {
                                    "type": "string",
                                    "description": "An optional country hint used to localize the search.",
                                },
                            },
                        },
                        "x-vicuna-family-id": "web_search",
                        "x-vicuna-family-name": "Web Search",
                        "x-vicuna-family-description": "Search the live web and return source-grounded evidence.",
                        "x-vicuna-method-name": "search",
                        "x-vicuna-method-description": "Run one Tavily web search query.",
                    },
                }],
                "tool_choice": "auto",
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            payload_schema = state["requests"][2]["body"]["tools"][0]["function"]["parameters"]
            assert payload_schema["required"] == ["query"]
            assert set(payload_schema["properties"].keys()) == {"query"}
            assert "topic" not in payload_schema["properties"]
            assert "search_depth" not in payload_schema["properties"]
            assert "max_results" not in payload_schema["properties"]
            assert "time_range" not in payload_schema["properties"]
            assert "include_domains" not in payload_schema["properties"]
            assert "exclude_domains" not in payload_schema["properties"]
            assert "country" not in payload_schema["properties"]
        finally:
            server.stop()


def test_provider_mode_strips_unsupported_array_keywords_from_projected_payload_schema():
    global server

    def staged_hard_memory_response(payload):
        if selector_tool_name(payload) == "select_family":
            return selector_choice_response(
                "memory-family",
                "Use Hard Memory.",
                "select_family",
                {"family": "Hard Memory"},
            )
        if selector_tool_name(payload) == "select_method":
            return selector_choice_response(
                "memory-method",
                "Use the write method.",
                "select_method",
                {"method": "write"},
            )
        if "submit_payload" in selector_tool_names(payload):
            return selector_choice_response(
                "memory-payload",
                "Only the required content field is needed.",
                "submit_payload",
                {"memories": [{"content": "Remember this."}]},
            )
        raise AssertionError(f"unexpected staged prompt: {payload['messages'][-1]['content']}")

    with run_mock_deepseek(staged_hard_memory_response) as (base_url, state):
        server = make_staged_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Write this to hard memory."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "hard_memory_write",
                        "description": "Archive explicit durable memories to Vicuña markdown hard memory.",
                        "parameters": {
                            "type": "object",
                            "description": "The hard-memory write payload.",
                            "required": ["memories"],
                            "properties": {
                                "memories": {
                                    "type": "array",
                                    "description": "The batch of durable memory primitives to archive.",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "description": "One durable memory primitive to archive.",
                                        "required": ["content"],
                                        "properties": {
                                            "content": {
                                                "type": "string",
                                                "description": "The durable memory content to archive.",
                                                "minLength": 1,
                                            },
                                            "title": {
                                                "type": "string",
                                                "description": "An optional short title for the memory.",
                                                "maxLength": 80,
                                            },
                                        },
                                    },
                                },
                                "containerTag": {
                                    "type": "string",
                                    "description": "An optional container tag used to group this write batch.",
                                },
                            },
                        },
                        "x-vicuna-family-id": "hard_memory",
                        "x-vicuna-family-name": "Hard Memory",
                        "x-vicuna-family-description": "Read from or write durable memory primitives in Vicuña hard memory.",
                        "x-vicuna-method-name": "write",
                        "x-vicuna-method-description": "Archive a batch of durable memory primitives.",
                    },
                }],
                "tool_choice": "auto",
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            payload_schema = state["requests"][2]["body"]["tools"][0]["function"]["parameters"]
            assert payload_schema["required"] == ["memories"]
            assert set(payload_schema["properties"].keys()) == {"memories"}
            memories_schema = payload_schema["properties"]["memories"]
            assert "minItems" not in memories_schema
            assert memories_schema["type"] == "array"
            item_schema = memories_schema["items"]
            assert item_schema["required"] == ["content"]
            assert set(item_schema["properties"].keys()) == {"content"}
            assert "minLength" not in item_schema["properties"]["content"]
        finally:
            server.stop()


def test_provider_logs_raw_malformed_tool_arguments_on_parse_failure():
    global server

    def malformed_arguments_response(payload):
        if selector_tool_name(payload) == "radarr_list_downloaded_movies":
            return {
                "id": "bad-tool-args",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Return the runtime tool call immediately.",
                        "tool_calls": [{
                            "id": "call_bad_args",
                            "type": "function",
                            "function": {
                                "name": "radarr_list_downloaded_movies",
                                "arguments": "{\"query\" \"Arrival\"}",
                            },
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        raise AssertionError(f"unexpected turn: {payload['messages'][-1]['content']}")

    with run_mock_deepseek(malformed_arguments_response) as (base_url, _):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": json.dumps([{
                "type": "function",
                "function": {
                    "name": "radarr_list_downloaded_movies",
                    "description": "List only the movies that are already fully downloaded in Radarr.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "description": "List only the movies that are already fully downloaded in Radarr.",
                    },
                    "x-vicuna-family-id": "radarr",
                    "x-vicuna-family-name": "Radarr",
                    "x-vicuna-family-description": "Inspect and manage the Radarr movie library on the media server.",
                    "x-vicuna-method-name": "list_downloaded_movies",
                    "x-vicuna-method-description": "List the movies already fully downloaded in Radarr.",
                },
            }]),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Check Radarr."},
                ],
                "stream": False,
            }, headers={
                "X-Client-Request-Id": "trace_bad_tool_args",
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "90",
                "X-Vicuna-Telegram-Conversation-Id": "tc-bad-args",
            })

            assert response.status_code == 500
            assert "malformed tool arguments json" in response.body["error"]["message"].lower()

            traces = server.make_request("GET", "/v1/debug/request-traces?request_id=trace_bad_tool_args&limit=30")
            assert traces.status_code == 200
            parse_fail = next(item for item in traces.body["items"] if item["event"] == "provider_tool_arguments_parse_failed")
            assert parse_fail["data"]["tool_name"] == "radarr_list_downloaded_movies"
            assert parse_fail["data"]["raw_arguments"] == "{\"query\" \"Arrival\"}"
        finally:
            server.stop()


def test_provider_mode_bridge_family_retry_mentions_telegram_mandatory_path_after_plain_text_drift():
    global server

    runtime_tools = [{
        "type": "function",
        "function": {
            "name": "radarr_list_downloaded_movies",
            "description": "List only the movies that are already fully downloaded in Radarr.",
            "parameters": {
                "type": "object",
                "properties": {},
                "description": "List only the movies that are already fully downloaded in Radarr.",
            },
            "x-vicuna-family-id": "radarr",
            "x-vicuna-family-name": "Radarr",
            "x-vicuna-family-description": "Inspect and manage the Radarr movie library on the media server.",
            "x-vicuna-method-name": "list_downloaded_movies",
            "x-vicuna-method-description": "List the movies already fully downloaded in Radarr.",
        },
    }]

    def plain_text_bridge_response(payload):
        saw_runtime_observation = any(
            message.get("role") == "tool" and "downloaded_movie_count" in str(message.get("content", ""))
            for message in payload["messages"]
        )
        if not saw_runtime_observation:
            return selector_choice_response(
                "bridge-runtime-step",
                "Inspect Radarr first.",
                "radarr_list_downloaded_movies",
                {},
            )
        return {
            "id": "bridge-plain-text-final",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Ready.",
                    "reasoning_content": "Bridge plain text should now be delivered directly.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    with run_mock_deepseek(plain_text_bridge_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": json.dumps(runtime_tools),
            "VICUNA_TELEGRAM_RUNTIME_TOOL_OBSERVATIONS_JSON": json.dumps({
                "radarr_list_downloaded_movies": {
                    "ok": True,
                    "downloaded_movie_count": 1,
                    "movies": [{"title": "Arrival"}],
                },
            }),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Check Radarr, then reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "88",
                "X-Vicuna-Telegram-Conversation-Id": "tc-retry",
            })

            assert response.status_code == 200
            assert response.body["vicuna_telegram_delivery"]["handled"] is True
            assert response.body["vicuna_telegram_delivery"]["source"] == "rich_plan"
            assert len(state["requests"]) == 2
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.body["items"][0]["text"] == "Ready."
        finally:
            server.stop()


def test_provider_mode_bridge_family_retry_handles_multibyte_plain_text_preview():
    global server

    long_plain_text = ("A" * 159) + "🙂 I should not break the retry prompt."

    def multibyte_response(_payload):
        return {
            "id": "bridge-family-utf8",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": long_plain_text,
                    "reasoning_content": "Return the multibyte bridge text directly.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    with run_mock_deepseek(multibyte_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "188",
                "X-Vicuna-Telegram-Conversation-Id": "tc-utf8-preview",
            })

            assert response.status_code == 200
            assert response.body["vicuna_telegram_delivery"]["handled"] is True
            assert len(state["requests"]) == 1
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.body["items"][0]["text"] == long_plain_text
        finally:
            server.stop()


def test_provider_mode_bridge_request_reuses_runtime_tool_and_staged_prompt_caches():
    global server

    def rich_response(_payload):
        return {
            "id": "cache-rich-plan",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("Done."),
                    "reasoning_content": "Return the cached bridge response directly.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(rich_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            for message_id in ("201", "202"):
                response = server.make_request("POST", "/v1/chat/completions", data={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": "Reply on Telegram."},
                    ],
                    "stream": False,
                }, headers={
                    "X-Vicuna-Telegram-Chat-Id": "12345",
                    "X-Vicuna-Telegram-Message-Id": message_id,
                    "X-Vicuna-Telegram-Conversation-Id": "tc-cache",
                })
                assert response.status_code == 200

            health = server.make_request("GET", "/health")
            assert health.body["bridge_runtime"]["telegram_runtime_tool_cache"] == {
                "cached": True,
                "hits": 1,
                "misses": 1,
                "loaded_from_override": True,
                "tool_count": 0,
            }
            assert health.body["bridge_runtime"]["staged_prompt_cache"] == {
                "cached": False,
                "hits": 0,
                "misses": 0,
                "family_count": 0,
            }
            assert all(request["body"]["temperature"] == 0.2 for request in state["requests"])
        finally:
            server.stop()


def test_provider_mode_bridge_request_fails_explicitly_when_runtime_catalog_is_unavailable():
    global server

    with run_mock_deepseek() as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_OPENCLAW_ENTRY_PATH": "/definitely/missing/openclaw-index.js",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "What movies do we have in Radarr?"},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "99",
                "X-Vicuna-Telegram-Conversation-Id": "tc3",
            })

            assert response.status_code == 500
            assert "unable to load Telegram runtime tool catalog" in response.body["error"]["message"]
            assert state["requests"] == []
        finally:
            server.stop()


def test_provider_mode_retries_invalid_bridge_scoped_family_selection_once():
    global server

    def rich_response(_payload):
        return {
            "id": "bridge-retry-rich-plan",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": rich_plan_text("Done."),
                    "reasoning_content": "Bridge delivery no longer requires selector retries.",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    with run_mock_deepseek(rich_response) as (base_url, state):
        server = make_provider_server(base_url, extra_env={
            "VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON": "[]",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Reply on Telegram."},
                ],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "101",
                "X-Vicuna-Telegram-Conversation-Id": "tc4",
            })

            assert response.status_code == 200
            assert response.body["vicuna_telegram_delivery"]["source"] == "rich_plan"
            assert len(state["requests"]) == 1
            assert state["requests"][0]["body"]["messages"][0]["content"].startswith("Bridge-scoped delivery contract:")
        finally:
            server.stop()


def test_provider_mode_cognitive_replay_admits_negative_episode():
    global server

    def negative_episode(_payload):
        return {
            "id": "chatcmpl-negative",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                    "reasoning_content": (
                        "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                        "There is an error and retry pressure."
                    ),
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
            },
        }

    with run_mock_deepseek(negative_episode) as (base_url, _):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "60000",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Handle the broken path."},
                ],
                "stream": False,
            })
            assert response.status_code == 200

            replay = server.make_request("GET", "/v1/emotive/cognitive-replay")
            assert replay.status_code == 200
            assert len(replay.body["entries"]) == 1
            entry = replay.body["entries"][0]
            assert entry["status"] == "open"
            assert entry["severity"]["negative_mass"] > 0.0
            assert entry["window_blocks"]
            assert "stuck" in entry["summary_excerpt"].lower()

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["emotive_runtime"]["cognitive_replay"]["open_count"] == 1
        finally:
            server.stop()


def test_provider_mode_idle_cognitive_replay_resolves_entry_without_recursive_admission():
    global server

    def replaying_response(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "cognitive replay mode" in first_message["content"].lower():
            return {
                "id": "chatcmpl-replay",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Better Path\n"
                            "1. Validate the failing assumption before acting.\n"
                            "2. Narrow the failing step.\n"
                            "3. Retry only after the inputs are corrected.\n\n"
                            "Why It Improves State\n"
                            "This reduces uncertainty and restores control."
                        ),
                        "reasoning_content": (
                            "First validate the assumptions, then isolate the failing step, "
                            "then retry with corrected inputs."
                        ),
                    },
                }],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 28,
                "total_tokens": 52,
            },
        }

        if first_message["role"] == "system" and "compressing a resolved cognitive replay" in first_message["content"].lower():
            return {
                "id": "chatcmpl-heuristic",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "heuristic_id": "heuristic_validate_before_retry",
                            "title": "Validate before retrying a failing path",
                            "trigger": {
                                "task_types": ["reasoning_trace"],
                                "tool_names": [],
                                "struct_tags": ["assistant_reasoning", "runtime_failure", "uncertainty_spike"],
                                "emotive_conditions": {
                                    "negative_mass": "elevated",
                                    "valence": "dropping",
                                    "dominance": "dropping",
                                },
                                "semantic_trigger_text": "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan.",
                            },
                            "diagnosis": {
                                "failure_mode": "retrying before validating the failing assumption keeps the system stuck",
                                "evidence": ["the replay improved after validation came first"],
                            },
                            "intervention": {
                                "constraints": [
                                    "Do not retry while the failing assumption remains unverified.",
                                ],
                                "preferred_actions": [
                                    "Validate the failing assumption, narrow the broken step, then retry.",
                                ],
                                "action_ranking_rules": [
                                    "Prefer validate-then-act over guess-then-retry.",
                                ],
                                "mid_reasoning_correction": "Pause and verify the broken assumption before taking the next action.",
                            },
                            "scope": {
                                "applies_when": [
                                    "the trace resembles the prior stuck retry loop",
                                ],
                                "avoid_when": [
                                    "the failure source is already validated",
                                ],
                            },
                            "confidence": {
                                "p_success": 0.81,
                                "calibration": "manual",
                                "notes": "Derived from one replay episode with improved valence and dominance.",
                            },
                        }),
                        "reasoning_content": "Compress the replay into a concrete validate-before-retry heuristic.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 44,
                    "total_tokens": 74,
                },
            }

        return {
            "id": "chatcmpl-negative",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                    "reasoning_content": (
                        "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                        "There is an error and retry pressure."
                    ),
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
            },
        }

    with run_mock_deepseek(replaying_response) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "250",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS": "2",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Handle the broken path."},
                ],
                "stream": False,
            })
            assert response.status_code == 200

            assert wait_for(lambda: len(state["requests"]) >= 3)
            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/cognitive-replay").body["worker"]["running"] is False
            )
            replay = server.make_request("GET", "/v1/emotive/cognitive-replay")
            assert replay.status_code == 200
            assert replay.body["worker"]["running"] is False
            assert len(replay.body["entries"]) == 1
            assert replay.body["entries"][0]["status"] == "resolved"
            assert replay.body["latest_result"]["comparison"]["improved"] is True
            assert replay.body["latest_result"]["replay_trace"]["cognitive_replay"] is True
            assert replay.body["latest_result"]["replay_trace"]["mode"] == "cognitive_replay"
            assert len(replay.body["entries"]) == 1

            outbound_replay = state["requests"][1]["body"]
            assert outbound_replay["messages"][0]["role"] == "system"
            assert "cognitive replay mode" in outbound_replay["messages"][0]["content"].lower()
        finally:
            server.stop()


def test_provider_mode_idle_cognitive_replay_defers_non_improving_entry():
    global server

    def replaying_response(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "cognitive replay mode" in first_message["content"].lower():
            return {
                "id": "chatcmpl-replay-bad",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "This is still wrong, broken, and unresolved. I am stuck, cannot act, "
                            "cannot decide, and the failure keeps getting worse with more error and retry pressure."
                        ),
                        "reasoning_content": (
                            "Maybe everything is wrong. I am unsure, stuck, cannot plan, cannot "
                            "act, and the error, failure, and retry loop keep getting worse."
                        ),
                    },
                }],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 12,
                    "total_tokens": 32,
                },
            }

        return {
            "id": "chatcmpl-negative",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                    "reasoning_content": (
                        "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                        "There is an error and retry pressure."
                    ),
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
            },
        }

    with run_mock_deepseek(replaying_response) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "250",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS": "1",
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Handle the broken path."},
                ],
                "stream": False,
            })
            assert response.status_code == 200

            assert wait_for(lambda: len(state["requests"]) >= 2)
            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/cognitive-replay").body["entries"][0]["status"] == "deferred"
            )
            replay = server.make_request("GET", "/v1/emotive/cognitive-replay")
            assert replay.status_code == 200
            assert replay.body["entries"][0]["status"] == "deferred"
            assert replay.body["latest_result"]["comparison"]["improved"] is False
            assert replay.body["latest_result"]["replay_trace"]["cognitive_replay"] is True
        finally:
            server.stop()


def test_provider_mode_resolved_replay_compresses_and_persists_heuristic_memory(tmp_path):
    global server
    heuristic_path = tmp_path / "heuristic-memory.json"

    def replaying_response(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "cognitive replay mode" in first_message["content"].lower():
            return {
                "id": "chatcmpl-replay",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Better Path\n"
                            "1. Validate the failing assumption before acting.\n"
                            "2. Narrow the failing step.\n"
                            "3. Retry only after the inputs are corrected.\n\n"
                            "Why It Improves State\n"
                            "This reduces uncertainty and restores control."
                        ),
                        "reasoning_content": (
                            "First validate the assumptions, then isolate the failing step, "
                            "then retry with corrected inputs."
                        ),
                    },
                }],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 28,
                    "total_tokens": 52,
                },
            }

        if first_message["role"] == "system" and "compressing a resolved cognitive replay" in first_message["content"].lower():
            return {
                "id": "chatcmpl-heuristic",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "heuristic_id": "heuristic_validate_before_retry",
                            "title": "Validate before retrying a failing path",
                            "trigger": {
                                "task_types": ["reasoning_trace"],
                                "tool_names": [],
                                "struct_tags": ["assistant_reasoning", "runtime_failure", "uncertainty_spike"],
                                "emotive_conditions": {
                                    "negative_mass": "elevated",
                                    "valence": "dropping",
                                    "dominance": "dropping",
                                },
                                "semantic_trigger_text": "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan.",
                            },
                            "diagnosis": {
                                "failure_mode": "retrying before validating the failing assumption keeps the system stuck",
                                "evidence": [
                                    "the bad path repeats uncertainty and failure pressure",
                                    "the better path validates assumptions before retrying",
                                ],
                            },
                            "intervention": {
                                "constraints": [
                                    "Do not retry while the failing assumption remains unverified.",
                                    "Insert one validation or narrowing step before acting again.",
                                ],
                                "preferred_actions": [
                                    "Validate the failing assumption, narrow the broken step, then retry.",
                                ],
                                "action_ranking_rules": [
                                    "Prefer validate-then-act over guess-then-retry.",
                                ],
                                "mid_reasoning_correction": "Pause and verify the broken assumption before taking the next action.",
                            },
                            "scope": {
                                "applies_when": [
                                    "the current trace shows uncertainty, failure pressure, and an impulse to retry",
                                ],
                                "avoid_when": [
                                    "the failure source is already validated and corrected",
                                ],
                            },
                            "confidence": {
                                "p_success": 0.81,
                                "calibration": "manual",
                                "notes": "Derived from one replay episode with improved valence and dominance.",
                            },
                        }),
                        "reasoning_content": "Compress the replay into a concrete validate-before-retry heuristic.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 36,
                    "completion_tokens": 54,
                    "total_tokens": 90,
                },
            }

        return {
            "id": "chatcmpl-negative",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                    "reasoning_content": (
                        "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                        "There is an error and retry pressure."
                    ),
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 20,
                "total_tokens": 32,
            },
        }

    with run_mock_deepseek(replaying_response) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "250",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS": "1",
            "VICUNA_HEURISTIC_MEMORY_PATH": str(heuristic_path),
        })
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Handle the broken path."},
                ],
                "stream": False,
            })
            assert response.status_code == 200

            assert wait_for(lambda: len(state["requests"]) >= 3)
            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/heuristics").body["record_count"] == 1
            )

            outbound_compression = state["requests"][2]["body"]
            assert outbound_compression["messages"][0]["role"] == "system"
            assert "compressing a resolved cognitive replay" in outbound_compression["messages"][0]["content"].lower()

            heuristics = server.make_request("GET", "/v1/emotive/heuristics")
            assert heuristics.status_code == 200
            assert heuristics.body["record_count"] == 1
            record = heuristics.body["records"][0]
            assert record["entry_id"]
            assert "assistant_reasoning:" in record["bad_path_text"]
            assert "validate the failing assumption" in record["better_path_content"].lower()
            assert record["heuristic"]["heuristic_id"] == "heuristic_validate_before_retry"
            assert record["heuristic"]["trigger"]["semantic_trigger_text"].startswith("Maybe this is wrong")
            assert heuristic_path.exists()

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["emotive_runtime"]["heuristic_memory"]["record_count"] == 1
        finally:
            server.stop()


def test_provider_mode_heuristic_guidance_injects_only_for_similar_trace_and_survives_restart(tmp_path):
    global server
    heuristic_path = tmp_path / "heuristic-memory.json"

    def replaying_response(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "cognitive replay mode" in first_message["content"].lower():
            return {
                "id": "chatcmpl-replay",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Better Path\n"
                            "1. Validate the failing assumption before acting.\n"
                            "2. Narrow the failing step.\n"
                            "3. Retry only after the inputs are corrected."
                        ),
                        "reasoning_content": (
                            "First validate the assumptions, then isolate the failing step, "
                            "then retry with corrected inputs."
                        ),
                    },
                }],
                "usage": {
                    "prompt_tokens": 24,
                    "completion_tokens": 22,
                    "total_tokens": 46,
                },
            }

        if first_message["role"] == "system" and "compressing a resolved cognitive replay" in first_message["content"].lower():
            return {
                "id": "chatcmpl-heuristic",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "heuristic_id": "heuristic_validate_before_retry",
                            "title": "Validate before retrying a failing path",
                            "trigger": {
                                "task_types": ["reasoning_trace"],
                                "tool_names": [],
                                "struct_tags": ["assistant_reasoning", "runtime_failure", "uncertainty_spike"],
                                "emotive_conditions": {
                                    "negative_mass": "elevated",
                                    "valence": "dropping",
                                    "dominance": "dropping",
                                },
                                "semantic_trigger_text": "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan.",
                            },
                            "diagnosis": {
                                "failure_mode": "retrying before validating the failing assumption keeps the system stuck",
                                "evidence": ["negative episode replay improved after validation first"],
                            },
                            "intervention": {
                                "constraints": [
                                    "Do not retry while the failing assumption remains unverified.",
                                ],
                                "preferred_actions": [
                                    "Validate the failing assumption, narrow the broken step, then retry.",
                                ],
                                "action_ranking_rules": [
                                    "Prefer validate-then-act over guess-then-retry.",
                                ],
                                "mid_reasoning_correction": "Pause and verify the broken assumption before taking the next action.",
                            },
                            "scope": {
                                "applies_when": [
                                    "the trace resembles the prior stuck retry loop",
                                ],
                                "avoid_when": [
                                    "the failure source is already validated",
                                ],
                            },
                            "confidence": {
                                "p_success": 0.81,
                                "calibration": "manual",
                                "notes": "Derived from one replay episode with improved valence and dominance.",
                            },
                        }),
                        "reasoning_content": "Compress the replay into a concrete validate-before-retry heuristic.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 44,
                    "total_tokens": 74,
                },
            }

        return {
            "id": "chatcmpl-generic",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Handled.",
                    "reasoning_content": "Respond directly.",
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }

    env = {
        "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "250",
        "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
        "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS": "1",
        "VICUNA_HEURISTIC_MEMORY_PATH": str(heuristic_path),
    }

    with run_mock_deepseek(replaying_response) as (base_url, state):
        server = make_provider_server(base_url, env)
        server.api_surface = "openai"
        try:
            server.start()

            seed = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Handle the broken path."},
                    {
                        "role": "assistant",
                        "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                        "reasoning_content": (
                            "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                            "There is an error and retry pressure."
                        ),
                    },
                ],
                "stream": False,
            })
            assert seed.status_code == 200

            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/heuristics").body["record_count"] == 1
            )

            similar = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                        "reasoning_content": (
                            "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                            "There is an error and retry pressure."
                        ),
                    },
                    {"role": "user", "content": "Continue from that stuck retry loop."},
                ],
                "stream": False,
            })
            assert similar.status_code == 200

            similar_outbound = state["requests"][-1]["body"]
            assert any(
                message["role"] == "system" and "[Critical Guidance | id=heuristic_validate_before_retry]" in message["content"]
                for message in similar_outbound["messages"]
            )

            dissimilar = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Summarize the healthy project status in one sentence."},
                ],
                "stream": False,
            })
            assert dissimilar.status_code == 200

            dissimilar_outbound = state["requests"][-1]["body"]
            assert not any(
                message["role"] == "system" and "[Critical Guidance | id=heuristic_validate_before_retry]" in message["content"]
                for message in dissimilar_outbound["messages"]
            )
        finally:
            server.stop()

    with run_mock_deepseek(replaying_response) as (base_url, state):
        server = make_provider_server(base_url, env)
        server.api_surface = "openai"
        try:
            server.start()

            heuristics = server.make_request("GET", "/v1/emotive/heuristics")
            assert heuristics.status_code == 200
            assert heuristics.body["record_count"] == 1

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I cannot resolve this problem. The failure is wrong and I am stuck.",
                        "reasoning_content": (
                            "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                            "There is an error and retry pressure."
                        ),
                    },
                    {"role": "user", "content": "Continue from that stuck retry loop."},
                ],
                "stream": False,
            })
            assert response.status_code == 200

            outbound = state["requests"][-1]["body"]
            assert any(
                message["role"] == "system" and "[Critical Guidance | id=heuristic_validate_before_retry]" in message["content"]
                for message in outbound["messages"]
            )
        finally:
            server.stop()


def test_provider_mode_ongoing_task_idle_surface_is_removed():
    global server

    def response_factory(_payload):
        return {
            "id": "chatcmpl-health-check",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }

    with run_mock_deepseek(response_factory) as (base_url, _state):
        server = make_provider_server(base_url, {
            "VICUNA_ONGOING_TASKS_ENABLED": "true",
        })
        server.api_surface = "openai"
        try:
            server.start()

            ongoing = server.make_request("GET", "/v1/emotive/ongoing-tasks")
            assert ongoing.status_code == 404

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert "ongoing_tasks" not in health.body["emotive_runtime"]
        finally:
            server.stop()


def test_provider_mode_ongoing_task_env_does_not_trigger_background_scheduler():
    global server

    def response_factory(_payload):
        return {
            "id": "chatcmpl-noop",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                },
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }

    with run_mock_deepseek(response_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_ENABLED": "false",
            "VICUNA_ONGOING_TASKS_ENABLED": "true",
            "VICUNA_ONGOING_TASKS_BASE_URL": base_url,
            "VICUNA_ONGOING_TASKS_AUTH_TOKEN": "test-key",
        })
        server.api_surface = "openai"
        try:
            server.start()
            time.sleep(0.35)
            assert requests_for_path(state, "/chat/completions") == []
        finally:
            server.stop()
