import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import pytest

from utils import ServerProcess

server: ServerProcess


def make_provider_server(base_url: str, extra_env: Optional[Dict[str, str]] = None) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_alias = "deepseek-reasoner"
    server.no_webui = True
    server.extra_env = {
        "VICUNA_DEEPSEEK_API_KEY": "test-key",
        "VICUNA_DEEPSEEK_BASE_URL": base_url,
        "VICUNA_DEEPSEEK_MODEL": "deepseek-reasoner",
        "VICUNA_DEEPSEEK_TIMEOUT_MS": "5000",
    }
    if extra_env:
        server.extra_env.update(extra_env)
    return server


@contextmanager
def run_mock_deepseek(response_factory=None):
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            state["requests"].append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": payload,
            })

            if self.path != "/chat/completions":
                self.send_response(404)
                self.end_headers()
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
            assert health.body["provider"]["model"] == "deepseek-reasoner"
            assert "runtime_persistence" not in health.body
            assert health.body["proactive_mailbox"]["live_stream_connected"] is False
            assert health.body["emotive_runtime"]["enabled"] is True
            assert health.body["emotive_runtime"]["embedding_backend"]["mode"] == "lexical_only"

            models = server.make_request("GET", "/v1/models")
            assert models.status_code == 200
            assert models.body["data"][0]["id"] == "deepseek-reasoner"
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
            assert response.body["model"] == "deepseek-reasoner"
            trace = response.body["vicuna_emotive_trace"]
            assert trace["embedding_mode"] == "lexical_only"
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
            assert outbound["body"]["model"] == "deepseek-reasoner"
            assert outbound["body"]["stream"] is True
            assert outbound["body"]["messages"] == [
                {"role": "user", "content": "Hello"},
            ]

            latest = server.make_request("GET", "/v1/emotive/trace/latest")
            assert latest.status_code == 200
            assert latest.body["trace"]["trace_id"] == trace["trace_id"]
            assert latest.body["retained_turns"] == 1
        finally:
            server.stop()


def test_provider_mode_responses_route_emits_reasoning_item():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/responses", data={
                "model": "deepseek-reasoner",
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
            assert response.body["vicuna_emotive_trace"]["final_vad"]["style_guide"]["tone_label"]
        finally:
            server.stop()


def test_provider_mode_chat_completion_round_trips_tools_and_tool_history():
    global server

    def tool_response(_payload):
        return {
            "id": "chatcmpl-mock-tools",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "The next step is to inspect due recurring tasks.",
                    "tool_calls": [{
                        "id": "call_due_2",
                        "type": "function",
                        "function": {
                            "name": "ongoing_tasks_get_due",
                            "arguments": "{\"task_id\":\"task_123\"}",
                        },
                    }],
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
                "messages": [
                    {"role": "user", "content": "Check the due ongoing tasks."},
                    {
                        "role": "assistant",
                        "content": None,
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
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["content"] is None
            assert choice["message"]["reasoning_content"] == "The next step is to inspect due recurring tasks."
            assert choice["message"]["tool_calls"] == [{
                "id": "call_due_2",
                "type": "function",
                "function": {
                    "name": "ongoing_tasks_get_due",
                    "arguments": "{\"task_id\":\"task_123\"}",
                },
            }]

            outbound = state["requests"][0]["body"]
            assert outbound["tools"] == request["tools"]
            assert outbound["tool_choice"] == "auto"
            assert outbound["parallel_tool_calls"] is False
            assert outbound["messages"][1]["tool_calls"] == request["messages"][1]["tool_calls"]
            assert outbound["messages"][2]["tool_call_id"] == "call_due_1"
            assert outbound["messages"][2]["content"] == "{\"tasks\":[]}"
        finally:
            server.stop()


def test_provider_mode_responses_route_emits_function_call_items():
    global server

    def tool_response(_payload):
        return {
            "id": "resp-mock-tools",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should poll the recurring task set first.",
                    "tool_calls": [{
                        "id": "call_due_9",
                        "type": "function",
                        "function": {
                            "name": "ongoing_tasks_get_due",
                            "arguments": "{}",
                        },
                    }],
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
            assert response.body["output"][1]["call_id"] == "call_due_9"
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
                "text": "Follow-up ready.",
                "dedupe_key": "final-followup",
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
