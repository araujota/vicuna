import json
import threading
import time
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
            assert health.body["emotive_runtime"]["cognitive_replay"]["enabled"] is True

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
            assert response.body["vicuna_emotive_trace"]["final_vad"]["style_guide"]["tone_label"]
            assert state["requests"][0]["body"]["max_tokens"] == 1024
        finally:
            server.stop()


def test_provider_mode_chat_completion_round_trips_tools_and_tool_history():
    global server

    def tool_response(payload):
        stage_prompt = payload["messages"][-1]["content"]
        if "choosing one tool family" in stage_prompt.lower():
            return {
                "id": "chatcmpl-stage-family",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "family": "Ongoing Tasks",
                        }),
                        "reasoning_content": "The ongoing task family is the right place to inspect what is due.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
            }
        if "choosing a method of the ongoing tasks" in stage_prompt.lower():
            return {
                "id": "chatcmpl-stage-method",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "method": "get_due",
                        }),
                        "reasoning_content": "The due-task getter will fetch the exact recurring task state I need.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
            }
        if "constructing a payload for the get_due" in stage_prompt.lower():
            return {
                "id": "chatcmpl-stage-payload",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "action": "submit",
                            "payload": {
                                "task_id": "task_123",
                            },
                        }),
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
            assert choice["finish_reason"] == "tool_calls"
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

            assert len(state["requests"]) == 3
            family_request = state["requests"][0]["body"]
            method_request = state["requests"][1]["body"]
            payload_request = state["requests"][2]["body"]

            assert "tools" not in family_request
            assert family_request["max_tokens"] == 1024
            assert family_request["response_format"] == {"type": "json_object"}
            assert family_request["messages"][2]["reasoning_content"] == request["messages"][1]["reasoning_content"]
            assert family_request["messages"][2]["tool_calls"] == request["messages"][1]["tool_calls"]
            assert family_request["messages"][3]["tool_call_id"] == "call_due_1"
            assert family_request["messages"][3]["content"] == "{\"tasks\":[]}"
            assert family_request["messages"][4]["role"] == "system"
            assert family_request["messages"][4]["content"].startswith("Current emotive guidance: valence=")
            assert "tone=" in family_request["messages"][4]["content"]
            assert family_request["messages"][-1]["role"] == "system"
            assert "choosing one tool family" in family_request["messages"][-1]["content"].lower()

            assert method_request["max_tokens"] == 1024
            assert method_request["response_format"] == {"type": "json_object"}
            assert method_request["messages"][-1]["role"] == "system"
            assert "choosing a method of the ongoing tasks" in method_request["messages"][-1]["content"].lower()

            assert payload_request["max_tokens"] == 1024
            assert payload_request["response_format"] == {"type": "json_object"}
            assert payload_request["messages"][-1]["role"] == "system"
            assert "constructing a payload for the get_due" in payload_request["messages"][-1]["content"].lower()
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
            assert outbound["messages"] == [
                {"role": "user", "content": "Summarize the project status."},
                {"role": "assistant", "content": "The project is on track."},
                {"role": "user", "content": "Now rewrite it more crisply."},
            ]
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
            assert first_outbound["max_tokens"] == 1024
            assert second_outbound["max_tokens"] == 1024
            assert first_outbound["thinking"] == {"type": "enabled"}
            assert second_outbound["thinking"] == {"type": "enabled"}
            assert first_outbound["messages"][3]["role"] == "system"
            assert second_outbound["messages"][3]["role"] == "system"
            assert first_outbound["messages"][3]["content"] != second_outbound["messages"][3]["content"]
        finally:
            server.stop()


def test_provider_mode_responses_route_emits_function_call_items():
    global server

    def tool_response(payload):
        stage_prompt = payload["messages"][-1]["content"]
        if "choosing one tool family" in stage_prompt.lower():
            return {
                "id": "resp-stage-family",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"family": "Ongoing Tasks"}),
                        "reasoning_content": "Recurring-task inspection belongs to the ongoing task family.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            }
        if "choosing a method of the ongoing tasks" in stage_prompt.lower():
            return {
                "id": "resp-stage-method",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "get_due"}),
                        "reasoning_content": "The due-task method is the right recurring-task operation.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            }
        if "constructing a payload for the get_due" in stage_prompt.lower():
            return {
                "id": "resp-stage-payload",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "action": "submit",
                            "payload": {},
                        }),
                        "reasoning_content": "No arguments are required for the due-task lookup.",
                    },
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                },
            }
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
        stage_prompt = payload["messages"][-1]["content"]
        if "choosing one tool family" in stage_prompt.lower():
            stage_counter["family"] += 1
            family = "Ongoing Tasks" if stage_counter["family"] == 1 else "Telegram"
            return {
                "id": f"stage-family-{stage_counter['family']}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"family": family}),
                        "reasoning_content": f"Choose the {family.lower()} family first.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the ongoing tasks" in stage_prompt.lower():
            return {
                "id": "stage-method-back",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "back"}),
                        "reasoning_content": "Back out and pick a better family.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the telegram" in stage_prompt.lower():
            return {
                "id": "stage-method-complete",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "complete"}),
                        "reasoning_content": "The tool loop is complete.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "tool loop is complete" in stage_prompt.lower():
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
        server = make_provider_server(base_url)
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
                        }
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
            assert all(request["body"]["max_tokens"] == 1024 for request in state["requests"])
            assert "choosing one tool family" in state["requests"][0]["body"]["messages"][-1]["content"].lower()
            assert "choosing a method of the ongoing tasks" in state["requests"][1]["body"]["messages"][-1]["content"].lower()
            assert "choosing one tool family" in state["requests"][2]["body"]["messages"][-1]["content"].lower()
            assert "choosing a method of the telegram" in state["requests"][3]["body"]["messages"][-1]["content"].lower()
            assert "tool loop is complete" in state["requests"][4]["body"]["messages"][-1]["content"].lower()
        finally:
            server.stop()


def test_provider_mode_retries_invalid_staged_family_selection_once():
    global server
    family_attempts = {"count": 0}

    def staged_response(payload):
        stage_prompt = payload["messages"][-1]["content"]
        if "choosing one tool family" in stage_prompt.lower():
            family_attempts["count"] += 1
            content = "" if family_attempts["count"] == 1 else json.dumps({"family": "Ongoing Tasks"})
            return {
                "id": f"retry-family-{family_attempts['count']}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "Pick the ongoing task family once the JSON is valid.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the ongoing tasks" in stage_prompt.lower():
            return {
                "id": "retry-family-method",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "get_due"}),
                        "reasoning_content": "Use the due-task getter.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "constructing a payload for the get_due" in stage_prompt.lower():
            return {
                "id": "retry-family-payload",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"action": "submit", "payload": {}}),
                        "reasoning_content": "No payload fields are required.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
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
        server = make_provider_server(base_url)
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
                        }
                    }
                }],
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ongoing_tasks_get_due"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert len(state["requests"]) == 4
            assert "Previous response error:" not in state["requests"][0]["body"]["messages"][-1]["content"]
            assert "Previous response error:" in state["requests"][1]["body"]["messages"][-1]["content"]
            assert "non-empty JSON object" in state["requests"][1]["body"]["messages"][-1]["content"]
        finally:
            server.stop()


def test_provider_mode_retries_invalid_staged_method_selection_once():
    global server
    method_attempts = {"count": 0}

    def staged_response(payload):
        stage_prompt = payload["messages"][-1]["content"]
        if "choosing one tool family" in stage_prompt.lower():
            return {
                "id": "retry-method-family",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"family": "Ongoing Tasks"}),
                        "reasoning_content": "Choose the ongoing tasks family.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the ongoing tasks" in stage_prompt.lower():
            method_attempts["count"] += 1
            content = "" if method_attempts["count"] == 1 else json.dumps({"method": "get_due"})
            return {
                "id": f"retry-method-{method_attempts['count']}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "Pick the due-task getter once the JSON is valid.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "constructing a payload for the get_due" in stage_prompt.lower():
            return {
                "id": "retry-method-payload",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"action": "submit", "payload": {}}),
                        "reasoning_content": "No payload fields are required.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
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
        server = make_provider_server(base_url)
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
                        }
                    }
                }],
                "stream": False,
            })

            assert response.status_code == 200
            assert response.body["choices"][0]["finish_reason"] == "tool_calls"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "ongoing_tasks_get_due"
            assert response.body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"
            assert len(state["requests"]) == 4
            assert "Previous response error:" not in state["requests"][1]["body"]["messages"][-1]["content"]
            assert "Previous response error:" in state["requests"][2]["body"]["messages"][-1]["content"]
            assert "non-empty JSON object" in state["requests"][2]["body"]["messages"][-1]["content"]
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


def test_provider_mode_normalizes_plain_text_bridge_completion_into_telegram_outbox():
    global server
    with run_mock_deepseek() as (base_url, _):
        server = make_provider_server(base_url)
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
                "source": "compat_plain_text",
            }

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"] == [{
                "sequence_number": 1,
                "kind": "message",
                "chat_scope": "12345",
                "telegram_method": "sendMessage",
                "telegram_payload": {
                    "text": "Hello from DeepSeek.",
                },
                "text": "Hello from DeepSeek.",
                "reply_to_message_id": 77,
                "dedupe_key": "bridge:tc1:77:sendMessage",
            }]
        finally:
            server.stop()


def test_provider_mode_executes_staged_telegram_tool_for_bridge_request():
    global server

    def staged_telegram_response(payload):
        stage_prompt = payload["messages"][-1]["content"].lower()
        if "choosing one tool family" in stage_prompt:
            return {
                "id": "stage-family-telegram",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"family": "Telegram"}),
                        "reasoning_content": "Use telegram.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the telegram" in stage_prompt:
            return {
                "id": "stage-method-telegram",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "relay"}),
                        "reasoning_content": "Relay the answer.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "constructing a payload for the relay method of the telegram tool family" in stage_prompt:
            return {
                "id": "stage-payload-telegram",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "action": "submit",
                            "payload": {
                                "request": {
                                    "method": "sendMessage",
                                    "payload": {
                                        "text": "<b>Ready</b>",
                                        "parse_mode": "HTML",
                                    },
                                },
                                "chat_scope": "12345",
                                "reply_to_message_id": 88,
                            },
                        }),
                        "reasoning_content": "Queue a structured Telegram reply.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        raise AssertionError(f"unexpected staged prompt: {payload['messages'][-1]['content']}")

    with run_mock_deepseek(staged_telegram_response) as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "Reply through Telegram."},
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "telegram_relay",
                        "description": "Queue one Telegram follow-up message through the provider-only bridge outbox.",
                        "parameters": {
                            "type": "object",
                            "description": "The Telegram relay payload.",
                            "anyOf": [
                                {"required": ["text"]},
                                {"required": ["request"]},
                            ],
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Optional simple plain-text Telegram reply.",
                                },
                                "request": {
                                    "type": "object",
                                    "description": "Optional structured Telegram Bot API send request.",
                                    "required": ["method", "payload"],
                                    "properties": {
                                        "method": {
                                            "type": "string",
                                            "description": "Allowed outbound Telegram method.",
                                        },
                                        "payload": {
                                            "type": "object",
                                            "description": "Telegram Bot API payload for the selected method.",
                                        },
                                    },
                                },
                                "chat_scope": {
                                    "type": "string",
                                    "description": "The Telegram chat scope to route the follow-up into.",
                                },
                                "reply_to_message_id": {
                                    "type": "integer",
                                    "description": "Optional Telegram message id to use as the reply anchor.",
                                },
                            },
                        },
                        "x-vicuna-family-id": "telegram",
                        "x-vicuna-family-name": "Telegram",
                        "x-vicuna-family-description": "Send direct user-facing follow-up messages through the Telegram bridge outbox.",
                        "x-vicuna-method-name": "relay",
                        "x-vicuna-method-description": "Queue one Telegram follow-up message as plain text or a structured Bot API send request.",
                    },
                }],
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
                "source": "tool_call",
            }

            assert len(state["requests"]) == 3
            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"] == [{
                "sequence_number": 1,
                "kind": "message",
                "chat_scope": "12345",
                "telegram_method": "sendMessage",
                "telegram_payload": {
                    "text": "<b>Ready</b>",
                    "parse_mode": "HTML",
                },
                "text": "<b>Ready</b>",
                "reply_to_message_id": 88,
                "dedupe_key": "bridge:tc2:88:sendMessage",
            }]
        finally:
            server.stop()


def test_provider_mode_allows_bridge_requests_to_receive_non_telegram_tool_calls():
    global server

    def staged_radarr_response(payload):
        stage_prompt = payload["messages"][-1]["content"].lower()
        if "choosing one tool family" in stage_prompt:
            return {
                "id": "stage-family-radarr",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"family": "Radarr"}),
                        "reasoning_content": "Use Radarr.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "choosing a method of the radarr" in stage_prompt:
            return {
                "id": "stage-method-radarr",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"method": "list_downloaded_movies"}),
                        "reasoning_content": "Inspect the downloaded movie list.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        if "constructing a payload for the list_downloaded_movies method of the radarr tool family" in stage_prompt:
            return {
                "id": "stage-payload-radarr",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "action": "submit",
                            "payload": {},
                        }),
                        "reasoning_content": "Submit the empty payload.",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }
        raise AssertionError(f"unexpected staged prompt: {payload['messages'][-1]['content']}")

    with run_mock_deepseek(staged_radarr_response) as (base_url, state):
        server = make_provider_server(base_url)
        server.api_surface = "openai"
        try:
            server.start()

            response = server.make_request("POST", "/v1/chat/completions", data={
                "model": "deepseek-reasoner",
                "messages": [
                    {"role": "user", "content": "What movies do we have in Radarr?"},
                ],
                "tools": [{
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
                }],
                "stream": False,
            }, headers={
                "X-Vicuna-Telegram-Chat-Id": "12345",
                "X-Vicuna-Telegram-Message-Id": "99",
                "X-Vicuna-Telegram-Conversation-Id": "tc3",
            })

            assert response.status_code == 200
            choice = response.body["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["content"] is None
            assert choice["message"]["reasoning_content"] == "Submit the empty payload."
            assert choice["message"]["tool_calls"] == [{
                "id": choice["message"]["tool_calls"][0]["id"],
                "type": "function",
                "function": {
                    "name": "radarr_list_downloaded_movies",
                    "arguments": "{}",
                },
            }]
            assert "vicuna_telegram_delivery" not in response.body
            assert len(state["requests"]) == 3

            outbox = server.make_request("GET", "/v1/telegram/outbox?after=0")
            assert outbox.status_code == 200
            assert outbox.body["items"] == []
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


def test_provider_mode_idle_ongoing_tasks_runs_due_task_and_updates_registry():
    global server
    task_text = "Every week, review the pending maintenance checklist and summarize the highest-risk item."
    registry_store = {
        "registry": {
            "schema_version": 1,
            "updated_at": "2026-03-01T00:00:00.000Z",
            "tasks": [
                {
                    "task_id": "task_weekly_review",
                    "task_text": task_text,
                    "frequency": {"interval": 1, "unit": "weeks"},
                    "created_at": "2026-02-01T00:00:00.000Z",
                    "updated_at": "2026-03-01T00:00:00.000Z",
                    "last_done_at": "2026-03-01T00:00:00.000Z",
                    "active": True,
                },
                {
                    "task_id": "task_not_due",
                    "task_text": "Every week, archive healthy logs.",
                    "frequency": {"interval": 1, "unit": "weeks"},
                    "created_at": "2026-03-20T00:00:00.000Z",
                    "updated_at": "2026-03-24T00:00:00.000Z",
                    "last_done_at": "2026-03-24T00:00:00.000Z",
                    "active": True,
                },
            ],
        },
    }

    def response_factory(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "one recurring ongoing task should run now" in first_message["content"].lower():
            return {
                "id": "chatcmpl-ongoing-decision",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "should_run": True,
                            "selected_task_id": "task_weekly_review",
                            "rationale": "The weekly review is overdue and should run before idle.",
                        }),
                        "reasoning_content": "The weekly task is overdue and more urgent than the healthy log archive.",
                    },
                }],
                "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
            }

        return {
            "id": "chatcmpl-ongoing-exec",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Highest risk: the maintenance checklist still contains an unverified backup failure.",
                    "reasoning_content": "Execute the stored weekly review task directly.",
                },
            }],
            "usage": {"prompt_tokens": 24, "completion_tokens": 18, "total_tokens": 42},
        }

    def route_factory(path, payload, _state):
        if path == "/v4/profile":
            return {
                "status": 200,
                "body": {
                    "searchResults": {
                        "results": [{
                            "title": "Ongoing task registry",
                            "metadata": {
                                "key": "ongoing-tasks-registry",
                                "title": "Ongoing task registry",
                            },
                            "memory": json.dumps(registry_store["registry"]),
                        }],
                    },
                },
            }
        if path == "/v4/memories":
            registry_store["registry"] = json.loads(payload["memories"][0]["content"])
            return {"status": 200, "body": {"ok": True}}
        return None

    with run_mock_deepseek(response_factory, route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "200",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_ONGOING_TASKS_ENABLED": "true",
            "VICUNA_ONGOING_TASKS_BASE_URL": base_url,
            "VICUNA_ONGOING_TASKS_AUTH_TOKEN": "test-key",
            "VICUNA_ONGOING_TASKS_POLL_MS": "1000",
            "VICUNA_ONGOING_TASKS_TIMEOUT_MS": "5000",
        })
        server.api_surface = "openai"
        try:
            server.start()

            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/ongoing-tasks").body["worker"]["last_completed_task_id"]
                == "task_weekly_review"
            )

            ongoing = server.make_request("GET", "/v1/emotive/ongoing-tasks")
            assert ongoing.status_code == 200
            assert ongoing.body["worker"]["last_decision"]["should_run"] is True
            assert ongoing.body["worker"]["last_decision"]["selected_task_id"] == "task_weekly_review"

            provider_requests = requests_for_path(state, "/chat/completions")
            decision_requests = [
                request for request in provider_requests
                if request["body"]["messages"][0]["role"] == "system"
                and "one recurring ongoing task should run now"
                in request["body"]["messages"][0]["content"].lower()
            ]
            execution_requests = [
                request for request in provider_requests
                if any(
                    message["role"] == "user" and message["content"] == task_text
                    for message in request["body"]["messages"]
                )
            ]
            assert len(decision_requests) == 1
            assert len(execution_requests) == 1

            saved_task = next(
                task for task in registry_store["registry"]["tasks"] if task["task_id"] == "task_weekly_review"
            )
            assert saved_task["last_done_at"] is not None
            assert saved_task["last_done_at"] != "2026-03-01T00:00:00.000Z"

            health = server.make_request("GET", "/health")
            assert health.status_code == 200
            assert health.body["emotive_runtime"]["ongoing_tasks"]["last_completed_task_id"] == "task_weekly_review"
        finally:
            server.stop()


def test_provider_mode_idle_ongoing_tasks_no_due_task_stays_idle():
    global server
    task_text = "Every month, review the quiet metrics dashboard and archive only if all checks remain green."
    registry_store = {
        "registry": {
            "schema_version": 1,
            "updated_at": "2026-03-24T00:00:00.000Z",
            "tasks": [{
                "task_id": "task_monthly_archive",
                "task_text": task_text,
                "frequency": {"interval": 1, "unit": "weeks"},
                "created_at": "2026-03-20T00:00:00.000Z",
                "updated_at": "2026-03-24T00:00:00.000Z",
                "last_done_at": "2026-03-24T00:00:00.000Z",
                "active": True,
            }],
        },
    }

    def response_factory(payload):
        first_message = payload["messages"][0]
        assert first_message["role"] == "system"
        assert "one recurring ongoing task should run now" in first_message["content"].lower()
        return {
            "id": "chatcmpl-ongoing-decision-idle",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "should_run": False,
                        "selected_task_id": "",
                        "rationale": "The registry says this task was just completed, so it should stay idle.",
                    }),
                    "reasoning_content": "The explicit timestamps say this task is not due yet.",
                },
            }],
            "usage": {"prompt_tokens": 22, "completion_tokens": 14, "total_tokens": 36},
        }

    def route_factory(path, payload, _state):
        if path == "/v4/profile":
            return {
                "status": 200,
                "body": {
                    "searchResults": {
                        "results": [{
                            "title": "Ongoing task registry",
                            "metadata": {
                                "key": "ongoing-tasks-registry",
                                "title": "Ongoing task registry",
                            },
                            "memory": json.dumps(registry_store["registry"]),
                        }],
                    },
                },
            }
        if path == "/v4/memories":
            pytest.fail("No ongoing-task completion should be persisted when nothing is due.")
        return None

    with run_mock_deepseek(response_factory, route_factory) as (base_url, state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "200",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_ONGOING_TASKS_ENABLED": "true",
            "VICUNA_ONGOING_TASKS_BASE_URL": base_url,
            "VICUNA_ONGOING_TASKS_AUTH_TOKEN": "test-key",
            "VICUNA_ONGOING_TASKS_POLL_MS": "1000",
            "VICUNA_ONGOING_TASKS_TIMEOUT_MS": "5000",
        })
        server.api_surface = "openai"
        try:
            server.start()

            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/ongoing-tasks").body["worker"]["last_decision"]["valid"]
                is True
            )

            ongoing = server.make_request("GET", "/v1/emotive/ongoing-tasks")
            assert ongoing.status_code == 200
            assert ongoing.body["worker"]["last_decision"]["should_run"] is False
            assert ongoing.body["worker"]["last_completed_task_id"] == ""

            provider_requests = requests_for_path(state, "/chat/completions")
            assert len(provider_requests) == 1
            assert not any(
                any(message["role"] == "user" and message["content"] == task_text for message in request["body"]["messages"])
                for request in provider_requests
            )
        finally:
            server.stop()


def test_provider_mode_idle_ongoing_task_execution_suppresses_replay_admission():
    global server
    task_text = "Every week, inspect the unstable retry path and recover it before the next idle cycle."
    registry_store = {
        "registry": {
            "schema_version": 1,
            "updated_at": "2026-03-10T00:00:00.000Z",
            "tasks": [{
                "task_id": "task_retry_recovery",
                "task_text": task_text,
                "frequency": {"interval": 1, "unit": "weeks"},
                "created_at": "2026-02-01T00:00:00.000Z",
                "updated_at": "2026-03-10T00:00:00.000Z",
                "last_done_at": "2026-03-10T00:00:00.000Z",
                "active": True,
            }],
        },
    }

    def response_factory(payload):
        first_message = payload["messages"][0]
        if first_message["role"] == "system" and "one recurring ongoing task should run now" in first_message["content"].lower():
            return {
                "id": "chatcmpl-ongoing-decision-negative",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "should_run": True,
                            "selected_task_id": "task_retry_recovery",
                            "rationale": "The weekly recovery task is overdue.",
                        }),
                        "reasoning_content": "This retry-path task has crossed its weekly cadence.",
                    },
                }],
                "usage": {"prompt_tokens": 24, "completion_tokens": 16, "total_tokens": 40},
            }

        return {
            "id": "chatcmpl-ongoing-negative-exec",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "I cannot resolve this retry path. The failure is wrong and I am stuck.",
                    "reasoning_content": (
                        "Maybe this is wrong. I am unsure, stuck, and cannot find a clear plan. "
                        "There is an error and retry pressure."
                    ),
                },
            }],
            "usage": {"prompt_tokens": 18, "completion_tokens": 20, "total_tokens": 38},
        }

    def route_factory(path, payload, _state):
        if path == "/v4/profile":
            return {
                "status": 200,
                "body": {
                    "searchResults": {
                        "results": [{
                            "title": "Ongoing task registry",
                            "metadata": {
                                "key": "ongoing-tasks-registry",
                                "title": "Ongoing task registry",
                            },
                            "memory": json.dumps(registry_store["registry"]),
                        }],
                    },
                },
            }
        if path == "/v4/memories":
            registry_store["registry"] = json.loads(payload["memories"][0]["content"])
            return {"status": 200, "body": {"ok": True}}
        return None

    with run_mock_deepseek(response_factory, route_factory) as (base_url, _state):
        server = make_provider_server(base_url, {
            "VICUNA_COGNITIVE_REPLAY_IDLE_AFTER_MS": "200",
            "VICUNA_COGNITIVE_REPLAY_POLL_MS": "50",
            "VICUNA_COGNITIVE_REPLAY_MAX_ATTEMPTS": "1",
            "VICUNA_ONGOING_TASKS_ENABLED": "true",
            "VICUNA_ONGOING_TASKS_BASE_URL": base_url,
            "VICUNA_ONGOING_TASKS_AUTH_TOKEN": "test-key",
            "VICUNA_ONGOING_TASKS_POLL_MS": "1000",
            "VICUNA_ONGOING_TASKS_TIMEOUT_MS": "5000",
        })
        server.api_surface = "openai"
        try:
            server.start()

            assert wait_for(
                lambda: server.make_request("GET", "/v1/emotive/ongoing-tasks").body["worker"]["last_completed_task_id"]
                == "task_retry_recovery"
            )

            replay = server.make_request("GET", "/v1/emotive/cognitive-replay")
            assert replay.status_code == 200
            assert replay.body["entries"] == []

            trace = server.make_request("GET", "/v1/emotive/trace/latest")
            assert trace.status_code == 200
            assert trace.body["trace"]["mode"] == "ongoing_task_execution"
            assert trace.body["trace"]["suppress_replay_admission"] is True
        finally:
            server.stop()
