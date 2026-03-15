import pytest
import requests
import time
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    assert res.body["status"] == "ok"
    assert res.body["state"] == "ready"
    assert "runtime_persistence" in res.body


def test_server_health_exposes_runtime_observability():
    global server
    server.extra_env = {
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "health check proactive message",
    }
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    assert res.body["waiting_active_tasks"] == 0
    assert res.body["external_bash_pending"] == 0
    assert res.body["external_hard_memory_pending"] == 0
    assert res.body["runtime_persistence"]["enabled"] is False
    assert res.body["proactive_mailbox"]["stored_responses"] == 1
    assert res.body["proactive_mailbox"]["publish_total"] == 1
    assert res.body["proactive_mailbox"]["live_stream_connected"] is False


def test_server_metrics_exposes_external_runtime_counters():
    global server
    server.server_metrics = True
    server.extra_env = {
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "metrics proactive message",
    }
    server.start()
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert "llamacpp:external_bash_dispatch_total" in res.body
    assert "llamacpp:external_hard_memory_dispatch_total" in res.body
    assert "llamacpp:waiting_active_tasks" in res.body
    assert "llamacpp:runtime_persistence_healthy" in res.body
    assert "llamacpp:proactive_publish_total" in res.body
    assert "llamacpp:proactive_responses" in res.body
    assert "llamacpp:proactive_live_stream_connected" in res.body


def test_runtime_snapshot_survives_restart(tmp_path):
    global server
    snapshot_path = tmp_path / "runtime-state.json"
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "persisted proactive message",
    }
    server.api_surface = "openai"
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": server.model_alias,
        "input": [
            {"role": "user", "content": "Hello there"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.0,
    })
    assert res.status_code == 200

    for _ in range(20):
        if snapshot_path.exists():
            break
        time.sleep(0.25)
    assert snapshot_path.exists()

    server.stop()

    server = ServerPreset.tinyllama2()
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
    }
    server.api_surface = "openai"
    server.start()
    health = server.make_request("GET", "/health")
    assert health.status_code == 200
    assert health.body["runtime_persistence"]["enabled"] is True
    assert health.body["runtime_persistence"]["restore_attempted"] is True
    assert health.body["runtime_persistence"]["healthy"] is True
    assert health.body["proactive_mailbox"]["stored_responses"] == 1
    events = server.collect_named_sse_events(
        "GET",
        "/v1/responses/stream?after=0",
        stop_when=lambda event_name, _: event_name == "response.completed",
    )
    assert events[-1][1]["response"]["output"][0]["content"][0]["text"] == "persisted proactive message"


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == server.n_slots
    default_val = res.body["default_generation_settings"]
    assert server.n_ctx is not None and server.n_slots is not None
    assert default_val["n_ctx"] == server.n_ctx / server.n_slots
    assert default_val["params"]["seed"] == server.seed


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots():
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501 # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.n_slots = 2
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) == server.n_slots
    assert server.n_ctx is not None and server.n_slots is not None
    assert res.body[0]["n_ctx"] == server.n_ctx / server.n_slots
    assert "params" not in res.body[0]


def test_load_split_model():
    global server
    server.offline = False
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_webui():
    global server
    # default: webui enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-webui
    server.no_webui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404


def test_server_model_aliases_and_tags():
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
