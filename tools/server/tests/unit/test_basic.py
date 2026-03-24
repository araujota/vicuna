import pytest
import requests
import time
import json
from pathlib import Path
from utils import ServerPreset, match_regex

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
    assert res.body["foreground_runtime"]["waiting_on_tool_result"] == 0
    assert res.body["foreground_runtime"]["waiting_on_deferred_delivery"] == 0
    assert res.body["foreground_runtime"]["waiting_emit_response"] == 0
    assert res.body["foreground_runtime"]["waiting_post_tool_decision"] == 0
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


def test_runtime_snapshot_migrates_legacy_bash_allowlist(tmp_path):
    global server
    snapshot_path = tmp_path / "runtime-state.json"
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
    }
    server.start()

    for _ in range(20):
        if snapshot_path.exists():
            break
        time.sleep(0.25)
    assert snapshot_path.exists()
    server.stop()

    snapshot = json.loads(snapshot_path.read_text())
    snapshot["bash_tool_config"]["allowed_commands"] = (
        "pwd,ls,find,rg,cat,head,tail,grep,git,tavily-web-search,"
        "tools/openclaw-harness/bin/tavily-web-search"
    )
    snapshot_path.write_text(json.dumps(snapshot))

    server = ServerPreset.tinyllama2()
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
        "VICUNA_BASH_TOOL_ALLOWED_COMMANDS": "",
    }
    server.start()
    health = server.make_request("GET", "/health")
    assert health.status_code == 200
    assert health.body["runtime_persistence"]["restore_attempted"] is True
    assert health.body["runtime_persistence"]["healthy"] is True
    server.stop()

    migrated_snapshot = json.loads(snapshot_path.read_text())
    assert migrated_snapshot["bash_tool_config"]["allowed_commands"] == ""


def test_runtime_snapshot_keeps_env_bash_policy_authoritative(tmp_path):
    global server
    snapshot_path = tmp_path / "runtime-state.json"
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "seed runtime snapshot",
    }
    server.start()

    for _ in range(20):
        if snapshot_path.exists():
            break
        time.sleep(0.25)
    assert snapshot_path.exists()
    server.stop()

    snapshot = json.loads(snapshot_path.read_text())
    snapshot["bash_tool_config"]["enabled"] = False
    snapshot["bash_tool_config"]["allowed_commands"] = "stale-command"
    snapshot["bash_tool_config"]["working_directory"] = "/tmp/stale-bash-workdir"
    snapshot_path.write_text(json.dumps(snapshot))

    repo_root = Path(__file__).resolve().parents[4]
    server = ServerPreset.tinyllama2()
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "reapply env policy",
        "VICUNA_BASH_TOOL_ENABLED": "1",
        "VICUNA_BASH_TOOL_ALLOWED_COMMANDS": "pwd",
        "VICUNA_BASH_TOOL_WORKDIR": str(repo_root),
    }
    server.start()
    health = server.make_request("GET", "/health")
    assert health.status_code == 200
    assert health.body["runtime_persistence"]["restore_attempted"] is True
    assert health.body["runtime_persistence"]["healthy"] is True
    server.stop()

    restored_snapshot = json.loads(snapshot_path.read_text())
    assert restored_snapshot["bash_tool_config"]["enabled"] is True
    assert restored_snapshot["bash_tool_config"]["allowed_commands"] == "pwd"
    assert restored_snapshot["bash_tool_config"]["working_directory"] == str(repo_root)


def test_unified_provenance_repository_records_self_improvement_events(tmp_path):
    global server
    snapshot_path = tmp_path / "runtime-state.json"
    provenance_path = tmp_path / "runtime-provenance.jsonl"
    server.server_metrics = True
    server.api_surface = "openai"
    server.extra_env = {
        "VICUNA_RUNTIME_STATE_PATH": str(snapshot_path),
        "VICUNA_PROVENANCE_ENABLED": "1",
        "VICUNA_PROVENANCE_LOG_PATH": str(provenance_path),
    }
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": server.model_alias,
        "input": [
            {"role": "user", "content": "Summarize your current plan in one sentence."},
        ],
        "max_output_tokens": 12,
        "temperature": 0.0,
    })
    assert res.status_code == 200

    lines = []
    for _ in range(20):
        segment_paths = sorted(tmp_path.glob("runtime-provenance*.jsonl"))
        if not segment_paths:
            time.sleep(0.25)
            continue
        lines = []
        for segment_path in segment_paths:
            lines.extend(
                line for line in segment_path.read_text().splitlines() if line.strip()
            )
        if lines:
            break
        time.sleep(0.25)

    assert lines
    events = [json.loads(line) for line in lines]
    assert any(event["event_kind"] == "active_loop" for event in events)
    active_event = next(event for event in events if event["event_kind"] == "active_loop")
    assert active_event["schema_version"] == 1
    assert active_event["sequence"] >= 1
    assert active_event["session_id"].startswith("prov_")
    assert active_event["payload"]["self_model"]["extension_summary"]["active_count"] >= 0
    assert "functional" in active_event["payload"]
    assert "process_functional" in active_event["payload"]
    assert "plan" in active_event["payload"]["active_loop"]
    assert isinstance(active_event["payload"]["active_loop"]["plan"]["steps"], list)
    assert "candidates" in active_event["payload"]["active_loop"]

    active_final_events = [
        event for event in events
        if event["event_kind"] == "active_loop" and event.get("source") == "active_final"
    ]
    if active_final_events:
        narration = active_final_events[-1]["payload"]["extra"]["narration"]
        assert narration["resolved_text"]

    tool_call_events = [event for event in events if event["event_kind"] == "tool_call"]
    if tool_call_events:
        tool_call_event = tool_call_events[-1]
        assert "command" in tool_call_event["payload"]
        assert "tool_call" in tool_call_event["payload"]
        assert tool_call_event["payload"]["command"]["command_id"] >= 1

    health = server.make_request("GET", "/health")
    assert health.status_code == 200
    assert health.body["provenance_repository"]["enabled"] is True
    assert health.body["provenance_repository"]["healthy"] is True
    assert health.body["provenance_repository"]["path"] == str(provenance_path)
    assert health.body["provenance_repository"]["active_path"].startswith(str(tmp_path))
    assert health.body["provenance_repository"]["retention_ms"] == 48 * 60 * 60 * 1000
    assert health.body["provenance_repository"]["append_total"] >= 1

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert "llamacpp:provenance_append_total" in metrics.body
    assert "llamacpp:provenance_active_loop_total" in metrics.body
    assert "llamacpp:provenance_enabled" in metrics.body
    assert "llamacpp:provenance_self_state_allostatic_divergence" in metrics.body


def test_idle_startup_throttles_dmn_ticks(tmp_path):
    global server
    provenance_path = tmp_path / "runtime-provenance.jsonl"
    server.extra_env = {
        "VICUNA_PROVENANCE_ENABLED": "1",
        "VICUNA_PROVENANCE_LOG_PATH": str(provenance_path),
    }
    server.start()
    time.sleep(1.5)

    health = server.make_request("GET", "/health")
    assert health.status_code == 200
    assert health.body["provenance_repository"]["enabled"] is True
    assert health.body["provenance_repository"]["dmn_total"] <= 1


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
