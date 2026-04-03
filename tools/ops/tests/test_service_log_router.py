#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
OPS_ROOT = TESTS_ROOT.parent
if str(OPS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPS_ROOT))

from service_log_router import (  # noqa: E402
    DailyJsonlWriter,
    build_record,
    parse_log_line,
    prune_service_logs,
    run_child,
)


def test_parse_runtime_request_trace_extracts_request_fields():
    line = 'srv  runtime_request_trace_log: request_trace: {"request_id":"req_123","component":"runtime","event":"request_completed","chat_id":"1"}'
    parsed = parse_log_line("runtime", line)
    assert parsed["event"] == "request_completed"
    assert parsed["request_id"] == "req_123"
    assert parsed["component"] == "runtime"
    assert parsed["chat_id"] == "1"


def test_build_record_normalizes_structured_json():
    record = build_record(
        service="telegram-bridge",
        invocation_id="inv_1",
        pid=123,
        stream="stdout",
        line='{"event":"vicuna_request_finished","requestId":"req_1","conversationId":"tc1","jobId":"tv_1","message":"done"}',
        timestamp=datetime(2026, 3, 29, 12, 0, 0).astimezone(),
    )
    assert record["event"] == "vicuna_request_finished"
    assert record["request_id"] == "req_1"
    assert record["conversation_id"] == "tc1"
    assert record["job_id"] == "tv_1"
    assert record["message"] == "done"


def test_daily_writer_updates_current_symlink_and_prunes_old_logs(tmp_path: Path):
    log_root = tmp_path / "logs"
    service = "runtime"
    for day in range(1, 10):
        path = log_root / service
        path.mkdir(parents=True, exist_ok=True)
        (path / f"2026-03-0{day}.jsonl").write_text("{}\n", encoding="utf-8")

    prune_service_logs(log_root, service, retention_days=7, today=date(2026, 3, 9))
    remaining = sorted(entry.name for entry in (log_root / service).glob("*.jsonl") if entry.name != "current.jsonl")
    assert remaining == [
        "2026-03-03.jsonl",
        "2026-03-04.jsonl",
        "2026-03-05.jsonl",
        "2026-03-06.jsonl",
        "2026-03-07.jsonl",
        "2026-03-08.jsonl",
        "2026-03-09.jsonl",
    ]

    writer = DailyJsonlWriter(service=service, log_root=log_root, retention_days=7)
    writer.write(
        build_record(
            service=service,
            invocation_id="inv_1",
            pid=1,
            stream="stdout",
            line='{"event":"smoke","message":"ok"}',
            timestamp=datetime(2026, 3, 9, 12, 0, 0).astimezone(),
        ),
        timestamp=datetime(2026, 3, 9, 12, 0, 0).astimezone(),
    )
    writer.close()
    assert os.readlink(log_root / service / "current.jsonl") == "2026-03-09.jsonl"


def test_run_child_writes_structured_logs(tmp_path: Path):
    log_root = tmp_path / "logs"
    command = [
        sys.executable,
        "-c",
        "import sys; print('{\"event\":\"hello\",\"request_id\":\"req_smoke\"}'); print('stderr line', file=sys.stderr)",
    ]
    returncode = run_child("ops", log_root, 7, command)
    assert returncode == 0

    log_path = log_root / "ops" / "current.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(item["event"] == "process_started" for item in lines)
    assert any(item["event"] == "hello" and item["request_id"] == "req_smoke" for item in lines)
    assert any(item["stream"] == "stderr" and item["message"] == "stderr line" for item in lines)
    assert any(item["event"] == "process_exited" for item in lines)
