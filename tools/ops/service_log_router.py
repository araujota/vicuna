#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vicuna.service_log_record.v1"
EXTRACTABLE_FIELDS = {
    "component",
    "route",
    "status",
    "stage",
    "delivery_mode",
    "error",
    "elapsed_ms",
    "elapsedMs",
}
CANONICAL_FIELD_MAP = {
    "requestId": "request_id",
    "traceId": "trace_id",
    "chatId": "chat_id",
    "conversationId": "conversation_id",
    "messageId": "message_id",
    "sequenceNumber": "sequence_number",
    "jobId": "job_id",
    "toolName": "tool_name",
    "policyVersion": "policy_version",
    "artifactVersion": "artifact_version",
    "operationId": "operation_id",
    "request_id": "request_id",
    "trace_id": "trace_id",
    "chat_id": "chat_id",
    "conversation_id": "conversation_id",
    "message_id": "message_id",
    "sequence_number": "sequence_number",
    "job_id": "job_id",
    "tool_name": "tool_name",
    "policy_version": "policy_version",
    "artifact_version": "artifact_version",
    "operation_id": "operation_id",
    "event": "event",
    "message": "message",
    "level": "level",
    "service": "service",
}
KNOWN_TOP_LEVEL_FIELDS = set(CANONICAL_FIELD_MAP.values()) | {
    "schema_version",
    "timestamp_ms",
    "timestamp",
    "invocation_id",
    "pid",
    "stream",
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _isoformat(ts: datetime) -> str:
    return ts.isoformat(timespec="milliseconds")


def _service_dir(log_root: Path, service: str) -> Path:
    return log_root / service


def _dated_log_path(log_root: Path, service: str, when: date) -> Path:
    return _service_dir(log_root, service) / f"{when.isoformat()}.jsonl"


def _current_log_symlink(log_root: Path, service: str) -> Path:
    return _service_dir(log_root, service) / "current.jsonl"


def _ensure_service_dir(log_root: Path, service: str) -> Path:
    path = _service_dir(log_root, service)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sync_current_symlink(log_root: Path, service: str, target: Path) -> None:
    link = _current_log_symlink(log_root, service)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)


def prune_service_logs(log_root: Path, service: str, retention_days: int, today: date | None = None) -> None:
    today = today or _now().date()
    cutoff = today - timedelta(days=max(retention_days - 1, 0))
    service_dir = _ensure_service_dir(log_root, service)
    for entry in service_dir.glob("*.jsonl"):
        if entry.name == "current.jsonl":
            continue
        try:
            entry_date = date.fromisoformat(entry.stem)
        except ValueError:
            continue
        if entry_date < cutoff:
            entry.unlink(missing_ok=True)


def _normalize_json_record(value: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    for key, raw_value in value.items():
        if key == "payload":
            record["payload"] = raw_value
            continue
        canonical = CANONICAL_FIELD_MAP.get(key)
        if canonical:
            record[canonical] = raw_value
        elif key in EXTRACTABLE_FIELDS:
            record[key] = raw_value
        else:
            payload[key] = raw_value
    if payload:
        record["payload"] = payload
    return record


def parse_log_line(service: str, raw_line: str) -> dict[str, Any]:
    line = raw_line.rstrip("\n")
    stripped = line.strip()
    if not stripped:
        return {"event": "blank_line", "message": ""}

    for candidate in (stripped,):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            normalized = _normalize_json_record(decoded)
            normalized.setdefault("event", "structured_line")
            normalized.setdefault("message", stripped)
            return normalized

    if "request_trace:" in line:
        suffix = line.split("request_trace:", 1)[1].strip()
        try:
            decoded = json.loads(suffix)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            normalized = _normalize_json_record(decoded)
            normalized.setdefault("event", str(decoded.get("event", "request_trace")))
            normalized.setdefault("message", line)
            normalized.setdefault("component", decoded.get("component"))
            return normalized

    if stripped.startswith("[") and "] " in stripped:
        prefix, rest = stripped.split("] ", 1)
        rest = rest.strip()
        try:
            decoded = json.loads(rest)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            normalized = _normalize_json_record(decoded)
            normalized.setdefault("event", "structured_line")
            normalized.setdefault("message", prefix.lstrip("["))
            payload = normalized.get("payload", {})
            if isinstance(payload, dict) and "prefix" not in payload:
                payload["prefix"] = prefix.lstrip("[")
                normalized["payload"] = payload
            return normalized

    return {"event": "raw_line", "message": line}


def build_record(
    *,
    service: str,
    invocation_id: str | None,
    pid: int | None,
    stream: str,
    line: str,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    timestamp = timestamp or _now()
    parsed = parse_log_line(service, line)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_ms": int(timestamp.timestamp() * 1000),
        "timestamp": _isoformat(timestamp),
        "service": service,
        "invocation_id": invocation_id,
        "pid": pid,
        "stream": stream,
        "event": parsed.pop("event", "raw_line"),
        "message": parsed.pop("message", line.rstrip("\n")),
    }
    for key, value in parsed.items():
        if value is None:
            continue
        record[key] = value
    return record


@dataclass
class DailyJsonlWriter:
    service: str
    log_root: Path
    retention_days: int

    def __post_init__(self) -> None:
        self._current_date: date | None = None
        self._handle = None
        _ensure_service_dir(self.log_root, self.service)

    def _ensure_handle(self, timestamp: datetime) -> None:
        target_date = timestamp.date()
        if self._handle is not None and self._current_date == target_date:
            return
        if self._handle is not None:
            self._handle.close()
        target_path = _dated_log_path(self.log_root, self.service, target_date)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = target_path.open("a", encoding="utf-8")
        self._current_date = target_date
        _sync_current_symlink(self.log_root, self.service, target_path)
        prune_service_logs(self.log_root, self.service, self.retention_days, today=target_date)

    def write(self, record: dict[str, Any], timestamp: datetime | None = None) -> None:
        timestamp = timestamp or _now()
        self._ensure_handle(timestamp)
        assert self._handle is not None
        self._handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def append_event(
    *,
    service: str,
    log_root: Path,
    retention_days: int,
    event: str,
    message: str,
    invocation_id: str | None = None,
    pid: int | None = None,
    fields: dict[str, Any] | None = None,
) -> None:
    writer = DailyJsonlWriter(service=service, log_root=log_root, retention_days=retention_days)
    payload = {
        "event": event,
        "message": message,
    }
    if fields:
        payload.update(fields)
    record = build_record(
        service=service,
        invocation_id=invocation_id,
        pid=pid,
        stream="stdout",
        line=json.dumps(payload),
    )
    writer.write(record)
    writer.close()


def _reader(stream_name: str, handle, output: queue.Queue[tuple[str, str]]) -> None:
    try:
        for line in handle:
            output.put((stream_name, line))
    finally:
        handle.close()


def run_child(service: str, log_root: Path, retention_days: int, command: list[str]) -> int:
    invocation_id = os.environ.get("INVOCATION_ID") or os.environ.get("VICUNA_INVOCATION_ID")
    writer = DailyJsonlWriter(service=service, log_root=log_root, retention_days=retention_days)
    writer.write(
        build_record(
            service=service,
            invocation_id=invocation_id,
            pid=None,
            stream="stdout",
            line=json.dumps({
                "event": "process_started",
                "message": "service process started",
                "payload": {"command": command},
            }),
        )
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    output: queue.Queue[tuple[str, str]] = queue.Queue()
    threads = [
        threading.Thread(target=_reader, args=("stdout", process.stdout, output), daemon=True),
        threading.Thread(target=_reader, args=("stderr", process.stderr, output), daemon=True),
    ]
    for thread in threads:
        thread.start()

    while True:
        try:
            stream, line = output.get(timeout=0.1)
        except queue.Empty:
            if process.poll() is not None and all(not thread.is_alive() for thread in threads):
                break
            continue
        writer.write(
            build_record(
                service=service,
                invocation_id=invocation_id,
                pid=process.pid,
                stream=stream,
                line=line,
            )
        )

    returncode = process.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    writer.write(
        build_record(
            service=service,
            invocation_id=invocation_id,
            pid=process.pid,
            stream="stdout",
            line=json.dumps({
                "event": "process_exited",
                "message": "service process exited",
                "payload": {"returncode": returncode},
            }),
        )
    )
    writer.close()
    return returncode


def _parse_field_assignment(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"invalid field assignment: {value}")
    key, raw = value.split("=", 1)
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route retained Vicuña service logs into daily JSONL files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--service", required=True)
    run_parser.add_argument("--log-root", default=os.environ.get("VICUNA_LOG_ROOT", "/var/log/vicuna"))
    run_parser.add_argument("--retention-days", type=int, default=int(os.environ.get("VICUNA_LOG_RETENTION_DAYS", "7")))
    run_parser.add_argument("cmd", nargs=argparse.REMAINDER)

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("--service", required=True)
    event_parser.add_argument("--event", required=True)
    event_parser.add_argument("--message", required=True)
    event_parser.add_argument("--log-root", default=os.environ.get("VICUNA_LOG_ROOT", "/var/log/vicuna"))
    event_parser.add_argument("--retention-days", type=int, default=int(os.environ.get("VICUNA_LOG_RETENTION_DAYS", "7")))
    event_parser.add_argument("--field", action="append", default=[])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            parser.error("run requires a child command after --")
        return run_child(
            service=args.service,
            log_root=Path(args.log_root),
            retention_days=args.retention_days,
            command=cmd,
        )

    if args.command == "event":
        fields: dict[str, Any] = {}
        for assignment in args.field:
            key, value = _parse_field_assignment(assignment)
            fields[key] = value
        append_event(
            service=args.service,
            log_root=Path(args.log_root),
            retention_days=args.retention_days,
            event=args.event,
            message=args.message,
            invocation_id=os.environ.get("INVOCATION_ID") or os.environ.get("VICUNA_INVOCATION_ID"),
            pid=os.getpid(),
            fields=fields,
        )
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
