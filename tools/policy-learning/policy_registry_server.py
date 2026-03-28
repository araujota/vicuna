#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from policy_registry import resolve_version_entry
from policy_trainer import load_policy_artifact, predict_action_with_confidence


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length).decode("utf-8")
    return json.loads(raw or "{}")


class PolicyRegistryService:
    def __init__(
        self,
        *,
        registry_dir: Path,
        model_name: str,
        default_alias: str,
        fallback_alias: str | None = None,
    ) -> None:
        self.registry_dir = registry_dir
        self.model_name = model_name
        self.default_alias = default_alias
        self.fallback_alias = fallback_alias

    def _resolve_entry(
        self,
        *,
        alias: str | None = None,
        version: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        if version is not None:
            entry = resolve_version_entry(
                self.registry_dir,
                self.model_name,
                version=version,
            )
            return entry, entry.get("resolved_alias") or ""

        requested_alias = alias or self.default_alias
        try:
            entry = resolve_version_entry(
                self.registry_dir,
                self.model_name,
                alias=requested_alias,
            )
            return entry, requested_alias
        except ValueError:
            if alias is not None or not self.fallback_alias or requested_alias == self.fallback_alias:
                raise
            entry = resolve_version_entry(
                self.registry_dir,
                self.model_name,
                alias=self.fallback_alias,
            )
            return entry, self.fallback_alias

    def health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": True,
            "model_name": self.model_name,
            "default_alias": self.default_alias,
            "fallback_alias": self.fallback_alias,
        }
        try:
            entry, resolved_alias = self._resolve_entry(alias=self.default_alias)
            artifact = load_policy_artifact(Path(entry["artifact_path"]))
            payload.update(
                {
                    "resolved_default_alias": resolved_alias,
                    "resolved_default_version": entry["resolved_version"],
                    "resolved_default_policy_version": artifact["policy_version"],
                }
            )
        except Exception as exc:
            payload["ok"] = False
            payload["error"] = str(exc)
        return payload

    def propose(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        requested_alias = request_payload.get("policy_alias")
        version_value = request_payload.get("artifact_version")
        version = int(version_value) if version_value is not None else None
        entry, resolved_alias = self._resolve_entry(alias=requested_alias, version=version)
        artifact = load_policy_artifact(Path(entry["artifact_path"]))
        prediction = predict_action_with_confidence(
            artifact=artifact,
            observation=request_payload.get("observation", {}),
            action_mask=request_payload.get("action_mask", {}),
        )
        return {
            "policy_version": artifact["policy_version"],
            "policy_alias": resolved_alias or requested_alias or self.default_alias,
            "artifact_version": entry["resolved_version"],
            "model_name": self.model_name,
            "action": prediction["action"],
            "confidence": prediction["confidence"],
        }


def create_server(
    *,
    host: str,
    port: int,
    registry_dir: Path,
    model_name: str,
    default_alias: str,
    fallback_alias: str | None = None,
) -> ThreadingHTTPServer:
    service = PolicyRegistryService(
        registry_dir=registry_dir,
        model_name=model_name,
        default_alias=default_alias,
        fallback_alias=fallback_alias,
    )

    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                payload = service.health()
                status = HTTPStatus.OK if payload.get("ok", False) else HTTPStatus.SERVICE_UNAVAILABLE
                self._write_json(int(status), payload)
                return
            self._write_json(int(HTTPStatus.NOT_FOUND), {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/policy/propose":
                self._write_json(int(HTTPStatus.NOT_FOUND), {"error": "not found"})
                return
            try:
                payload = _read_json_body(self)
                response = service.propose(payload)
            except ValueError as exc:
                self._write_json(int(HTTPStatus.NOT_FOUND), {"error": str(exc)})
                return
            except json.JSONDecodeError as exc:
                self._write_json(int(HTTPStatus.BAD_REQUEST), {"error": str(exc)})
                return
            except Exception as exc:
                self._write_json(int(HTTPStatus.INTERNAL_SERVER_ERROR), {"error": str(exc)})
                return
            self._write_json(int(HTTPStatus.OK), response)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return ThreadingHTTPServer((host, port), Handler)


def run_server(
    *,
    host: str,
    port: int,
    registry_dir: Path,
    model_name: str,
    default_alias: str,
    fallback_alias: str | None = None,
) -> None:
    server = create_server(
        host=host,
        port=port,
        registry_dir=registry_dir,
        model_name=model_name,
        default_alias=default_alias,
        fallback_alias=fallback_alias,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve registry-backed policy proposals")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18081, type=int)
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--default-alias", default="candidate")
    parser.add_argument("--fallback-alias")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        default_alias=args.default_alias,
        fallback_alias=args.fallback_alias,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
