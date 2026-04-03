#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from policy_registry import resolve_version_entry
from policy_trainer import load_policy_artifact, predict_action_with_confidence


def _load_artifact_for_serving(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version", ""))
    if schema_version == "vicuna.policy_artifact.v1":
        return "policy", load_policy_artifact(path)
    if schema_version == "vicuna.ppo_policy_artifact.v1":
        from ppo_policy import load_ppo_policy_artifact

        return "ppo_policy", load_ppo_policy_artifact(path)
    raise ValueError(f"unsupported artifact schema_version {schema_version}")


def _predict_from_artifact(
    artifact_kind: str,
    artifact: dict[str, Any],
    *,
    observation: dict[str, Any],
    action_mask: dict[str, Any],
) -> dict[str, Any]:
    if artifact_kind == "policy":
        return predict_action_with_confidence(
            artifact=artifact,
            observation=observation,
            action_mask=action_mask,
        )
    if artifact_kind == "ppo_policy":
        from ppo_policy import predict_ppo_action_with_confidence

        return predict_ppo_action_with_confidence(
            artifact=artifact,
            observation=observation,
            action_mask=action_mask,
        )
    raise ValueError(f"unsupported artifact kind {artifact_kind}")


def emit_log(event: str, **fields: Any) -> None:
    payload = {
        "schema_version": "vicuna.service_event.v1",
        "timestamp_ms": int(time.time() * 1000),
        "service": "policy-registry",
        "event": event,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


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
            artifact_kind, artifact = _load_artifact_for_serving(Path(entry["artifact_path"]))
            payload.update(
                {
                    "artifact_kind": artifact_kind,
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
        artifact_kind, artifact = _load_artifact_for_serving(Path(entry["artifact_path"]))
        prediction = _predict_from_artifact(
            artifact_kind,
            artifact,
            observation=request_payload.get("observation", {}),
            action_mask=request_payload.get("action_mask", {}),
        )
        return {
            "policy_version": artifact["policy_version"],
            "policy_alias": resolved_alias or requested_alias or self.default_alias,
            "artifact_version": entry["resolved_version"],
            "artifact_kind": artifact_kind,
            "model_name": self.model_name,
            "action": prediction["action"],
            "confidence": prediction["confidence"],
            "rollout": prediction.get("rollout"),
        }

    def resolve_artifact(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        requested_alias = request_payload.get("artifact_alias") or request_payload.get("policy_alias")
        version_value = request_payload.get("artifact_version")
        version = int(version_value) if version_value is not None else None
        artifact_kind = request_payload.get("artifact_kind")
        entry, resolved_alias = self._resolve_entry(alias=requested_alias, version=version)
        if artifact_kind and entry.get("artifact_kind") != artifact_kind:
            raise ValueError(
                f"resolved artifact kind {entry.get('artifact_kind')} does not match requested {artifact_kind}"
            )
        artifact_path = Path(entry["artifact_path"])
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        return {
            "model_name": self.model_name,
            "artifact_kind": entry.get("artifact_kind"),
            "artifact_alias": resolved_alias or requested_alias,
            "artifact_version": entry.get("artifact_version"),
            "registry_version": entry.get("resolved_version"),
            "artifact_path": str(artifact_path),
            "artifact": artifact_payload,
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
                emit_log("health_checked", path=self.path, status=int(status), ok=payload.get("ok", False))
                self._write_json(int(status), payload)
                return
            emit_log("not_found", path=self.path, method="GET")
            self._write_json(int(HTTPStatus.NOT_FOUND), {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/v1/policy/propose", "/v1/artifacts/resolve"}:
                emit_log("not_found", path=self.path, method="POST")
                self._write_json(int(HTTPStatus.NOT_FOUND), {"error": "not found"})
                return
            try:
                payload = _read_json_body(self)
                if self.path == "/v1/policy/propose":
                    response = service.propose(payload)
                    emit_log(
                        "proposal_served",
                        path=self.path,
                        request_id=payload.get("observation", {}).get("request_id"),
                        policy_alias=response.get("policy_alias"),
                        artifact_version=response.get("artifact_version"),
                        policy_version=response.get("policy_version"),
                    )
                else:
                    response = service.resolve_artifact(payload)
                    emit_log(
                        "artifact_resolved",
                        path=self.path,
                        artifact_kind=response.get("artifact_kind"),
                        artifact_alias=response.get("artifact_alias"),
                        artifact_version=response.get("artifact_version"),
                        registry_version=response.get("registry_version"),
                    )
            except ValueError as exc:
                emit_log("proposal_not_found", path=self.path, error=str(exc))
                self._write_json(int(HTTPStatus.NOT_FOUND), {"error": str(exc)})
                return
            except json.JSONDecodeError as exc:
                emit_log("proposal_bad_request", path=self.path, error=str(exc))
                self._write_json(int(HTTPStatus.BAD_REQUEST), {"error": str(exc)})
                return
            except Exception as exc:
                emit_log("proposal_failed", path=self.path, error=str(exc))
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
    emit_log(
        "service_starting",
        host=host,
        port=port,
        model_name=model_name,
        default_alias=default_alias,
        fallback_alias=fallback_alias,
    )
    server = create_server(
        host=host,
        port=port,
        registry_dir=registry_dir,
        model_name=model_name,
        default_alias=default_alias,
        fallback_alias=fallback_alias,
    )
    try:
        emit_log("service_ready", host=host, port=port)
        server.serve_forever()
    finally:
        emit_log("service_stopped", host=host, port=port)
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
