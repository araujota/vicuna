#!/usr/bin/env python3
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request


UPSTREAM_URL = os.environ.get("VICUNA_RUNPOD_MISTRAL_UPSTREAM_URL", "").rstrip("/")
LISTEN_HOST = os.environ.get("VICUNA_RUNPOD_MISTRAL_RELAY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("VICUNA_RUNPOD_MISTRAL_RELAY_PORT", "18081"))
TIMEOUT_MS = int(os.environ.get("VICUNA_RUNPOD_MISTRAL_RELAY_TIMEOUT_MS", "1800000"))
AUTH_TOKEN = os.environ.get("VICUNA_RUNPOD_INFERENCE_AUTH_TOKEN", "")
DEFAULT_MODEL = os.environ.get("VICUNA_RUNPOD_MISTRAL_MODEL", "google/gemma-4-31B-it")
UPSTREAM_AUTH_TOKEN = os.environ.get("VICUNA_RUNPOD_MISTRAL_UPSTREAM_AUTH_TOKEN", "")


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _masked_url(url: str) -> str | None:
    if not url:
        return None
    if "://" not in url:
        return url
    scheme, remainder = url.split("://", 1)
    if "@" in remainder:
        _, remainder = remainder.rsplit("@", 1)
    return f"{scheme}://{remainder}"


def _sanitize_forward_body(body: dict) -> dict:
    prepared = dict(body)
    for key in (
        "x-vicuna-provider-max-tokens-override",
        "x-vicuna-provider-reasoning-budget-override",
        "reasoning_budget_tokens",
        "thinking",
        "relay_request_kind",
        "pod_policy_authority",
        "host_self_model",
        "original_body",
    ):
        prepared.pop(key, None)
    if not prepared.get("model"):
        prepared["model"] = DEFAULT_MODEL
    return prepared


def _prepare_forward_body(payload: dict) -> dict:
    if isinstance(payload.get("body"), dict):
        return _sanitize_forward_body(payload["body"])
    return _sanitize_forward_body(payload)


def _forward_json(path: str, payload: dict) -> tuple[int, dict]:
    if not UPSTREAM_URL:
        raise RuntimeError("VICUNA_RUNPOD_MISTRAL_UPSTREAM_URL is not configured")
    url = f"{UPSTREAM_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if UPSTREAM_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {UPSTREAM_AUTH_TOKEN}"
    req = request.Request(url, data=data, headers=headers, method="POST")
    timeout = max(1.0, TIMEOUT_MS / 1000.0)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = {}
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload


def _probe_health() -> tuple[int, dict]:
    if not UPSTREAM_URL:
        return 503, {"ok": False, "message": "upstream URL is not configured"}
    url = f"{UPSTREAM_URL}/health"
    headers = {}
    if UPSTREAM_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {UPSTREAM_AUTH_TOKEN}"
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return 503, {"ok": False, "message": str(exc)}


def _adapt_openai_to_legacy(payload: dict) -> dict:
    choice = {}
    message = {}
    if isinstance(payload.get("choices"), list) and payload["choices"]:
        choice = payload["choices"][0] or {}
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                message = {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "ok": True,
        "result": {
            "content": message.get("content", "") or "",
            "reasoning_content": message.get("reasoning_content", "") or "",
            "finish_reason": choice.get("finish_reason", "stop") or "stop",
            "tool_calls": message.get("tool_calls", []) or [],
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "emotive_trace": payload.get("vicuna_emotive_trace"),
        },
        "upstream": {
            "model": payload.get("model"),
            "id": payload.get("id"),
        },
    }


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _check_auth(self) -> bool:
        if not AUTH_TOKEN:
            return True
        header = self.headers.get("Authorization", "").strip()
        return header == f"Bearer {AUTH_TOKEN}"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[vicuna-runpod-mistral-relay] " + fmt % args + "\n")

    def do_GET(self) -> None:
        if self.path not in ("/health", "/v1/health"):
            _json_response(self, 404, {"error": {"message": "not found"}})
            return
        status, payload = _probe_health()
        _json_response(self, 200 if status < 400 else 503, {
            "ok": status < 400,
            "relay": "mistral_chat_proxy",
            "upstream_url": _masked_url(UPSTREAM_URL),
            "upstream_status": status,
            "upstream_health": payload,
        })

    def do_POST(self) -> None:
        if self.path not in ("/v1/inference/run", "/inference/run", "/v1/chat/completions", "/chat/completions"):
            _json_response(self, 404, {"error": {"message": "not found"}})
            return
        if not self._check_auth():
            _json_response(self, 401, {"error": {"message": "invalid runpod inference bearer token"}})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            _json_response(self, 400, {"error": {"message": f"invalid JSON: {exc}"}})
            return

        prepared = _prepare_forward_body(payload)
        status, upstream_payload = _forward_json("/v1/chat/completions", prepared)
        if self.path in ("/v1/inference/run", "/inference/run"):
            legacy_payload = _adapt_openai_to_legacy(upstream_payload) if status < 400 else upstream_payload
            _json_response(self, 200 if status < 400 else status, legacy_payload)
            return
        _json_response(self, 200 if status < 400 else status, upstream_payload)


def main() -> int:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), RelayHandler)
    sys.stderr.write(
        "[vicuna-runpod-mistral-relay] listening on "
        f"http://{LISTEN_HOST}:{LISTEN_PORT}, upstream={_masked_url(UPSTREAM_URL)}\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
