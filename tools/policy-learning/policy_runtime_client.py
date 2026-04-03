#!/usr/bin/env python3

from __future__ import annotations

import json
from urllib import parse, request


class PolicyRuntimeClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict | None = None,
    ) -> dict:
        target_url = f"{self.base_url}{path}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(target_url, data=body, method=method, headers=headers)
        with request.urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_status(self) -> dict:
        return self._request_json("/v1/policy/status")

    def get_transitions(self, *, limit: int = 512, request_id: str | None = None) -> dict:
        params = {"limit": str(limit)}
        if request_id:
            params["request_id"] = request_id
        query = parse.urlencode(params)
        return self._request_json(f"/v1/policy/transitions?{query}")

    def get_decode_traces(self, *, limit: int = 128, request_id: str | None = None) -> dict:
        params = {"limit": str(limit)}
        if request_id:
            params["request_id"] = request_id
        query = parse.urlencode(params)
        return self._request_json(f"/v1/policy/decode-traces?{query}")

    def get_runtime_artifacts(self) -> dict:
        return self._request_json("/v1/policy/runtime-artifacts")

    def apply_runtime_artifact(self, payload: dict) -> dict:
        return self._request_json(
            "/v1/policy/runtime-artifacts",
            method="POST",
            payload=payload,
        )

    def propose_candidate(self, candidate_url: str, payload: dict) -> dict:
        base = candidate_url.rstrip("/")
        if not base.endswith("/v1/policy/propose"):
            base = f"{base}/v1/policy/propose"
        req = request.Request(
            base,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
