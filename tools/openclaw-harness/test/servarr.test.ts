import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_RADARR_BASE_URL,
  DEFAULT_SONARR_BASE_URL,
} from "../src/config.js";
import {
  errorEnvelope,
  parseCliInvocation,
  resolveServarrConfig,
  servarrRequestJson,
  ServarrToolError,
} from "../src/servarr.js";

test("resolveServarrConfig falls back to LAN NAS defaults", () => {
  const originalFetch = globalThis.fetch;
  assert.equal(resolveServarrConfig("radarr", "/tmp/does-not-exist.json").baseUrl, DEFAULT_RADARR_BASE_URL);
  assert.equal(resolveServarrConfig("sonarr", "/tmp/does-not-exist.json").baseUrl, DEFAULT_SONARR_BASE_URL);
  globalThis.fetch = originalFetch;
});

test("parseCliInvocation decodes the base64 tool payload", () => {
  const encoded = Buffer.from(JSON.stringify({ action: "lookup_movie", term: "Alien" }), "utf8").toString("base64");
  const parsed = parseCliInvocation([`--payload-base64=${encoded}`, "--secrets-path=/tmp/secrets.json"]);
  assert.equal(parsed.secretsPath, "/tmp/secrets.json");
  assert.equal(parsed.payload.action, "lookup_movie");
  assert.equal(parsed.payload.term, "Alien");
});

test("servarrRequestJson returns a typed missing-api-key error before fetch", async () => {
  await assert.rejects(
    servarrRequestJson(
      {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: undefined
      },
      "GET",
      "/api/v3/system/status"
    ),
    (error: unknown) => error instanceof ServarrToolError && error.kind === "missing_api_key"
  );
});

test("errorEnvelope preserves typed Servarr errors", () => {
  const envelope = errorEnvelope(
    "sonarr",
    "add_series",
    DEFAULT_SONARR_BASE_URL,
    new ServarrToolError("authorization_failed", "Sonarr request failed", { status: 401 })
  );

  assert.equal(envelope.ok, false);
  assert.equal(envelope.error?.kind, "authorization_failed");
  assert.equal(envelope.error?.details?.status, 401);
});
