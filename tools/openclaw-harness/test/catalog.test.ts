import test from "node:test";
import assert from "node:assert/strict";

import { buildCatalog } from "../src/catalog.js";
import { resolveInvocation } from "../src/invoke.js";

test("default catalog includes exec and hard-memory", () => {
  const catalog = buildCatalog();
  assert.equal(catalog.capabilities.length, 2);
  assert.deepEqual(
    catalog.capabilities.map((capability) => ({
      tool_surface_id: capability.tool_surface_id,
      capability_id: capability.capability_id
    })),
    [
      {
        tool_surface_id: "vicuna.exec.main",
        capability_id: "openclaw.exec.command"
      },
      {
        tool_surface_id: "vicuna.memory.hard_query",
        capability_id: "openclaw.vicuna.hard_memory_query"
      }
    ]
  );
});

test("unknown capability ids are rejected", () => {
  const catalog = buildCatalog();
  assert.throws(() =>
    resolveInvocation(catalog, {
      invocation_id: "ocinv_1",
      tool_surface_id: "vicuna.exec.main",
      capability_id: "openclaw.exec.fake",
      vicuna_session_id: "sess_1",
      vicuna_run_id: "run_1",
      origin_phase: "active",
      arguments_json: { command: "pwd" },
      requested_mode: "sync",
      deadline_ms: 1000,
      provenance_request_id: "prov_1"
    })
  );
});

test("tool surface and capability id must match exactly", () => {
  const catalog = buildCatalog();
  assert.throws(() =>
    resolveInvocation(catalog, {
      invocation_id: "ocinv_2",
      tool_surface_id: "vicuna.memory.hard_query",
      capability_id: "openclaw.exec.command",
      vicuna_session_id: "sess_1",
      vicuna_run_id: "run_1",
      origin_phase: "dmn",
      arguments_json: { command: "pwd" },
      requested_mode: "sync",
      deadline_ms: 1000,
      provenance_request_id: "prov_2"
    })
  );
});
