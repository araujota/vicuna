import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { buildCatalog, buildRuntimeCatalog } from "../src/catalog.js";
import {
  defaultPaths,
  loadToolSecrets,
  saveToolSecrets,
  upsertServarrConfig,
  upsertTavilyApiKey
} from "../src/config.js";
import { assertCapabilityDescriptor } from "../src/contracts.js";
import { resolveInvocation } from "../src/invoke.js";
import { writeRuntimeCatalog } from "../src/runtime-catalog.js";

const COG_TOOL_FLAG_ACTIVE_ELIGIBLE = 1 << 0;
const COG_TOOL_FLAG_DMN_ELIGIBLE = 1 << 1;
const COG_TOOL_FLAG_SIMULATION_SAFE = 1 << 2;
const COG_TOOL_FLAG_REMEDIATION_SAFE = 1 << 3;
const COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT = 1 << 4;

test("default catalog includes exec and hard-memory", () => {
  const catalog = buildCatalog();
  assert.equal(catalog.capabilities.length, 4);
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
      },
      {
        tool_surface_id: "vicuna.memory.hard_write",
        capability_id: "openclaw.vicuna.hard_memory_write"
      },
      {
        tool_surface_id: "vicuna.codex.main",
        capability_id: "openclaw.vicuna.codex_cli"
      }
    ]
  );
});

test("exec capability description and parameters describe host-local observation", () => {
  const catalog = buildCatalog();
  const execCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.exec.command"
  );

  assert.ok(execCapability);
  assert.match(execCapability.description, /host-local state/i);
  assert.match(execCapability.description, /current working directory/i);
  assert.match(execCapability.description, /repository state/i);

  const schema = execCapability.input_schema_json as {
    properties: {
      command: { description?: string };
      workdir: { description?: string };
    };
  };

  assert.match(schema.properties.command.description ?? "", /pwd/i);
  assert.match(schema.properties.workdir.description ?? "", /working directory/i);
});

test("capability validation rejects missing parameter descriptions", () => {
  assert.throws(() =>
    assertCapabilityDescriptor({
      capability_id: "openclaw.exec.command",
      tool_surface_id: "vicuna.exec.main",
      capability_kind: "tool",
      owner_plugin_id: "openclaw-core",
      tool_name: "exec",
      description: "Run a command",
      input_schema_json: {
        type: "object",
        properties: {
          command: { type: "string" }
        }
      },
      output_contract: "pending_then_result",
      side_effect_class: "system_exec",
      approval_mode: "policy_driven",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/openclaw-core/tool/exec",
      tool_kind: 4,
      tool_flags: COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      latency_class: 1,
      max_steps_reserved: 1,
      dispatch_backend: "legacy_bash"
    }),
  /input_schema_json\.properties\.command/
  );
});

test("catalog capabilities declare explicit cognitive eligibility flags", () => {
  const catalog = buildCatalog();
  const flagsByCapability = new Map(
    catalog.capabilities.map((capability) => [capability.capability_id, capability.tool_flags])
  );

  assert.equal(
    flagsByCapability.get("openclaw.exec.command"),
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE |
      COG_TOOL_FLAG_DMN_ELIGIBLE |
      COG_TOOL_FLAG_REMEDIATION_SAFE |
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
  );
  assert.equal(
    flagsByCapability.get("openclaw.vicuna.hard_memory_query"),
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE |
      COG_TOOL_FLAG_DMN_ELIGIBLE |
      COG_TOOL_FLAG_SIMULATION_SAFE |
      COG_TOOL_FLAG_REMEDIATION_SAFE
  );
  assert.equal(
    flagsByCapability.get("openclaw.vicuna.hard_memory_write"),
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE |
      COG_TOOL_FLAG_DMN_ELIGIBLE |
      COG_TOOL_FLAG_REMEDIATION_SAFE |
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
  );
  assert.equal(
    flagsByCapability.get("openclaw.vicuna.codex_cli"),
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE |
      COG_TOOL_FLAG_DMN_ELIGIBLE |
      COG_TOOL_FLAG_REMEDIATION_SAFE |
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
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

test("runtime catalog always includes Radarr and Sonarr and adds Tavily when configured", () => {
  assert.deepEqual(
    buildRuntimeCatalog().capabilities.map((capability) => capability.capability_id),
    ["openclaw.servarr.radarr", "openclaw.servarr.sonarr"]
  );
  const catalog = buildRuntimeCatalog({
    secrets: {
      tools: {
        tavily: {
          api_key: "test-key"
        }
      }
    }
  });
  assert.deepEqual(
    catalog.capabilities.map((capability) => capability.capability_id),
    [
      "openclaw.servarr.radarr",
      "openclaw.servarr.sonarr",
      "openclaw.tavily.web_search"
    ]
  );
  const webSearchCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.tavily.web_search"
  );
  assert.ok(webSearchCapability);
  assert.equal(
    webSearchCapability?.tool_flags,
    COG_TOOL_FLAG_ACTIVE_ELIGIBLE |
      COG_TOOL_FLAG_DMN_ELIGIBLE |
      COG_TOOL_FLAG_REMEDIATION_SAFE |
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
  );
  assert.deepEqual(
    Object.keys((webSearchCapability?.input_schema_json as { properties: Record<string, unknown> }).properties).sort(),
    [
      "country",
      "exclude_domains",
      "include_domains",
      "max_results",
      "query",
      "search_depth",
      "time_range",
      "topic",
    ]
  );
  const webSearchSchema = webSearchCapability?.input_schema_json as {
    properties: Record<string, { description?: string }>;
  };
  for (const value of Object.values(webSearchSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
});

test("runtime catalog exposes fully described Radarr and Sonarr schemas", () => {
  const catalog = buildRuntimeCatalog();
  const radarrCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.radarr"
  );
  const sonarrCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.sonarr"
  );

  assert.ok(radarrCapability);
  assert.ok(sonarrCapability);

  const radarrSchema = radarrCapability?.input_schema_json as {
    properties: Record<string, { description?: string }>;
  };
  const sonarrSchema = sonarrCapability?.input_schema_json as {
    properties: Record<string, { description?: string }>;
  };

  assert.match(radarrCapability?.description ?? "", /Radarr movie library/i);
  assert.match(sonarrCapability?.description ?? "", /Sonarr series library/i);
  assert.ok("action" in radarrSchema.properties);
  assert.ok("action" in sonarrSchema.properties);

  for (const value of Object.values(radarrSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(sonarrSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
});

test("OpenClaw secrets persist Tavily config and emit a runtime catalog", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-harness-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const runtimeCatalogPath = path.join(tempDir, "openclaw-catalog.json");

  const secrets = upsertTavilyApiKey({}, "test-key");
  saveToolSecrets(secretsPath, secrets);
  assert.equal(loadToolSecrets(secretsPath).tools?.tavily?.api_key, "test-key");

  writeRuntimeCatalog(runtimeCatalogPath, secretsPath);
  const runtimeCatalog = JSON.parse(fs.readFileSync(runtimeCatalogPath, "utf8"));
  assert.deepEqual(
    runtimeCatalog.capabilities.map((capability: { capability_id: string }) => capability.capability_id),
    [
      "openclaw.servarr.radarr",
      "openclaw.servarr.sonarr",
      "openclaw.tavily.web_search"
    ]
  );
});

test("OpenClaw secrets persist Servarr config while keeping the tools visible", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-servarr-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");

  const secrets = upsertServarrConfig({}, "radarr", "radarr-key", "http://10.0.0.218:7878");
  saveToolSecrets(secretsPath, secrets);
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.api_key, "radarr-key");
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.base_url, "http://10.0.0.218:7878");
});

test("runtime catalog path honors the configured fabric catalog path", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-harness-env-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const configuredCatalogPath = path.join(tempDir, "configured", "runtime-catalog.json");

  saveToolSecrets(secretsPath, upsertTavilyApiKey({}, "test-key"));

  process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH = configuredCatalogPath;
  const paths = defaultPaths(tempDir);
  assert.equal(paths.runtimeCatalogPath, configuredCatalogPath);

  writeRuntimeCatalog(configuredCatalogPath, secretsPath);
  assert.equal(fs.existsSync(configuredCatalogPath), true);

  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH;
});
