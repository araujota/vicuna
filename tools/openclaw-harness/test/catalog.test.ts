import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { buildCatalog, buildRuntimeCatalog } from "../src/catalog.js";
import { FIXED_CHAPTARR_ROOT_FOLDER_PATH } from "../src/chaptarr-client.js";
import {
  defaultPaths,
  loadToolSecrets,
  saveToolSecrets,
  upsertApiToolConfig,
  upsertServarrConfig,
  upsertTavilyApiKey
} from "../src/config.js";
import { assertCapabilityDescriptor } from "../src/contracts.js";
import { resolveInvocation } from "../src/invoke.js";
import { writeRuntimeCatalog } from "../src/runtime-catalog.js";
import { FIXED_RADARR_ROOT_FOLDER_PATH, FIXED_SONARR_ROOT_FOLDER_PATH } from "../src/servarr.js";

const COG_TOOL_FLAG_ACTIVE_ELIGIBLE = 1 << 0;
const COG_TOOL_FLAG_DMN_ELIGIBLE = 1 << 1;
const COG_TOOL_FLAG_SIMULATION_SAFE = 1 << 2;
const COG_TOOL_FLAG_REMEDIATION_SAFE = 1 << 3;
const COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT = 1 << 4;

test("default catalog includes hard-memory and codex builtins", () => {
  const catalog = buildCatalog();
  assert.equal(catalog.capabilities.length, 3);
  assert.deepEqual(
    catalog.capabilities.map((capability) => ({
      tool_surface_id: capability.tool_surface_id,
      capability_id: capability.capability_id
    })),
    [
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

test("default catalog does not expose the raw exec capability", () => {
  const catalog = buildCatalog();
  const removedExecCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.exec.command"
  );
  assert.equal(removedExecCapability, undefined);
});

test("capability validation rejects missing parameter descriptions", () => {
  assert.throws(() =>
    assertCapabilityDescriptor({
      capability_id: "openclaw.test.query",
      tool_surface_id: "vicuna.test.query",
      capability_kind: "memory_adapter",
      owner_plugin_id: "openclaw-test",
      tool_name: "memory_query_test",
      description: "Query synthetic memory",
      input_schema_json: {
        type: "object",
        properties: {
          query: { type: "string" }
        }
      },
      output_contract: "completed_result",
      side_effect_class: "memory_read",
      execution_safety_class: "read_only",
      approval_mode: "none",
      execution_modes: ["sync"],
      provenance_namespace: "openclaw/test/memory_query",
      tool_kind: 2,
      tool_flags: COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      latency_class: 1,
      max_steps_reserved: 1,
      dispatch_backend: "legacy_hard_memory"
    }),
  /input_schema_json\.properties\.query/
  );
});

test("catalog capabilities declare explicit cognitive eligibility flags", () => {
  const catalog = buildCatalog();
  const flagsByCapability = new Map(
    catalog.capabilities.map((capability) => [capability.capability_id, capability.tool_flags])
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
      tool_surface_id: "vicuna.memory.hard_query",
      capability_id: "openclaw.vicuna.hard_memory_query.fake",
      vicuna_session_id: "sess_1",
      vicuna_run_id: "run_1",
      origin_phase: "active",
      arguments_json: { query: "recent preferences" },
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
      capability_id: "openclaw.vicuna.hard_memory_write",
      vicuna_session_id: "sess_1",
      vicuna_run_id: "run_1",
      origin_phase: "dmn",
      arguments_json: { query: "recent preferences" },
      requested_mode: "sync",
      deadline_ms: 1000,
      provenance_request_id: "prov_2"
    })
  );
});

test("runtime catalog always includes Radarr, Sonarr, and Chaptarr and adds Tavily when configured", () => {
  const baseCapabilityIds = buildRuntimeCatalog().capabilities.map((capability) => capability.capability_id);
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.radarr.inspect"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.sonarr.inspect"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.chaptarr.inspect"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.radarr.download-movie"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.radarr.delete-movie"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.sonarr.download-series"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.sonarr.delete-series"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.chaptarr.download-author"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.chaptarr.download-book"));
  assert.ok(baseCapabilityIds.includes("openclaw.servarr.chaptarr.delete-book"));
  const catalog = buildRuntimeCatalog({
    secrets: {
      tools: {
        tavily: {
          api_key: "test-key"
        }
      }
    }
  });
  assert.ok(catalog.capabilities.some((capability) => capability.capability_id === "openclaw.tavily.web_search"));
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

test("runtime catalog exposes fully described Radarr, Sonarr, and Chaptarr schemas", () => {
  const catalog = buildRuntimeCatalog();
  const radarrCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.radarr.download-movie"
  );
  const sonarrCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.sonarr.download-series"
  );
  const chaptarrCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.chaptarr.download-author"
  );
  const chaptarrAddBookCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.chaptarr.download-book"
  );
  const chaptarrSearchCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.chaptarr.search"
  );
  const chaptarrBookLookupCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.chaptarr.book-lookup"
  );
  const radarrDeleteCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.radarr.delete-movie"
  );
  const sonarrDeleteCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.sonarr.delete-series"
  );
  const chaptarrDeleteCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.chaptarr.delete-book"
  );
  const radarrInspectCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.radarr.inspect"
  );
  const radarrSearchCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.radarr.search"
  );
  const sonarrSearchCapability = catalog.capabilities.find(
    (capability) => capability.capability_id === "openclaw.servarr.sonarr.search"
  );

  assert.ok(radarrCapability);
  assert.ok(sonarrCapability);
  assert.ok(chaptarrCapability);
  assert.ok(chaptarrAddBookCapability);
  assert.ok(chaptarrSearchCapability);
  assert.ok(chaptarrBookLookupCapability);
  assert.ok(radarrDeleteCapability);
  assert.ok(sonarrDeleteCapability);
  assert.ok(chaptarrDeleteCapability);
  assert.ok(radarrInspectCapability);
  assert.ok(radarrSearchCapability);
  assert.ok(sonarrSearchCapability);

  const radarrSchema = radarrCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const sonarrSchema = sonarrCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const chaptarrSchema = chaptarrCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const chaptarrSearchSchema = chaptarrSearchCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
  };
  const radarrDeleteSchema = radarrDeleteCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const sonarrDeleteSchema = sonarrDeleteCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const chaptarrDeleteSchema = chaptarrDeleteCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const chaptarrAddBookSchema = chaptarrAddBookCapability?.input_schema_json as {
    properties: Record<string, { description?: string; enum?: string[] }>;
    required?: string[];
  };
  const radarrInspectSchema = radarrInspectCapability?.input_schema_json as {
    properties: Record<string, { description?: string }>;
  };

  assert.match(radarrCapability?.description ?? "", /Start Radarr movie acquisition/i);
  assert.match(sonarrCapability?.description ?? "", /Start Sonarr series acquisition/i);
  assert.match(chaptarrCapability?.description ?? "", /Start ebook-only Chaptarr acquisition for an author's catalog/i);
  assert.match(chaptarrAddBookCapability?.description ?? "", /Start ebook-only Chaptarr acquisition for a specific title/i);
  assert.match(chaptarrSearchCapability?.description ?? "", /ebook-capable matches/i);
  assert.match(chaptarrBookLookupCapability?.description ?? "", /ebook-capable matches/i);
  assert.match(radarrDeleteCapability?.description ?? "", /remove its files from `\/movies` by default/i);
  assert.match(sonarrDeleteCapability?.description ?? "", /remove its files from `\/tv` by default/i);
  assert.match(chaptarrDeleteCapability?.description ?? "", /remove its files from `\/books` by default/i);
  assert.match(radarrInspectCapability?.description ?? "", /Inspect the current Radarr movie library/i);
  assert.ok(!("action" in radarrSchema.properties));
  assert.ok(!("action" in sonarrSchema.properties));
  assert.ok(!("action" in chaptarrSchema.properties));
  assert.ok(!("action" in radarrDeleteSchema.properties));
  assert.ok(!("action" in sonarrDeleteSchema.properties));
  assert.ok(!("action" in chaptarrDeleteSchema.properties));
  assert.ok(!("action" in chaptarrAddBookSchema.properties));
  assert.ok(!("action" in radarrInspectSchema.properties));
  assert.deepEqual(radarrCapability?.fixed_arguments_json, { action: "download_movie" });
  assert.deepEqual(sonarrCapability?.fixed_arguments_json, { action: "download_series" });
  assert.deepEqual(chaptarrCapability?.fixed_arguments_json, { action: "download_author" });
  assert.deepEqual(chaptarrAddBookCapability?.fixed_arguments_json, { action: "download_book" });
  assert.deepEqual(radarrDeleteCapability?.fixed_arguments_json, { action: "delete_movie" });
  assert.deepEqual(sonarrDeleteCapability?.fixed_arguments_json, { action: "delete_series" });
  assert.deepEqual(chaptarrDeleteCapability?.fixed_arguments_json, { action: "delete_book" });
  assert.deepEqual(chaptarrSearchCapability?.fixed_arguments_json, { action: "search" });
  assert.deepEqual(radarrInspectCapability?.fixed_arguments_json, { action: "inspect" });
  assert.deepEqual(radarrSchema.required, ["term"]);
  assert.deepEqual(sonarrSchema.required, ["term"]);
  assert.deepEqual(chaptarrSchema.required, ["term"]);
  assert.deepEqual(chaptarrAddBookSchema.required, ["term"]);
  assert.equal(radarrDeleteSchema.required, undefined);
  assert.equal(sonarrDeleteSchema.required, undefined);
  assert.equal(chaptarrDeleteSchema.required, undefined);
  assert.ok(radarrSchema.properties.minimum_availability.enum?.includes("deleted"));
  assert.ok(!("quality_profile_id" in radarrSchema.properties));
  assert.ok(!("quality_profile_id" in sonarrSchema.properties));
  assert.ok(!("quality_profile_id" in chaptarrSchema.properties));
  assert.ok(!("quality_profile_id" in chaptarrAddBookSchema.properties));
  assert.ok(!("metadata_profile_id" in chaptarrSchema.properties));
  assert.ok(!("metadata_profile_id" in chaptarrAddBookSchema.properties));
  assert.ok(!("monitored" in chaptarrSchema.properties));
  assert.ok(!("monitored" in chaptarrAddBookSchema.properties));
  assert.ok(!("search_for_missing_books" in chaptarrSchema.properties));
  assert.ok(!("search_for_new_book" in chaptarrAddBookSchema.properties));
  assert.ok(!("root_folder_path" in radarrSchema.properties));
  assert.ok(!("root_folder_path" in sonarrSchema.properties));
  assert.ok(!("media_type" in chaptarrSchema.properties));
  assert.ok(!("media_type" in chaptarrAddBookSchema.properties));
  assert.ok(!("root_folder_path" in chaptarrSchema.properties));
  assert.ok(!("root_folder_path" in chaptarrAddBookSchema.properties));
  assert.match(chaptarrSearchSchema.properties.provider.description ?? "", /Hardcover/i);
  assert.match(chaptarrAddBookCapability?.description ?? "", /same tool call/i);
  assert.match(chaptarrAddBookCapability?.description ?? "", /starts acquisition\/search and does not guarantee a completed download\/import/i);
  assert.match(radarrSearchCapability?.description ?? "", /does not add the movie or start a download/i);
  assert.match(sonarrSearchCapability?.description ?? "", /does not add the series or start a download/i);
  assert.match(radarrDeleteSchema.properties.delete_files.description ?? "", /remove the movie files from disk/i);
  assert.match(sonarrDeleteSchema.properties.delete_files.description ?? "", /remove the series files from disk/i);
  assert.match(chaptarrDeleteSchema.properties.delete_files.description ?? "", /remove the ebook files from disk/i);
  assert.match(radarrDeleteSchema.properties.add_import_exclusion.description ?? "", /import exclusion/i);
  assert.match(sonarrDeleteSchema.properties.add_import_list_exclusion.description ?? "", /import-list exclusion/i);
  assert.match(chaptarrDeleteSchema.properties.add_import_list_exclusion.description ?? "", /import-list exclusion/i);
  assert.equal(radarrSearchCapability?.side_effect_class, "service_read");
  assert.equal(sonarrSearchCapability?.side_effect_class, "service_read");
  assert.equal(chaptarrSearchCapability?.side_effect_class, "service_read");
  assert.equal(radarrSearchCapability?.execution_safety_class, "read_only");
  assert.equal(sonarrSearchCapability?.execution_safety_class, "read_only");
  assert.equal(chaptarrSearchCapability?.execution_safety_class, "read_only");
  assert.equal(radarrCapability?.side_effect_class, "service_acquisition");
  assert.equal(sonarrCapability?.side_effect_class, "service_acquisition");
  assert.equal(chaptarrAddBookCapability?.side_effect_class, "service_acquisition");
  assert.equal(radarrCapability?.execution_safety_class, "approval_required");
  assert.equal(sonarrCapability?.execution_safety_class, "approval_required");
  assert.equal(chaptarrAddBookCapability?.execution_safety_class, "approval_required");
  assert.equal(radarrDeleteCapability?.side_effect_class, "service_api");
  assert.equal(sonarrDeleteCapability?.side_effect_class, "service_api");
  assert.equal(chaptarrDeleteCapability?.side_effect_class, "service_api");
  assert.equal(radarrDeleteCapability?.execution_safety_class, "approval_required");
  assert.equal(sonarrDeleteCapability?.execution_safety_class, "approval_required");
  assert.equal(chaptarrDeleteCapability?.execution_safety_class, "approval_required");
  assert.match(radarrCapability?.description ?? "", new RegExp(FIXED_RADARR_ROOT_FOLDER_PATH.replace("/", "\\/")));
  assert.match(sonarrCapability?.description ?? "", new RegExp(FIXED_SONARR_ROOT_FOLDER_PATH.replace("/", "\\/")));
  assert.match(chaptarrCapability?.description ?? "", new RegExp(FIXED_CHAPTARR_ROOT_FOLDER_PATH.replace("/", "\\/")));
  assert.match(chaptarrAddBookCapability?.description ?? "", new RegExp(FIXED_CHAPTARR_ROOT_FOLDER_PATH.replace("/", "\\/")));

  for (const value of Object.values(radarrSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(sonarrSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(chaptarrSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(radarrDeleteSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(sonarrDeleteSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(chaptarrDeleteSchema.properties)) {
    assert.ok(typeof value.description === "string" && value.description.length > 0);
  }
  for (const value of Object.values(chaptarrAddBookSchema.properties)) {
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
  const capabilityIds = runtimeCatalog.capabilities.map((capability: { capability_id: string }) => capability.capability_id);
  assert.ok(capabilityIds.includes("openclaw.servarr.radarr.inspect"));
  assert.ok(capabilityIds.includes("openclaw.servarr.sonarr.inspect"));
  assert.ok(capabilityIds.includes("openclaw.servarr.chaptarr.inspect"));
  assert.ok(capabilityIds.includes("openclaw.tavily.web_search"));
});

test("OpenClaw secrets persist media-tool config while keeping the tools visible", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-servarr-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");

  const secrets = upsertServarrConfig({}, "radarr", "radarr-key", "http://10.0.0.218:7878");
  saveToolSecrets(secretsPath, secrets);
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.api_key, "radarr-key");
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.base_url, "http://10.0.0.218:7878");

  const withChaptarr = upsertApiToolConfig(loadToolSecrets(secretsPath), "chaptarr", "chaptarr-key", "http://10.0.0.218:8789");
  saveToolSecrets(secretsPath, withChaptarr);
  assert.equal(loadToolSecrets(secretsPath).tools?.chaptarr?.api_key, "chaptarr-key");
  assert.equal(loadToolSecrets(secretsPath).tools?.chaptarr?.base_url, "http://10.0.0.218:8789");
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
