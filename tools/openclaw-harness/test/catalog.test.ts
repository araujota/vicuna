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
import { writeRuntimeCatalog } from "../src/runtime-catalog.js";

const MEDIA_CAPABILITY_IDS = [
  "openclaw.servarr.radarr.list-downloaded-movies",
  "openclaw.servarr.radarr.download-movie",
  "openclaw.servarr.radarr.delete-movies",
  "openclaw.servarr.sonarr.list-downloaded-series",
  "openclaw.servarr.sonarr.download-series",
  "openclaw.servarr.sonarr.delete-series",
  "openclaw.servarr.chaptarr.list-downloaded-books",
  "openclaw.servarr.chaptarr.download-book",
  "openclaw.servarr.chaptarr.delete-books",
] as const;

const ONGOING_TASK_CAPABILITY_IDS = [
  "openclaw.vicuna.ongoing-tasks.create",
  "openclaw.vicuna.ongoing-tasks.get",
  "openclaw.vicuna.ongoing-tasks.get-due",
  "openclaw.vicuna.ongoing-tasks.edit",
  "openclaw.vicuna.ongoing-tasks.delete",
  "openclaw.vicuna.ongoing-tasks.complete",
] as const;

const TELEGRAM_CAPABILITY_IDS = [
  "openclaw.vicuna.telegram_relay",
] as const;

test("default catalog exposes only hard-memory builtins", () => {
  const catalog = buildCatalog();
  assert.deepEqual(
    catalog.capabilities.map((capability) => capability.capability_id),
    [
      "openclaw.vicuna.hard_memory_query",
      "openclaw.vicuna.hard_memory_write",
    ]
  );
});

test("runtime catalog exposes only the narrowed media surface plus optional Tavily", () => {
  const baseCatalog = buildRuntimeCatalog();
  const capabilityIds = new Set(baseCatalog.capabilities.map((capability) => capability.capability_id));

  for (const capabilityId of MEDIA_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  for (const capabilityId of ONGOING_TASK_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  for (const capabilityId of TELEGRAM_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }

  assert.equal(capabilityIds.has("openclaw.vicuna.codex_cli"), false);
  assert.equal(capabilityIds.has("openclaw.servarr.radarr.search"), false);
  assert.equal(capabilityIds.has("openclaw.servarr.sonarr.search"), false);
  assert.equal(capabilityIds.has("openclaw.servarr.chaptarr.search"), false);
  assert.equal(capabilityIds.has("openclaw.servarr.chaptarr.download-author"), false);

  const tavilyCatalog = buildRuntimeCatalog({
    secrets: {
      tools: {
        tavily: {
          api_key: "test-key"
        }
      }
    }
  });
  assert.equal(
    tavilyCatalog.capabilities.some((capability) => capability.capability_id === "openclaw.tavily.web_search"),
    true
  );
});

test("media capability schemas are tightly scoped to list, download, and delete workflows", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const radarrList = capabilities.get("openclaw.servarr.radarr.list-downloaded-movies");
  const radarrDownload = capabilities.get("openclaw.servarr.radarr.download-movie");
  const radarrDelete = capabilities.get("openclaw.servarr.radarr.delete-movies");
  const sonarrList = capabilities.get("openclaw.servarr.sonarr.list-downloaded-series");
  const sonarrDelete = capabilities.get("openclaw.servarr.sonarr.delete-series");
  const chaptarrList = capabilities.get("openclaw.servarr.chaptarr.list-downloaded-books");
  const chaptarrDownload = capabilities.get("openclaw.servarr.chaptarr.download-book");
  const chaptarrDelete = capabilities.get("openclaw.servarr.chaptarr.delete-books");

  assert.ok(radarrList && radarrDownload && radarrDelete);
  assert.ok(sonarrList && sonarrDelete);
  assert.ok(chaptarrList && chaptarrDownload && chaptarrDelete);

  assert.deepEqual(radarrList.fixed_arguments_json, { action: "list_downloaded_movies" });
  assert.deepEqual(sonarrList?.fixed_arguments_json, { action: "list_downloaded_series" });
  assert.deepEqual(chaptarrList?.fixed_arguments_json, { action: "list_downloaded_books" });
  assert.deepEqual(radarrDelete.fixed_arguments_json, { action: "delete_movies" });
  assert.deepEqual(chaptarrDelete?.fixed_arguments_json, { action: "delete_books" });

  const radarrListSchema = radarrList.input_schema_json as { properties: Record<string, unknown> };
  const radarrDownloadSchema = radarrDownload.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const radarrDeleteSchema = radarrDelete.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const chaptarrDownloadSchema = chaptarrDownload.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const chaptarrDeleteSchema = chaptarrDelete.input_schema_json as { properties: Record<string, unknown>; required?: string[] };

  assert.deepEqual(Object.keys(radarrListSchema.properties), []);
  assert.deepEqual(radarrDownloadSchema.required, ["term"]);
  assert.equal("terms" in radarrDeleteSchema.properties, true);
  assert.equal("movie_ids" in radarrDeleteSchema.properties, true);
  assert.equal("series_ids" in (sonarrDelete?.input_schema_json as { properties: Record<string, unknown> }).properties, true);
  assert.deepEqual(chaptarrDownloadSchema.required, ["term"]);
  assert.equal("book_ids" in chaptarrDeleteSchema.properties, true);
  assert.equal("terms" in chaptarrDeleteSchema.properties, true);
});

test("ongoing-task capability schemas stay narrow and action-specific", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const createTask = capabilities.get("openclaw.vicuna.ongoing-tasks.create");
  const getTask = capabilities.get("openclaw.vicuna.ongoing-tasks.get");
  const getDue = capabilities.get("openclaw.vicuna.ongoing-tasks.get-due");
  const editTask = capabilities.get("openclaw.vicuna.ongoing-tasks.edit");
  const deleteTask = capabilities.get("openclaw.vicuna.ongoing-tasks.delete");
  const completeTask = capabilities.get("openclaw.vicuna.ongoing-tasks.complete");

  assert.ok(createTask && getTask && getDue && editTask && deleteTask && completeTask);
  assert.deepEqual(createTask.fixed_arguments_json, { action: "create" });
  assert.deepEqual(getTask.fixed_arguments_json, { action: "get" });
  assert.deepEqual(getDue.fixed_arguments_json, { action: "get", due_only: true });
  assert.deepEqual(editTask.fixed_arguments_json, { action: "edit" });
  assert.deepEqual(deleteTask.fixed_arguments_json, { action: "delete" });
  assert.deepEqual(completeTask.fixed_arguments_json, { action: "complete" });

  const createSchema = createTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const getSchema = getTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const getDueSchema = getDue.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const editSchema = editTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const deleteSchema = deleteTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const completeSchema = completeTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };

  assert.deepEqual(createSchema.required, ["task_text", "interval", "unit"]);
  assert.deepEqual(Object.keys(getDueSchema.properties), []);
  assert.deepEqual(getSchema.required, undefined);
  assert.deepEqual(editSchema.required, ["task_id"]);
  assert.deepEqual(deleteSchema.required, ["task_id"]);
  assert.deepEqual(completeSchema.required, ["task_id"]);
  assert.equal("active" in editSchema.properties, true);
  assert.equal("completed_at" in completeSchema.properties, true);
});

test("telegram relay capability stays narrow and does not expose the deleted ask/codex surface", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const telegramRelay = capabilities.get("openclaw.vicuna.telegram_relay");
  assert.ok(telegramRelay);
  assert.equal(telegramRelay.tool_surface_id, "vicuna.telegram.relay");
  assert.equal(telegramRelay.tool_name, "telegram_relay");

  const relaySchema = telegramRelay.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  assert.deepEqual(relaySchema.required, ["text"]);
  assert.equal("chat_scope" in relaySchema.properties, true);
  assert.equal("reply_to_message_id" in relaySchema.properties, true);
  assert.equal("dedupe_key" in relaySchema.properties, true);
  assert.equal(capabilities.has("openclaw.vicuna.codex_cli"), false);
  assert.equal(capabilities.has("openclaw.vicuna.ask_with_options"), false);
});

test("runtime catalog writing preserves the narrowed tool surface", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-runtime-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const runtimeCatalogPath = path.join(tempDir, "openclaw-catalog.json");

  saveToolSecrets(secretsPath, upsertTavilyApiKey({}, "test-key"));
  writeRuntimeCatalog(runtimeCatalogPath, secretsPath);

  const runtimeCatalog = JSON.parse(fs.readFileSync(runtimeCatalogPath, "utf8")) as {
    capabilities: Array<{ capability_id: string }>;
  };
  const capabilityIds = new Set(runtimeCatalog.capabilities.map((capability) => capability.capability_id));

  for (const capabilityId of MEDIA_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  for (const capabilityId of ONGOING_TASK_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  for (const capabilityId of TELEGRAM_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  assert.equal(capabilityIds.has("openclaw.tavily.web_search"), true);
  assert.equal(capabilityIds.has("openclaw.vicuna.codex_cli"), false);
});

test("OpenClaw secrets still persist media-tool config and configured runtime catalog paths", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-secrets-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const configuredCatalogPath = path.join(tempDir, "configured", "runtime-catalog.json");

  const secrets = upsertServarrConfig({}, "radarr", "radarr-key", "http://10.0.0.218:7878");
  saveToolSecrets(secretsPath, secrets);

  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.api_key, "radarr-key");
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.base_url, "http://10.0.0.218:7878");

  process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH = configuredCatalogPath;
  const paths = defaultPaths(tempDir);
  assert.equal(paths.runtimeCatalogPath, configuredCatalogPath);
  assert.equal(
    paths.ongoingTasksWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "ongoing-tasks-api")
  );
  assert.equal(
    paths.telegramRelayWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "telegram-relay-api")
  );
  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH;
});
