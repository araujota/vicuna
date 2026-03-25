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
import { buildProviderToolsFromRuntimeCatalog, invokeRuntimeCapability } from "../src/invoke.js";
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

const PARSED_DOCUMENT_CAPABILITY_IDS = [
  "openclaw.vicuna.parsed-documents.search-chunks",
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
  for (const capabilityId of PARSED_DOCUMENT_CAPABILITY_IDS) {
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

  const relaySchema = telegramRelay.input_schema_json as { properties: Record<string, unknown>; required?: string[]; anyOf?: unknown[] };
  assert.equal(Array.isArray(relaySchema.anyOf), true);
  assert.equal("chat_scope" in relaySchema.properties, true);
  assert.equal("request" in relaySchema.properties, true);
  assert.equal("reply_to_message_id" in relaySchema.properties, true);
  assert.equal("dedupe_key" in relaySchema.properties, true);
  assert.match(String((relaySchema.properties.request as { description?: string }).description ?? ""), /sendMessage/);
  assert.match(String(telegramRelay.method_description ?? ""), /structured Bot API send request/i);
  assert.match(String(telegramRelay.description ?? ""), /parse_mode/i);
  assert.equal(capabilities.has("openclaw.vicuna.codex_cli"), false);
  assert.equal(capabilities.has("openclaw.vicuna.ask_with_options"), false);
});

test("parsed-document search capability stays narrow and query-driven", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const parsedDocuments = capabilities.get("openclaw.vicuna.parsed-documents.search-chunks");
  assert.ok(parsedDocuments);
  assert.equal(parsedDocuments.tool_surface_id, "vicuna.documents.parsed.search_chunks");
  assert.equal(parsedDocuments.tool_name, "parsed_documents_search_chunks");

  const searchSchema = parsedDocuments.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  assert.deepEqual(searchSchema.required, ["query"]);
  assert.equal("limit" in searchSchema.properties, true);
  assert.equal("threshold" in searchSchema.properties, true);
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
  for (const capabilityId of PARSED_DOCUMENT_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  assert.equal(capabilityIds.has("openclaw.tavily.web_search"), true);
  assert.equal(capabilityIds.has("openclaw.vicuna.codex_cli"), false);
});

test("runtime catalog converts into provider tool definitions with staged metadata", () => {
  const runtimeCatalog = buildRuntimeCatalog();
  const converted = buildProviderToolsFromRuntimeCatalog(runtimeCatalog, {
    excludeToolNames: ["telegram_relay"],
  });

  assert.equal(converted.tools.some((tool) => tool.function.name === "radarr_download_movie"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "telegram_relay"), false);
  assert.equal(converted.excluded.some((entry) => entry.tool_name === "telegram_relay"), true);

  const radarrDownload = converted.tools.find((tool) => tool.function.name === "radarr_download_movie");
  assert.ok(radarrDownload);
  const functionRecord = radarrDownload.function as Record<string, unknown>;
  const parameters = functionRecord.parameters as Record<string, unknown>;
  assert.equal(functionRecord["x-vicuna-family-id"], "radarr");
  assert.equal(functionRecord["x-vicuna-family-name"], "Radarr");
  assert.equal(functionRecord["x-vicuna-method-name"], "download_movie");
  assert.equal(functionRecord["x-vicuna-method-description"], "Start Radarr movie acquisition for the requested title.");
  assert.equal(typeof parameters.description, "string");
  assert.match(String(parameters.description ?? ""), /Start Radarr movie acquisition/i);
});

test("provider tool conversion excludes capabilities with missing nested schema descriptions", () => {
  const runtimeCatalog = buildRuntimeCatalog();
  const baseCapability = runtimeCatalog.capabilities.find(
    (capability) => capability.tool_name === "radarr_download_movie",
  );
  assert.ok(baseCapability);
  const brokenCapability = {
    ...baseCapability,
    capability_id: "openclaw.test.broken-radarr",
    tool_surface_id: "vicuna.test.broken_radarr",
    tool_name: "broken_radarr_download_movie",
    input_schema_json: {
      type: "object",
      properties: {
        term: {
          type: "string",
        },
      },
      required: ["term"],
    },
  };

  const converted = buildProviderToolsFromRuntimeCatalog({
    catalog_version: runtimeCatalog.catalog_version,
    capabilities: [brokenCapability],
  });

  assert.equal(converted.tools.length, 0);
  assert.equal(converted.excluded.length, 1);
  assert.match(converted.excluded[0].reason, /missing a description/i);
});

test("invokeRuntimeCapability dispatches wrapper commands using merged fixed arguments", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "radarr_download_movie",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  const result = await invokeRuntimeCapability(capability, {
    term: "Aliens",
    monitored: true,
  }, {
    paths: {
      ...defaultPaths("/tmp/vicuna-openclaw-test"),
      radarrWrapperPath: "/tmp/vicuna-openclaw-test/bin/radarr-api",
    },
    execFileImpl: async (command, args) => {
      calls.push({ command, args });
      return {
        stdout: JSON.stringify({ ok: true, started: true }),
        stderr: "",
      };
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "/tmp/vicuna-openclaw-test/bin/radarr-api");
  const payloadArg = calls[0].args.find((entry) => entry.startsWith("--payload-base64="));
  assert.ok(payloadArg);
  const mergedArguments = JSON.parse(
    Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
  );
  assert.equal(mergedArguments.action, "download_movie");
  assert.equal(mergedArguments.term, "Aliens");
  assert.equal(mergedArguments.monitored, true);
  assert.deepEqual(result.observation, { ok: true, started: true });
});

test("OpenClaw secrets still persist media-tool config and configured runtime catalog paths", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-secrets-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const configuredCatalogPath = path.join(tempDir, "configured", "runtime-catalog.json");
  const configuredSecretsPath = path.join(tempDir, "configured", "tool-secrets.json");

  const secrets = upsertServarrConfig({}, "radarr", "radarr-key", "http://10.0.0.218:7878");
  saveToolSecrets(secretsPath, secrets);

  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.api_key, "radarr-key");
  assert.equal(loadToolSecrets(secretsPath).tools?.radarr?.base_url, "http://10.0.0.218:7878");

  process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH = configuredCatalogPath;
  process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH = configuredSecretsPath;
  const paths = defaultPaths(tempDir);
  assert.equal(paths.runtimeCatalogPath, configuredCatalogPath);
  assert.equal(paths.secretsPath, configuredSecretsPath);
  assert.equal(
    paths.ongoingTasksWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "ongoing-tasks-api")
  );
  assert.equal(
    paths.telegramRelayWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "telegram-relay-api")
  );
  assert.equal(
    paths.parsedDocumentsWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "parsed-documents-search")
  );
  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH;
  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH;
});
