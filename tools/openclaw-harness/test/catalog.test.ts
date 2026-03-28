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

const RUNTIME_CAPABILITY_IDS = [
  "openclaw.vicuna.media.read",
  "openclaw.vicuna.media.download",
  "openclaw.vicuna.media.delete",
  "openclaw.vicuna.hard_memory_read",
  "openclaw.vicuna.hard_memory_write_flattened",
  "openclaw.vicuna.skill_read",
  "openclaw.vicuna.skill_create",
  "openclaw.vicuna.host_shell",
  "openclaw.vicuna.ongoing-task.create",
  "openclaw.vicuna.ongoing-task.delete",
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

test("runtime catalog exposes the flattened runtime surface plus optional Tavily", () => {
  const baseCatalog = buildRuntimeCatalog();
  const capabilityIds = new Set(baseCatalog.capabilities.map((capability) => capability.capability_id));

  for (const capabilityId of RUNTIME_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }

  assert.equal(capabilityIds.has("openclaw.vicuna.telegram_relay"), false);
  assert.equal(capabilityIds.has("openclaw.vicuna.parsed-documents.search-chunks"), false);
  assert.equal(capabilityIds.has("openclaw.vicuna.ongoing-tasks.get"), false);
  assert.equal(capabilityIds.has("openclaw.servarr.radarr.download-movie"), false);
  assert.equal(capabilityIds.has("openclaw.vicuna.codex_cli"), false);

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

test("flattened media capability schemas stay narrow and explicit", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const mediaRead = capabilities.get("openclaw.vicuna.media.read");
  const mediaDownload = capabilities.get("openclaw.vicuna.media.download");
  const mediaDelete = capabilities.get("openclaw.vicuna.media.delete");

  assert.ok(mediaRead && mediaDownload && mediaDelete);

  const readSchema = mediaRead.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const downloadSchema = mediaDownload.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const deleteSchema = mediaDelete.input_schema_json as { properties: Record<string, unknown>; required?: string[] };

  assert.deepEqual(readSchema.required, ["media_kind", "query"]);
  assert.deepEqual(downloadSchema.required, ["media_kind", "query"]);
  assert.deepEqual(deleteSchema.required, ["media_kind", "query"]);
  assert.equal("backend_hint" in readSchema.properties, true);
  assert.equal("status_filter" in readSchema.properties, true);
  assert.equal("limit" in readSchema.properties, true);
  assert.equal("id_hint" in downloadSchema.properties, true);
  assert.equal("delete_files" in deleteSchema.properties, true);
});

test("flattened ongoing-task capability schemas only expose create and delete", () => {
  const catalog = buildRuntimeCatalog();
  const capabilities = new Map(catalog.capabilities.map((capability) => [capability.capability_id, capability]));

  const createTask = capabilities.get("openclaw.vicuna.ongoing-task.create");
  const deleteTask = capabilities.get("openclaw.vicuna.ongoing-task.delete");

  assert.ok(createTask && deleteTask);
  assert.equal(capabilities.has("openclaw.vicuna.ongoing-tasks.get"), false);
  assert.equal(capabilities.has("openclaw.vicuna.ongoing-tasks.complete"), false);

  const createSchema = createTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };
  const deleteSchema = deleteTask.input_schema_json as { properties: Record<string, unknown>; required?: string[] };

  assert.deepEqual(createSchema.required, ["task_text", "interval", "unit"]);
  assert.deepEqual(deleteSchema.required, ["task_id"]);
  assert.deepEqual((createSchema.properties.unit as { enum?: unknown[] }).enum, ["minute", "hour", "day", "week"]);
});

test("runtime catalog writing preserves the flattened runtime surface", () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-openclaw-runtime-"));
  const secretsPath = path.join(tempDir, "openclaw-tool-secrets.json");
  const runtimeCatalogPath = path.join(tempDir, "openclaw-catalog.json");

  saveToolSecrets(secretsPath, upsertTavilyApiKey({}, "test-key"));
  writeRuntimeCatalog(runtimeCatalogPath, secretsPath);

  const runtimeCatalog = JSON.parse(fs.readFileSync(runtimeCatalogPath, "utf8")) as {
    capabilities: Array<{ capability_id: string }>;
  };
  const capabilityIds = new Set(runtimeCatalog.capabilities.map((capability) => capability.capability_id));

  for (const capabilityId of RUNTIME_CAPABILITY_IDS) {
    assert.equal(capabilityIds.has(capabilityId), true, `${capabilityId} should be present`);
  }
  assert.equal(capabilityIds.has("openclaw.tavily.web_search"), true);
  assert.equal(capabilityIds.has("openclaw.vicuna.telegram_relay"), false);
});

test("runtime catalog converts flattened capabilities into provider tools with preserved metadata", () => {
  const runtimeCatalog = buildRuntimeCatalog();
  const converted = buildProviderToolsFromRuntimeCatalog(runtimeCatalog);

  assert.equal(converted.excluded.length, 0);
  assert.equal(converted.tools.some((tool) => tool.function.name === "media_download"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "host_shell"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "skill_read"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "skill_create"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "ongoing_task_create"), true);
  assert.equal(converted.tools.some((tool) => tool.function.name === "telegram_relay"), false);

  const mediaDownload = converted.tools.find((tool) => tool.function.name === "media_download");
  assert.ok(mediaDownload);
  const functionRecord = mediaDownload.function as Record<string, unknown>;
  const parameters = functionRecord.parameters as Record<string, unknown>;
  assert.equal(functionRecord["x-vicuna-family-id"], "media");
  assert.equal(functionRecord["x-vicuna-family-name"], "Media");
  assert.equal(functionRecord["x-vicuna-method-name"], "download");
  assert.equal(functionRecord["x-vicuna-method-description"], "Start movie, series, or book acquisition through one direct tool call.");
  assert.equal(typeof parameters.description, "string");
  assert.match(String(parameters.description ?? ""), /direct tool call/i);

  const hostShell = converted.tools.find((tool) => tool.function.name === "host_shell");
  assert.ok(hostShell);
  const hostShellFunction = hostShell.function as Record<string, unknown>;
  const hostShellParameters = hostShellFunction.parameters as Record<string, unknown>;
  assert.equal(hostShellFunction["x-vicuna-family-id"], "host_shell");
  assert.equal(hostShellFunction["x-vicuna-method-name"], "execute");
  assert.match(String(hostShellFunction.description ?? ""), /last-resort/i);
  assert.match(String(hostShellFunction.description ?? ""), /web_search/i);
  assert.equal(typeof hostShellParameters.description, "string");

  const hardMemoryWrite = converted.tools.find((tool) => tool.function.name === "hard_memory_write");
  assert.ok(hardMemoryWrite);
  assert.doesNotMatch(String(hardMemoryWrite.function.description ?? ""), /Supermemory/i);

  const skillCreate = converted.tools.find((tool) => tool.function.name === "skill_create");
  assert.ok(skillCreate);
  assert.match(String(skillCreate.function.description ?? ""), /directly asks/i);
});

test("provider tool conversion excludes capabilities with missing nested schema descriptions", () => {
  const runtimeCatalog = buildRuntimeCatalog();
  const baseCapability = runtimeCatalog.capabilities.find(
    (capability) => capability.tool_name === "media_download",
  );
  assert.ok(baseCapability);
  const brokenCapability = {
    ...baseCapability,
    capability_id: "openclaw.test.broken-media-download",
    tool_surface_id: "vicuna.test.broken_media_download",
    tool_name: "broken_media_download",
    input_schema_json: {
      type: "object",
      properties: {
        query: {
          type: "string",
        },
      },
      required: ["query"],
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

test("invokeRuntimeCapability dispatches flattened media downloads through the correct wrapper", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "media_download",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  const result = await invokeRuntimeCapability(capability, {
    media_kind: "movie",
    query: "Aliens",
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
  assert.deepEqual(result.observation, { ok: true, started: true });
});

test("invokeRuntimeCapability dispatches flattened media reads through supported list actions", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "media_read",
  );
  assert.ok(capability);

  const cases = [
    {
      mediaKind: "movie",
      command: "/tmp/vicuna-openclaw-test/bin/radarr-api",
      expectedAction: "list_movies",
    },
    {
      mediaKind: "series",
      command: "/tmp/vicuna-openclaw-test/bin/sonarr-api",
      expectedAction: "list_series",
    },
    {
      mediaKind: "book",
      command: "/tmp/vicuna-openclaw-test/bin/chaptarr-api",
      expectedAction: "list_books",
    },
  ] as const;

  for (const entry of cases) {
    const calls: Array<{ command: string; args: string[] }> = [];
    await invokeRuntimeCapability(capability, {
      media_kind: entry.mediaKind,
      query: "Alien",
    }, {
      paths: {
        ...defaultPaths("/tmp/vicuna-openclaw-test"),
        radarrWrapperPath: "/tmp/vicuna-openclaw-test/bin/radarr-api",
        sonarrWrapperPath: "/tmp/vicuna-openclaw-test/bin/sonarr-api",
        chaptarrWrapperPath: "/tmp/vicuna-openclaw-test/bin/chaptarr-api",
      },
      execFileImpl: async (command, args) => {
        calls.push({ command, args });
        return {
          stdout: JSON.stringify({ ok: true, entries: [] }),
          stderr: "",
        };
      },
    });

    assert.equal(calls.length, 1);
    assert.equal(calls[0].command, entry.command);
    const payloadArg = calls[0].args.find((value) => value.startsWith("--payload-base64="));
    assert.ok(payloadArg);
    const payload = JSON.parse(
      Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
    );
    assert.equal(payload.action, entry.expectedAction);
    assert.equal(payload.term, "Alien");
  }
});

test("invokeRuntimeCapability preserves downloaded-only flattened media reads", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "media_read",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  await invokeRuntimeCapability(capability, {
    media_kind: "movie",
    query: "Alien",
    status_filter: "downloaded",
  }, {
    paths: {
      ...defaultPaths("/tmp/vicuna-openclaw-test"),
      radarrWrapperPath: "/tmp/vicuna-openclaw-test/bin/radarr-api",
    },
    execFileImpl: async (command, args) => {
      calls.push({ command, args });
      return {
        stdout: JSON.stringify({ ok: true, entries: [] }),
        stderr: "",
      };
    },
  });

  assert.equal(calls.length, 1);
  const payloadArg = calls[0].args.find((value) => value.startsWith("--payload-base64="));
  assert.ok(payloadArg);
  const payload = JSON.parse(
    Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
  );
  assert.equal(payload.action, "list_downloaded_movies");
  assert.equal(payload.term, undefined);
});

test("invokeRuntimeCapability dispatches flattened ongoing-task creation with minute cadence", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "ongoing_task_create",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  const result = await invokeRuntimeCapability(capability, {
    task_text: "Check due tasks.",
    interval: 15,
    unit: "minute",
  }, {
    paths: {
      ...defaultPaths("/tmp/vicuna-openclaw-test"),
      ongoingTasksWrapperPath: "/tmp/vicuna-openclaw-test/bin/ongoing-tasks-api",
    },
    execFileImpl: async (command, args) => {
      calls.push({ command, args });
      return {
        stdout: JSON.stringify({ ok: true, task: { frequency: { interval: 15, unit: "minutes" } } }),
        stderr: "",
      };
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "/tmp/vicuna-openclaw-test/bin/ongoing-tasks-api");
  const payloadArg = calls[0].args.find((entry) => entry.startsWith("--payload-base64="));
  assert.ok(payloadArg);
  const mergedArguments = JSON.parse(
    Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
  );
  assert.equal(mergedArguments.action, "create");
  assert.equal(mergedArguments.interval, 15);
  assert.equal(mergedArguments.unit, "minutes");
  assert.equal(result.observation.task.frequency.unit, "minutes");
});

test("invokeRuntimeCapability dispatches hard-memory writes through the dedicated wrapper", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "hard_memory_write",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  const result = await invokeRuntimeCapability(capability, {
    memories: [{ content: "Remember the markdown backend.", key: "markdown-backend" }],
  }, {
    paths: {
      ...defaultPaths("/tmp/vicuna-openclaw-test"),
      hardMemoryWrapperPath: "/tmp/vicuna-openclaw-test/bin/hard-memory-api",
    },
    execFileImpl: async (command, args) => {
      calls.push({ command, args });
      return {
        stdout: JSON.stringify({ family: "hard_memory", ok: true, written: 1 }),
        stderr: "",
      };
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "/tmp/vicuna-openclaw-test/bin/hard-memory-api");
  const payloadArg = calls[0].args.find((entry) => entry.startsWith("--payload-base64="));
  assert.ok(payloadArg);
  const mergedArguments = JSON.parse(
    Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
  );
  assert.deepEqual(mergedArguments, {
    memories: [{ content: "Remember the markdown backend.", key: "markdown-backend" }],
  });
  assert.deepEqual(result.observation, { family: "hard_memory", ok: true, written: 1 });
});

test("invokeRuntimeCapability dispatches host_shell through the dedicated wrapper", async () => {
  const capability = buildRuntimeCatalog().capabilities.find(
    (entry) => entry.tool_name === "host_shell",
  );
  assert.ok(capability);

  const calls: Array<{ command: string; args: string[] }> = [];
  const result = await invokeRuntimeCapability(capability, {
    command: "pwd",
    purpose: "Inspect the current workspace root.",
    working_directory: "notes",
    timeout_ms: 2500,
  }, {
    paths: {
      ...defaultPaths("/tmp/vicuna-openclaw-test"),
      hostShellWrapperPath: "/tmp/vicuna-openclaw-test/bin/host-shell-api",
    },
    execFileImpl: async (command, args) => {
      calls.push({ command, args });
      return {
        stdout: JSON.stringify({ family: "host_shell", ok: true, summary: "ok" }),
        stderr: "",
      };
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "/tmp/vicuna-openclaw-test/bin/host-shell-api");
  const payloadArg = calls[0].args.find((entry) => entry.startsWith("--payload-base64="));
  assert.ok(payloadArg);
  const mergedArguments = JSON.parse(
    Buffer.from(payloadArg.slice("--payload-base64=".length), "base64").toString("utf8"),
  );
  assert.equal(mergedArguments.command, "pwd");
  assert.equal(mergedArguments.purpose, "Inspect the current workspace root.");
  assert.equal(mergedArguments.working_directory, "notes");
  assert.equal(mergedArguments.timeout_ms, 2500);
  assert.deepEqual(result.observation, { family: "host_shell", ok: true, summary: "ok" });
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
  assert.equal(
    paths.hardMemoryWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "hard-memory-api")
  );
  assert.equal(
    paths.hostShellWrapperPath,
    path.join(tempDir, "tools", "openclaw-harness", "bin", "host-shell-api")
  );
  assert.equal(paths.hostShellRoot, path.join(tempDir, ".cache", "vicuna", "host-shell-home"));
  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH;
  delete process.env.VICUNA_OPENCLAW_TOOL_FABRIC_SECRETS_PATH;
});
