import fs from "node:fs";
import path from "node:path";

import { buildRuntimeCatalog, type RuntimeCatalogOptions } from "./catalog.js";
import { buildProviderToolsFromRuntimeCatalog } from "./invoke.js";
import { ensureParentDir, loadToolSecrets } from "./config.js";
import { assertCapabilityCatalog, type CapabilityCatalog } from "./contracts.js";

export function loadRuntimeCatalog(options: RuntimeCatalogOptions & { secretsPath: string }) {
  const { secretsPath, ...catalogOptions } = options;
  return buildRuntimeCatalog({
    ...catalogOptions,
    secrets: options.secrets ?? loadToolSecrets(secretsPath)
  });
}

export function readRuntimeCatalog(runtimeCatalogPath: string): CapabilityCatalog | undefined {
  if (!fs.existsSync(runtimeCatalogPath)) {
    return undefined;
  }
  const raw = fs.readFileSync(runtimeCatalogPath, "utf8");
  if (!raw.trim()) {
    return undefined;
  }
  return assertCapabilityCatalog(JSON.parse(raw) as CapabilityCatalog);
}

export function loadRuntimeCatalogState(
  options: RuntimeCatalogOptions & { secretsPath: string; runtimeCatalogPath?: string }
): CapabilityCatalog {
  const persisted = options.runtimeCatalogPath ? readRuntimeCatalog(options.runtimeCatalogPath) : undefined;
  if (persisted) {
    return persisted;
  }
  return loadRuntimeCatalog(options);
}

export function writeRuntimeCatalog(runtimeCatalogPath: string, secretsPath: string) {
  const catalog = loadRuntimeCatalog({ secretsPath });
  writeJsonAtomically(runtimeCatalogPath, catalog);
  return catalog;
}

export type TelegramRuntimeToolSnapshot = {
  schema_version: "vicuna.telegram_runtime_tools.v1";
  generated_at_ms: number;
  entry_path: string;
  node_bin: string;
  runtime_catalog_path: string;
  tool_count: number;
  tools: unknown[];
};

function writeJsonAtomically(targetPath: string, payload: unknown): void {
  ensureParentDir(targetPath);
  const tempPath = path.join(
    path.dirname(targetPath),
    `.${path.basename(targetPath)}.tmp-${process.pid}-${Date.now()}`
  );
  fs.writeFileSync(tempPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  fs.renameSync(tempPath, targetPath);
}

export function writeTelegramRuntimeToolsSnapshot(options: {
  snapshotPath: string;
  secretsPath: string;
  runtimeCatalogPath: string;
  entryPath: string;
  nodeBin: string;
}): TelegramRuntimeToolSnapshot {
  const runtimeCatalog = loadRuntimeCatalogState({
    secretsPath: options.secretsPath,
    runtimeCatalogPath: options.runtimeCatalogPath,
  });
  const converted = buildProviderToolsFromRuntimeCatalog(runtimeCatalog, {
    excludeToolNames: ["telegram_relay"],
  });
  const snapshot: TelegramRuntimeToolSnapshot = {
    schema_version: "vicuna.telegram_runtime_tools.v1",
    generated_at_ms: Date.now(),
    entry_path: options.entryPath,
    node_bin: options.nodeBin,
    runtime_catalog_path: options.runtimeCatalogPath,
    tool_count: converted.tools.length,
    tools: converted.tools,
  };
  writeJsonAtomically(options.snapshotPath, snapshot);
  return snapshot;
}
