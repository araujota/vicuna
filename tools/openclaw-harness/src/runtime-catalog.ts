import fs from "node:fs";

import { buildRuntimeCatalog, type RuntimeCatalogOptions } from "./catalog.js";
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
  ensureParentDir(runtimeCatalogPath);
  fs.writeFileSync(runtimeCatalogPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  return catalog;
}
