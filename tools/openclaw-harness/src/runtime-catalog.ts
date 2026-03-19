import fs from "node:fs";

import { buildRuntimeCatalog, type RuntimeCatalogOptions } from "./catalog.js";
import { ensureParentDir, loadToolSecrets } from "./config.js";

export function loadRuntimeCatalog(options: RuntimeCatalogOptions & { secretsPath: string }) {
  const { secretsPath, ...catalogOptions } = options;
  return buildRuntimeCatalog({
    ...catalogOptions,
    secrets: options.secrets ?? loadToolSecrets(secretsPath)
  });
}

export function writeRuntimeCatalog(runtimeCatalogPath: string, secretsPath: string) {
  const catalog = loadRuntimeCatalog({ secretsPath });
  ensureParentDir(runtimeCatalogPath);
  fs.writeFileSync(runtimeCatalogPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  return catalog;
}
