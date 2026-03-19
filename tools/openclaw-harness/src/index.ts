import { buildCatalog } from "./catalog.js";
import { defaultPaths, loadToolSecrets, saveToolSecrets, upsertTavilyApiKey } from "./config.js";
import { resolveInvocation } from "./invoke.js";
import { loadRuntimeCatalog, writeRuntimeCatalog } from "./runtime-catalog.js";

export * from "./contracts.js";
export * from "./catalog.js";
export * from "./config.js";
export * from "./invoke.js";
export * from "./runtime-catalog.js";

function defaultCliPaths() {
  return defaultPaths();
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const command = process.argv[2] ?? "catalog";
  if (command === "catalog") {
    process.stdout.write(`${JSON.stringify(buildCatalog(), null, 2)}\n`);
  } else if (command === "runtime-catalog") {
    const paths = defaultCliPaths();
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath,
          catalog: loadRuntimeCatalog({ secretsPath: paths.secretsPath })
        },
        null,
        2
      )}\n`
    );
  } else if (command === "sync-runtime-catalog") {
    const paths = defaultCliPaths();
    writeRuntimeCatalog(paths.runtimeCatalogPath, paths.secretsPath);
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath
        },
        null,
        2
      )}\n`
    );
  } else if (command === "install-tavily") {
    const apiKey = process.argv[3];
    if (!apiKey) {
      throw new Error("install-tavily requires an api key argument");
    }
    const paths = defaultCliPaths();
    const secrets = upsertTavilyApiKey(loadToolSecrets(paths.secretsPath), apiKey);
    saveToolSecrets(paths.secretsPath, secrets);
    writeRuntimeCatalog(paths.runtimeCatalogPath, paths.secretsPath);
    process.stdout.write(
      `${JSON.stringify(
        {
          secrets_path: paths.secretsPath,
          runtime_catalog_path: paths.runtimeCatalogPath,
          tavily_enabled: true
        },
        null,
        2
      )}\n`
    );
  } else if (command === "validate") {
    const payload = process.argv[3];
    if (!payload) {
      throw new Error("validate requires a JSON payload");
    }
    const invocation = JSON.parse(payload);
    const capability = resolveInvocation(buildCatalog(), invocation);
    process.stdout.write(`${JSON.stringify(capability, null, 2)}\n`);
  } else {
    throw new Error(`unknown command: ${command}`);
  }
}
