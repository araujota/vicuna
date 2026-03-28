import { buildCatalog } from "./catalog.js";
import {
  defaultPaths,
  loadToolSecrets,
  saveToolSecrets,
  upsertApiToolConfig,
  upsertServarrConfig,
  upsertTavilyApiKey
} from "./config.js";
import {
  buildProviderToolsFromRuntimeCatalog,
  invokeRuntimeCapability,
  resolveCapabilityByToolName,
  resolveInvocation,
} from "./invoke.js";
import { loadRuntimeCatalogState, writeRuntimeCatalog } from "./runtime-catalog.js";

export * from "./contracts.js";
export * from "./catalog.js";
export * from "./config.js";
export * from "./invoke.js";
export * from "./ongoing-tasks.js";
export * from "./runtime-catalog.js";
export {
  HardMemoryToolError,
  errorEnvelope as hardMemoryErrorEnvelope,
  handleHardMemory,
  parseCliInvocation as parseHardMemoryCliInvocation,
  resolveHardMemoryConfig,
  runHardMemoryCli,
  type HardMemoryConfig,
  type HardMemoryInvocation,
  type HardMemoryQueryResult,
  type HardMemoryResponseEnvelope,
} from "./hard-memory.js";
export {
  SkillsToolError,
  handleSkills,
  parseCliInvocation as parseSkillsCliInvocation,
  resolveSkillsConfig,
  runSkillsCli,
  type SkillsConfig,
  type SkillsInvocation,
  type SkillsResponseEnvelope,
} from "./skills.js";
export {
  HostShellToolError,
  handleHostShell,
  parseCliInvocation as parseHostShellCliInvocation,
  resolveHostShellConfig,
  runHostShellCli,
  type HostShellConfig,
  type HostShellInvocation,
  type HostShellOutputSummary,
  type HostShellResponseEnvelope,
} from "./host-shell.js";
export {
  ParsedDocumentsToolError,
  errorEnvelope as parsedDocumentsErrorEnvelope,
  handleParsedDocuments,
  parseCliInvocation as parseParsedDocumentsCliInvocation,
  resolveParsedDocumentsConfig,
  runParsedDocumentsCli,
  type ParsedDocumentsConfig,
  type ParsedDocumentsInvocation,
  type ParsedDocumentsResponseEnvelope,
  type ParsedDocumentSearchResult,
} from "./parsed-documents.js";
export {
  TelegramRelayToolError,
  errorEnvelope as telegramRelayErrorEnvelope,
  handleTelegramRelay,
  parseCliInvocation as parseTelegramRelayCliInvocation,
  resolveTelegramRelayConfig,
  type TelegramRelayConfig,
  type TelegramRelayInvocation,
  type TelegramRelayResponseEnvelope,
} from "./telegram-relay.js";

function defaultCliPaths() {
  return defaultPaths();
}

function parseCliFlags(argv: string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (const item of argv) {
    if (!item.startsWith("--")) {
      continue;
    }
    const withoutPrefix = item.slice(2);
    const separator = withoutPrefix.indexOf("=");
    if (separator < 0) {
      flags.set(withoutPrefix, "true");
      continue;
    }
    flags.set(withoutPrefix.slice(0, separator), withoutPrefix.slice(separator + 1));
  }
  return flags;
}

function parseJsonObjectArgument(encoded: string | undefined, flagName: string): Record<string, unknown> {
  if (!encoded) {
    return {};
  }
  const decoded = Buffer.from(encoded, "base64").toString("utf8");
  const parsed = JSON.parse(decoded);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${flagName} must decode to a JSON object`);
  }
  return parsed as Record<string, unknown>;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const command = process.argv[2] ?? "catalog";
  const flags = parseCliFlags(process.argv.slice(3));
  if (command === "catalog") {
    process.stdout.write(`${JSON.stringify(buildCatalog(), null, 2)}\n`);
  } else if (command === "runtime-catalog") {
    const paths = defaultCliPaths();
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath,
          catalog: loadRuntimeCatalogState({
            secretsPath: paths.secretsPath,
            runtimeCatalogPath: paths.runtimeCatalogPath,
          })
        },
        null,
        2
      )}\n`
    );
  } else if (command === "runtime-tools") {
    const paths = defaultCliPaths();
    const excludeToolNames = (flags.get("exclude-tool-name") ?? "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    const catalog = loadRuntimeCatalogState({
      secretsPath: paths.secretsPath,
      runtimeCatalogPath: paths.runtimeCatalogPath,
    });
    const tools = buildProviderToolsFromRuntimeCatalog(catalog, { excludeToolNames });
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath,
          tools: tools.tools,
          excluded: tools.excluded,
        },
        null,
        2
      )}\n`
    );
  } else if (command === "invoke-runtime") {
    const toolName = flags.get("tool-name")?.trim();
    if (!toolName) {
      throw new Error("invoke-runtime requires --tool-name=<name>");
    }
    const providerArguments = parseJsonObjectArgument(flags.get("arguments-base64"), "--arguments-base64");
    const paths = defaultCliPaths();
    const catalog = loadRuntimeCatalogState({
      secretsPath: paths.secretsPath,
      runtimeCatalogPath: paths.runtimeCatalogPath,
    });
    const capability = resolveCapabilityByToolName(catalog, toolName);
    invokeRuntimeCapability(capability, providerArguments, { paths })
      .then((result) => {
        process.stdout.write(
          `${JSON.stringify(
            {
              tool_name: capability.tool_name,
              capability_id: capability.capability_id,
              tool_surface_id: capability.tool_surface_id,
              merged_arguments: result.mergedArguments,
              observation: result.observation,
            },
            null,
            2
          )}\n`
        );
      })
      .catch((error) => {
        process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
        process.exitCode = 1;
      });
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
  } else if (command === "install-radarr" || command === "install-sonarr" || command === "install-chaptarr") {
    const apiKey = process.argv[3];
    if (!apiKey) {
      throw new Error(`${command} requires an api key argument`);
    }
    const optionalBaseUrl = process.argv[4];
    const paths = defaultCliPaths();
    const toolId =
      command === "install-radarr"
        ? "radarr"
        : command === "install-sonarr"
          ? "sonarr"
          : "chaptarr";
    const secrets =
      toolId === "chaptarr"
        ? upsertApiToolConfig(loadToolSecrets(paths.secretsPath), toolId, apiKey, optionalBaseUrl)
        : upsertServarrConfig(loadToolSecrets(paths.secretsPath), toolId, apiKey, optionalBaseUrl);
    saveToolSecrets(paths.secretsPath, secrets);
    writeRuntimeCatalog(paths.runtimeCatalogPath, paths.secretsPath);
    process.stdout.write(
      `${JSON.stringify(
        {
          secrets_path: paths.secretsPath,
          runtime_catalog_path: paths.runtimeCatalogPath,
          tool: toolId,
          installed: true
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
