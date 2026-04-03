import { buildCatalog } from "./catalog.js";
import { defaultPaths, loadToolSecrets } from "./config.js";
import { isCliEntrypoint } from "./cli-entrypoint.js";
import {
  buildProviderToolsFromRuntimeCatalog,
  invokeRuntimeCapability,
  resolveCapabilityByToolName,
} from "./invoke.js";
import {
  loadRuntimeCatalogState,
  writeRuntimeCatalog,
  writeTelegramRuntimeToolsSnapshot,
} from "./runtime-catalog.js";

export * from "./contracts.js";
export * from "./catalog.js";
export * from "./config.js";
export * from "./invoke.js";
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
  TelegramRelayToolError,
  errorEnvelope as telegramRelayErrorEnvelope,
  handleTelegramRelay,
  parseCliInvocation as parseTelegramRelayCliInvocation,
  resolveTelegramRelayConfig,
  type TelegramRelayConfig,
  type TelegramRelayInvocation,
  type TelegramRelayResponseEnvelope,
} from "./telegram-relay.js";

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

if (isCliEntrypoint(import.meta.url, process.argv[1])) {
  const command = process.argv[2] ?? "catalog";
  const flags = parseCliFlags(process.argv.slice(3));
  const paths = defaultPaths();

  if (command === "catalog") {
    process.stdout.write(`${JSON.stringify(buildCatalog(), null, 2)}\n`);
  } else if (command === "runtime-catalog") {
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath,
          catalog: loadRuntimeCatalogState({
            secretsPath: paths.secretsPath,
            runtimeCatalogPath: paths.runtimeCatalogPath,
          }),
        },
        null,
        2
      )}\n`
    );
  } else if (command === "runtime-tools") {
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
              ...(result.correctness ? { correctness: result.correctness } : {}),
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
    writeRuntimeCatalog(paths.runtimeCatalogPath, paths.secretsPath);
    process.stdout.write(
      `${JSON.stringify(
        {
          runtime_catalog_path: paths.runtimeCatalogPath,
        },
        null,
        2
      )}\n`
    );
  } else if (command === "publish-telegram-runtime-tools") {
    const snapshotPath = flags.get("snapshot-path")?.trim() || paths.telegramRuntimeToolsSnapshotPath;
    const snapshot = writeTelegramRuntimeToolsSnapshot({
      snapshotPath,
      secretsPath: paths.secretsPath,
      runtimeCatalogPath: paths.runtimeCatalogPath,
      entryPath: process.argv[1] ?? "",
      nodeBin: process.execPath,
    });
    process.stdout.write(
      `${JSON.stringify(
        {
          snapshot_path: snapshotPath,
          tool_count: snapshot.tool_count,
        },
        null,
        2
      )}\n`
    );
  } else if (command === "show-secrets") {
    process.stdout.write(`${JSON.stringify(loadToolSecrets(paths.secretsPath), null, 2)}\n`);
  } else {
    throw new Error(`unsupported command: ${command}`);
  }
}
