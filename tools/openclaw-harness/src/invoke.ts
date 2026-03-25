import { execFile } from "node:child_process";
import { promisify } from "node:util";

import type { CapabilityCatalog, CapabilityDescriptor, ToolInvocation } from "./contracts.js";
import { assertInvocation, assertSchemaDescriptions } from "./contracts.js";
import { defaultPaths, type OpenClawPaths } from "./config.js";

const execFileAsync = promisify(execFile);

export type ProviderToolDefinition = {
  type: "function";
  function: Record<string, unknown>;
};

export type ProviderToolBuildResult = {
  tools: ProviderToolDefinition[];
  excluded: Array<{ capability_id: string; tool_name: string; reason: string }>;
};

export type RuntimeInvocationResult = {
  capability: CapabilityDescriptor;
  mergedArguments: Record<string, unknown>;
  observation: unknown;
};

export function resolveInvocation(
  catalog: CapabilityCatalog,
  invocation: ToolInvocation
): CapabilityDescriptor {
  assertInvocation(invocation);
  const capability = catalog.capabilities.find(
    (entry) =>
      entry.capability_id === invocation.capability_id &&
      entry.tool_surface_id === invocation.tool_surface_id
  );
  if (!capability) {
    throw new Error(
      `unknown capability selection: ${invocation.tool_surface_id} (${invocation.capability_id})`
    );
  }
  return capability;
}

export function resolveCapabilityByToolName(
  catalog: CapabilityCatalog,
  toolName: string
): CapabilityDescriptor {
  const normalizedToolName = String(toolName ?? "").trim();
  const capability = catalog.capabilities.find((entry) => entry.tool_name === normalizedToolName);
  if (!capability) {
    throw new Error(`unknown runtime tool name: ${normalizedToolName}`);
  }
  return capability;
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function normalizedRootDescription(capability: CapabilityDescriptor): string {
  return (
    capability.description?.trim() ||
    capability.method_description?.trim() ||
    capability.tool_family_description?.trim() ||
    `The payload for ${capability.tool_name}.`
  );
}

function ensureProviderSchemaDescription(capability: CapabilityDescriptor): Record<string, unknown> {
  const schema = cloneJson(capability.input_schema_json ?? {});
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error(`runtime capability ${capability.capability_id} has a non-object input schema`);
  }
  const record = schema as Record<string, unknown>;
  if (typeof record.description !== "string" || record.description.trim() === "") {
    record.description = normalizedRootDescription(capability);
  }
  assertSchemaDescriptions(record, "parameters", true);
  return record;
}

function capabilityHasStagedMetadata(capability: CapabilityDescriptor): boolean {
  return Boolean(
    capability.tool_name?.trim() &&
    capability.tool_family_id?.trim() &&
    capability.tool_family_name?.trim() &&
    capability.tool_family_description?.trim() &&
    capability.method_name?.trim() &&
    capability.method_description?.trim()
  );
}

export function buildProviderToolFromCapability(capability: CapabilityDescriptor): ProviderToolDefinition {
  if (!capabilityHasStagedMetadata(capability)) {
    throw new Error(`runtime capability ${capability.capability_id} is missing staged tool metadata`);
  }

  return {
    type: "function",
    function: {
      name: capability.tool_name,
      description: capability.description,
      parameters: ensureProviderSchemaDescription(capability),
      "x-vicuna-family-id": capability.tool_family_id,
      "x-vicuna-family-name": capability.tool_family_name,
      "x-vicuna-family-description": capability.tool_family_description,
      "x-vicuna-method-name": capability.method_name,
      "x-vicuna-method-description": capability.method_description,
      "x-vicuna-tool-surface-id": capability.tool_surface_id,
      "x-vicuna-capability-id": capability.capability_id,
    },
  };
}

export function buildProviderToolsFromRuntimeCatalog(
  catalog: CapabilityCatalog,
  options: { excludeToolNames?: string[] } = {}
): ProviderToolBuildResult {
  const excludedToolNames = new Set((options.excludeToolNames ?? []).map((value) => String(value ?? "").trim()).filter(Boolean));
  const tools: ProviderToolDefinition[] = [];
  const excluded: ProviderToolBuildResult["excluded"] = [];

  for (const capability of catalog.capabilities) {
    if (excludedToolNames.has(capability.tool_name)) {
      excluded.push({
        capability_id: capability.capability_id,
        tool_name: capability.tool_name,
        reason: "excluded_by_name",
      });
      continue;
    }
    try {
      tools.push(buildProviderToolFromCapability(capability));
    } catch (error) {
      excluded.push({
        capability_id: capability.capability_id,
        tool_name: capability.tool_name,
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return { tools, excluded };
}

export function mergeCapabilityArguments(
  capability: CapabilityDescriptor,
  providerArguments: Record<string, unknown>
): Record<string, unknown> {
  const baseArguments =
    capability.fixed_arguments_json && typeof capability.fixed_arguments_json === "object" && !Array.isArray(capability.fixed_arguments_json)
      ? capability.fixed_arguments_json
      : {};
  return {
    ...providerArguments,
    ...baseArguments,
  };
}

function runtimeDispatchCommand(
  capability: CapabilityDescriptor,
  mergedArguments: Record<string, unknown>,
  paths: OpenClawPaths
): { command: string; args: string[] } {
  const payloadBase64 = Buffer.from(JSON.stringify(mergedArguments), "utf8").toString("base64");
  const payloadArgs = [
    `--payload-base64=${payloadBase64}`,
    `--secrets-path=${paths.secretsPath}`,
  ];

  switch (capability.tool_family_id) {
    case "radarr":
      return { command: paths.radarrWrapperPath, args: payloadArgs };
    case "sonarr":
      return { command: paths.sonarrWrapperPath, args: payloadArgs };
    case "chaptarr":
      return { command: paths.chaptarrWrapperPath, args: payloadArgs };
    case "ongoing_tasks":
      return { command: paths.ongoingTasksWrapperPath, args: payloadArgs };
    case "parsed_documents":
      return { command: paths.parsedDocumentsWrapperPath, args: payloadArgs };
    case "telegram":
      return { command: paths.telegramRelayWrapperPath, args: payloadArgs };
    case "web_search": {
      const query = mergedArguments.query;
      if (typeof query !== "string" || query.trim() === "") {
        throw new Error("web_search requires a non-empty query string");
      }
      const args = [
        `--query-url=${encodeURIComponent(query.trim())}`,
        `--secrets-path=${paths.secretsPath}`,
      ];
      if (typeof mergedArguments.topic === "string" && mergedArguments.topic.trim()) {
        args.push(`--topic=${mergedArguments.topic.trim()}`);
      }
      if (typeof mergedArguments.search_depth === "string" && mergedArguments.search_depth.trim()) {
        args.push(`--search-depth=${mergedArguments.search_depth.trim()}`);
      }
      if (typeof mergedArguments.max_results === "number" && Number.isInteger(mergedArguments.max_results)) {
        args.push(`--max-results=${mergedArguments.max_results}`);
      }
      if (typeof mergedArguments.time_range === "string" && mergedArguments.time_range.trim()) {
        args.push(`--time-range=${mergedArguments.time_range.trim()}`);
      }
      if (Array.isArray(mergedArguments.include_domains) && mergedArguments.include_domains.length > 0) {
        args.push(
          `--include-domains=${mergedArguments.include_domains.map((item) => encodeURIComponent(String(item))).join(",")}`
        );
      }
      if (Array.isArray(mergedArguments.exclude_domains) && mergedArguments.exclude_domains.length > 0) {
        args.push(
          `--exclude-domains=${mergedArguments.exclude_domains.map((item) => encodeURIComponent(String(item))).join(",")}`
        );
      }
      if (typeof mergedArguments.country === "string" && mergedArguments.country.trim()) {
        args.push(`--country=${encodeURIComponent(mergedArguments.country.trim())}`);
      }
      return { command: paths.tavilyWrapperPath, args };
    }
    default:
      throw new Error(`runtime capability ${capability.capability_id} has no dispatch mapping`);
  }
}

export async function invokeRuntimeCapability(
  capability: CapabilityDescriptor,
  providerArguments: Record<string, unknown>,
  options: {
    paths?: OpenClawPaths;
    execFileImpl?: typeof execFileAsync;
  } = {}
): Promise<RuntimeInvocationResult> {
  const paths = options.paths ?? defaultPaths();
  const mergedArguments = mergeCapabilityArguments(capability, providerArguments);
  const dispatch = runtimeDispatchCommand(capability, mergedArguments, paths);
  const { stdout, stderr } = await (options.execFileImpl ?? execFileAsync)(dispatch.command, dispatch.args, {
    env: process.env,
    maxBuffer: 32 * 1024 * 1024,
  });
  const rawOutput = String(stdout ?? "").trim();
  if (!rawOutput) {
    throw new Error(
      `runtime capability ${capability.capability_id} produced no stdout${stderr?.trim() ? `: ${stderr.trim()}` : ""}`
    );
  }

  let observation: unknown = rawOutput;
  try {
    observation = JSON.parse(rawOutput);
  } catch {
    observation = rawOutput;
  }

  return {
    capability,
    mergedArguments,
    observation,
  };
}

