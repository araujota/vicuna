import { execFile } from "node:child_process";
import { promisify } from "node:util";

import type {
  CapabilityCatalog,
  CapabilityDescriptor,
  ToolCorrectnessSignal,
  ToolInvocation,
} from "./contracts.js";
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
  correctness?: ToolCorrectnessSignal;
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
  const excludedToolNames = new Set(
    (options.excludeToolNames ?? []).map((value) => String(value ?? "").trim()).filter(Boolean)
  );
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
  const payloadArgs = [`--payload-base64=${payloadBase64}`, `--secrets-path=${paths.secretsPath}`];

  switch (capability.dispatch_backend) {
    case "hard_memory":
      return { command: paths.hardMemoryWrapperPath, args: payloadArgs };
    case "skills":
      return { command: paths.skillsWrapperPath, args: payloadArgs };
    case "host_shell":
      return { command: paths.hostShellWrapperPath, args: payloadArgs };
    case "telegram":
      return { command: paths.telegramRelayWrapperPath, args: payloadArgs };
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
  let correctness: ToolCorrectnessSignal | undefined;
  try {
    observation = JSON.parse(rawOutput);
  } catch {
    observation = rawOutput;
  }

  if (observation && typeof observation === "object" && !Array.isArray(observation)) {
    const record = observation as Record<string, unknown>;
    const candidate = record.correctness;
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      correctness = candidate as ToolCorrectnessSignal;
    }
    if ("observation" in record) {
      observation = record.observation;
    }
  }

  return {
    capability,
    mergedArguments,
    observation,
    ...(correctness ? { correctness } : {}),
  };
}
