import type { CapabilityCatalog, CapabilityDescriptor } from "./contracts.js";
import { assertCapabilityCatalog } from "./contracts.js";

export type BuiltinToolId = "exec" | "hard_memory_query";

export type CatalogOptions = {
  enabledTools?: BuiltinToolId[];
  enableExec?: boolean;
  enableHardMemoryQuery?: boolean;
};

function execCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.exec.command",
    tool_surface_id: "vicuna.exec.main",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-core",
    tool_name: "exec",
    description: "Run a command through bounded execution policy",
    input_schema_json: {
      type: "object",
      required: ["command"],
      properties: {
        command: { type: "string" },
        workdir: { type: "string" }
      }
    },
    output_contract: "pending_then_result",
    side_effect_class: "system_exec",
    approval_mode: "policy_driven",
    execution_modes: ["sync", "background", "approval_gated"],
    provenance_namespace: "openclaw/openclaw-core/tool/exec",
    tool_kind: 4,
    tool_flags: 0,
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

function hardMemoryCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_query",
    tool_surface_id: "vicuna.memory.hard_query",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_query",
    description: "Query Vicuña hard memory with typed results",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string" }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_read",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_query",
    tool_kind: 2,
    tool_flags: 0,
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_hard_memory"
  };
}

const BUILTIN_CAPABILITIES: Record<BuiltinToolId, () => CapabilityDescriptor> = {
  exec: execCapability,
  hard_memory_query: hardMemoryCapability
};

export function buildCatalog(options: CatalogOptions = {}): CapabilityCatalog {
  const enabledTools = new Set<BuiltinToolId>(
    options.enabledTools ??
      (Object.keys(BUILTIN_CAPABILITIES) as BuiltinToolId[]).filter((toolId) => {
        if (toolId === "exec") {
          return options.enableExec !== false;
        }
        if (toolId === "hard_memory_query") {
          return options.enableHardMemoryQuery !== false;
        }
        return true;
      })
  );
  const capabilities: CapabilityDescriptor[] = [];
  for (const toolId of Object.keys(BUILTIN_CAPABILITIES) as BuiltinToolId[]) {
    if (!enabledTools.has(toolId)) {
      continue;
    }
    capabilities.push(BUILTIN_CAPABILITIES[toolId]());
  }
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities
  });
}
