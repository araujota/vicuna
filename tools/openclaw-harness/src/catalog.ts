import type { CapabilityCatalog, CapabilityDescriptor } from "./contracts.js";
import { assertCapabilityCatalog } from "./contracts.js";
import type { OpenClawToolSecrets } from "./config.js";

export type BuiltinToolId = "exec" | "hard_memory_query" | "hard_memory_write" | "codex";

export type CatalogOptions = {
  enabledTools?: BuiltinToolId[];
  enableExec?: boolean;
  enableHardMemoryQuery?: boolean;
  enableHardMemoryWrite?: boolean;
  enableCodex?: boolean;
};

export type RuntimeCatalogOptions = {
  secrets?: OpenClawToolSecrets;
};

const COG_TOOL_FLAG_ACTIVE_ELIGIBLE = 1 << 0;
const COG_TOOL_FLAG_DMN_ELIGIBLE = 1 << 1;
const COG_TOOL_FLAG_SIMULATION_SAFE = 1 << 2;
const COG_TOOL_FLAG_REMEDIATION_SAFE = 1 << 3;
const COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT = 1 << 4;

function combineToolFlags(...flags: number[]): number {
  return flags.reduce((mask, flag) => mask | flag, 0);
}

function execCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.exec.command",
    tool_surface_id: "vicuna.exec.main",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-core",
    tool_name: "exec",
    description: "Run one bounded command invocation through the execution policy; do not use shell chaining or redirection",
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
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
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
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_SIMULATION_SAFE,
      COG_TOOL_FLAG_REMEDIATION_SAFE
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_hard_memory"
  };
}

function hardMemoryWriteCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.hard_memory_write",
    tool_surface_id: "vicuna.memory.hard_write",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_write",
    description: "Archive explicit durable memories to Vicuña hard memory and Supermemory",
    input_schema_json: {
      type: "object",
      required: ["memories"],
      properties: {
        memories: {
          type: "array",
          minItems: 1,
          items: {
            type: "object",
            required: ["content"],
            properties: {
              content: { type: "string" },
              title: { type: "string" },
              key: { type: "string" },
              kind: { type: "string" },
              domain: { type: "string" },
              tags: { type: "array", items: { type: "string" } },
              importance: { type: "number" },
              confidence: { type: "number" },
              gainBias: { type: "number" },
              allostaticRelevance: { type: "number" },
              isStatic: { type: "boolean" }
            }
          }
        },
        containerTag: { type: "string" }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/memory_adapter/hard_memory_write",
    tool_kind: 3,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_hard_memory"
  };
}

function codexCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.vicuna.codex_cli",
    tool_surface_id: "vicuna.codex.main",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-runtime",
    tool_name: "codex",
    description: "Use the local Codex CLI to implement a repository change and rebuild the runtime",
    input_schema_json: {
      type: "object",
      required: ["task"],
      properties: {
        task: { type: "string" }
      }
    },
    output_contract: "pending_then_result",
    side_effect_class: "self_modification",
    approval_mode: "none",
    execution_modes: ["background"],
    provenance_namespace: "openclaw/vicuna-runtime/tool/codex",
    tool_kind: 5,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "legacy_codex"
  };
}

function tavilyWebSearchCapability(): CapabilityDescriptor {
  return {
    capability_id: "openclaw.tavily.web_search",
    tool_surface_id: "vicuna.web.search.tavily",
    capability_kind: "tool",
    owner_plugin_id: "openclaw-tavily",
    tool_name: "web_search",
    description: "Search the live web through Tavily and return ranked results with snippets",
    input_schema_json: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string" },
        topic: { type: "string", enum: ["general", "news"] },
        search_depth: { type: "string", enum: ["basic", "advanced"] },
        max_results: { type: "integer", minimum: 1, maximum: 10 }
      }
    },
    output_contract: "completed_result",
    side_effect_class: "network_read",
    approval_mode: "policy_driven",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/openclaw-tavily/tool/web_search",
    tool_kind: 4,
    tool_flags: combineToolFlags(
      COG_TOOL_FLAG_ACTIVE_ELIGIBLE,
      COG_TOOL_FLAG_DMN_ELIGIBLE,
      COG_TOOL_FLAG_REMEDIATION_SAFE,
      COG_TOOL_FLAG_EXTERNAL_SIDE_EFFECT
    ),
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "legacy_bash"
  };
}

const BUILTIN_CAPABILITIES: Record<BuiltinToolId, () => CapabilityDescriptor> = {
  exec: execCapability,
  hard_memory_query: hardMemoryCapability,
  hard_memory_write: hardMemoryWriteCapability,
  codex: codexCapability
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
        if (toolId === "hard_memory_write") {
          return options.enableHardMemoryWrite !== false;
        }
        if (toolId === "codex") {
          return options.enableCodex !== false;
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

export function buildRuntimeCatalog(options: RuntimeCatalogOptions = {}): CapabilityCatalog {
  const capabilities: CapabilityDescriptor[] = [];
  const tavilyApiKey = options.secrets?.tools?.tavily?.api_key?.trim();
  if (tavilyApiKey) {
    capabilities.push(tavilyWebSearchCapability());
  }
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities
  });
}
