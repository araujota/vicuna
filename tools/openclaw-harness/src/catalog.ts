import type { CapabilityCatalog, CapabilityDescriptor } from "./contracts.js";
import { assertCapabilityCatalog } from "./contracts.js";
import type { OpenClawToolSecrets } from "./config.js";

export type BuiltinToolId =
  | "hard_memory_query"
  | "hard_memory_write"
  | "skill_read"
  | "skill_create"
  | "host_shell"
  | "telegram_relay";

export type CatalogOptions = {
  enabledTools?: BuiltinToolId[];
};

export type RuntimeCatalogOptions = {
  secrets?: OpenClawToolSecrets;
};

function capability(
  descriptor: CapabilityDescriptor
): CapabilityDescriptor {
  return descriptor;
}

function hardMemoryQueryCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.hard_memory_query",
    tool_surface_id: "vicuna.memory.hard_query",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_query",
    tool_family_id: "hard_memory",
    tool_family_name: "Hard Memory",
    tool_family_description: "Read or write durable host-owned memory records.",
    method_name: "query",
    method_description: "Query durable hard memory.",
    description: "Query durable host-owned hard memory.",
    input_schema_json: {
      type: "object",
      description: "Payload for reading hard memory.",
      required: ["query"],
      properties: {
        query: { type: "string", description: "The retrieval query." },
        limit: { type: "integer", description: "Optional result cap." },
        domain: { type: "string", description: "Optional domain filter." },
      },
      additionalProperties: false,
    },
    output_contract: "completed_result",
    side_effect_class: "memory_read",
    execution_safety_class: "read_only",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/query",
    tool_kind: 2,
    tool_flags: 0,
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "hard_memory",
  });
}

function hardMemoryWriteCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.hard_memory_write",
    tool_surface_id: "vicuna.memory.hard_write",
    capability_kind: "memory_adapter",
    owner_plugin_id: "vicuna-memory",
    tool_name: "hard_memory_write",
    tool_family_id: "hard_memory",
    tool_family_name: "Hard Memory",
    tool_family_description: "Read or write durable host-owned memory records.",
    method_name: "write",
    method_description: "Write durable hard memory records.",
    description: "Write durable host-owned hard memory.",
    input_schema_json: {
      type: "object",
      description: "Payload for writing hard memory.",
      required: ["memories"],
      properties: {
        memories: {
          type: "array",
          description: "The records to persist.",
          minItems: 1,
          items: {
            type: "object",
            description: "One memory record.",
            required: ["content"],
            properties: {
              content: { type: "string", description: "Memory content." },
              title: { type: "string", description: "Optional title." },
              key: { type: "string", description: "Optional stable key." },
              kind: { type: "string", description: "Optional kind label." },
              domain: { type: "string", description: "Optional domain label." },
              tags: {
                type: "array",
                description: "Optional tags.",
                items: { type: "string", description: "One tag." },
              },
            },
            additionalProperties: true,
          },
        },
        containerTag: { type: "string", description: "Optional container tag." },
      },
      additionalProperties: false,
    },
    output_contract: "completed_result",
    side_effect_class: "memory_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-memory/write",
    tool_kind: 3,
    tool_flags: 0,
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "hard_memory",
  });
}

function skillReadCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.skill_read",
    tool_surface_id: "vicuna.skills.read",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-skills",
    tool_name: "skill_read",
    tool_family_id: "skills",
    tool_family_name: "Skills",
    tool_family_description: "Read and write host-owned skill markdown files.",
    method_name: "read",
    method_description: "Read one skill file.",
    description: "Read one host-owned skill file.",
    input_schema_json: {
      type: "object",
      description: "Payload for reading one skill.",
      required: ["name"],
      properties: {
        name: { type: "string", description: "The skill name or file stem." },
      },
      additionalProperties: false,
    },
    output_contract: "completed_result",
    side_effect_class: "filesystem_read",
    execution_safety_class: "read_only",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-skills/read",
    tool_kind: 4,
    tool_flags: 0,
    latency_class: 1,
    max_steps_reserved: 2,
    dispatch_backend: "skills",
  });
}

function skillCreateCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.skill_create",
    tool_surface_id: "vicuna.skills.create",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-skills",
    tool_name: "skill_create",
    tool_family_id: "skills",
    tool_family_name: "Skills",
    tool_family_description: "Read and write host-owned skill markdown files.",
    method_name: "create",
    method_description: "Create or update one skill file.",
    description: "Create or update one host-owned skill file.",
    input_schema_json: {
      type: "object",
      description: "Payload for writing one skill.",
      required: ["name", "content"],
      properties: {
        name: { type: "string", description: "The skill name or file stem." },
        content: { type: "string", description: "Markdown content for the skill." },
      },
      additionalProperties: false,
    },
    output_contract: "completed_result",
    side_effect_class: "filesystem_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-skills/create",
    tool_kind: 4,
    tool_flags: 0,
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "skills",
  });
}

function hostShellCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.host_shell",
    tool_surface_id: "vicuna.host.shell",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-host-shell",
    tool_name: "host_shell",
    tool_family_id: "host_shell",
    tool_family_name: "Host Shell",
    tool_family_description: "Run bounded host shell commands in the host sandbox.",
    method_name: "run",
    method_description: "Run one host-shell command.",
    description: "Run one bounded host-shell command.",
    input_schema_json: {
      type: "object",
      description: "Payload for a host-shell command.",
      required: ["command"],
      properties: {
        command: { type: "string", description: "The shell command to run." },
        cwd: { type: "string", description: "Optional working directory." },
      },
      additionalProperties: false,
    },
    output_contract: "completed_result",
    side_effect_class: "filesystem_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-host-shell/run",
    tool_kind: 4,
    tool_flags: 0,
    latency_class: 2,
    max_steps_reserved: 3,
    dispatch_backend: "host_shell",
  });
}

function telegramRelayCapability(): CapabilityDescriptor {
  return capability({
    capability_id: "openclaw.vicuna.telegram_relay",
    tool_surface_id: "vicuna.telegram.relay",
    capability_kind: "tool",
    owner_plugin_id: "vicuna-telegram",
    tool_name: "telegram_relay",
    tool_family_id: "telegram",
    tool_family_name: "Telegram Relay",
    tool_family_description: "Send bounded relay actions through the Telegram bridge.",
    method_name: "relay",
    method_description: "Send a Telegram relay action.",
    description: "Send a bounded action through the Telegram relay.",
    input_schema_json: {
      type: "object",
      description: "Payload for the Telegram relay wrapper.",
      required: ["action"],
      properties: {
        action: { type: "string", description: "The relay action to invoke." },
        payload: { type: "object", description: "Optional relay payload." },
      },
      additionalProperties: true,
    },
    output_contract: "completed_result",
    side_effect_class: "network_write",
    execution_safety_class: "approval_required",
    approval_mode: "none",
    execution_modes: ["sync"],
    provenance_namespace: "openclaw/vicuna-telegram/relay",
    tool_kind: 4,
    tool_flags: 0,
    latency_class: 2,
    max_steps_reserved: 2,
    dispatch_backend: "telegram",
  });
}

const ALL_CAPABILITIES = [
  hardMemoryQueryCapability,
  hardMemoryWriteCapability,
  skillReadCapability,
  skillCreateCapability,
  hostShellCapability,
  telegramRelayCapability,
];

export function buildCatalog(options: CatalogOptions = {}): CapabilityCatalog {
  const enabled = options.enabledTools ? new Set(options.enabledTools) : undefined;
  const capabilities = ALL_CAPABILITIES
    .map((factory) => factory())
    .filter((entry) => !enabled || enabled.has(entry.tool_name as BuiltinToolId));
  return assertCapabilityCatalog({
    catalog_version: 1,
    capabilities,
  });
}

export function buildRuntimeCatalog(_options: RuntimeCatalogOptions = {}): CapabilityCatalog {
  return buildCatalog();
}
