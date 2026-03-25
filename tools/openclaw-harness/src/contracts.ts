export type CapabilityExecutionMode = "sync" | "background" | "approval_gated";

export type CapabilityDescriptor = {
  capability_id: string;
  tool_surface_id: string;
  capability_kind: string;
  owner_plugin_id: string;
  tool_name: string;
  tool_family_id?: string;
  tool_family_name?: string;
  tool_family_description?: string;
  method_name?: string;
  method_description?: string;
  description: string;
  input_schema_json: Record<string, unknown>;
  fixed_arguments_json?: Record<string, unknown>;
  output_contract: string;
  side_effect_class: string;
  execution_safety_class: string;
  approval_mode: string;
  execution_modes: CapabilityExecutionMode[];
  provenance_namespace: string;
  tool_kind: number;
  tool_flags: number;
  latency_class: number;
  max_steps_reserved: number;
  dispatch_backend: string;
};

export type CapabilityCatalog = {
  catalog_version: number;
  capabilities: CapabilityDescriptor[];
};

export type ToolInvocation = {
  invocation_id: string;
  tool_surface_id: string;
  capability_id: string;
  vicuna_session_id: string;
  vicuna_run_id: string;
  origin_phase: "active" | "dmn";
  arguments_json: Record<string, unknown>;
  requested_mode: CapabilityExecutionMode;
  deadline_ms: number;
  provenance_request_id: string;
};

export type ToolObservationStatus =
  | "pending"
  | "awaiting_approval"
  | "running"
  | "completed"
  | "failed"
  | "rejected"
  | "cancelled"
  | "timed_out";

export type ToolObservation = {
  invocation_id: string;
  status: ToolObservationStatus;
  summary_text: string;
  structured_payload_json: Record<string, unknown>;
  runtime_metrics_json: Record<string, unknown>;
  observed_at_ms: number;
};

function schemaNodeRequiresDescription(node: unknown, isRoot: boolean): boolean {
  if (isRoot || typeof node !== "object" || node === null || Array.isArray(node)) {
    return false;
  }
  const record = node as Record<string, unknown>;
  return "type" in record || "properties" in record || "items" in record;
}

export function assertSchemaDescriptions(node: unknown, path: string, isRoot = false): void {
  if (typeof node !== "object" || node === null || Array.isArray(node)) {
    return;
  }

  const record = node as Record<string, unknown>;
  if (schemaNodeRequiresDescription(record, isRoot)) {
    if (typeof record.description !== "string" || record.description.trim() === "") {
      throw new Error(`input_schema_json is missing a description at ${path}`);
    }
  }

  if ("properties" in record && typeof record.properties === "object" && record.properties !== null && !Array.isArray(record.properties)) {
    for (const [key, value] of Object.entries(record.properties as Record<string, unknown>)) {
      assertSchemaDescriptions(value, `${path}.properties.${key}`);
    }
  }

  if (!("items" in record)) {
    return;
  }

  if (Array.isArray(record.items)) {
    record.items.forEach((item, index) => {
      assertSchemaDescriptions(item, `${path}.items[${index}]`);
    });
    return;
  }

  assertSchemaDescriptions(record.items, `${path}.items`);
}

export function assertCapabilityDescriptor(descriptor: CapabilityDescriptor): CapabilityDescriptor {
  if (!descriptor.capability_id) {
    throw new Error("capability_id is required");
  }
  if (!descriptor.tool_surface_id) {
    throw new Error("tool_surface_id is required");
  }
  if (!descriptor.tool_name) {
    throw new Error("tool_name is required");
  }
  if (!descriptor.provenance_namespace) {
    throw new Error("provenance_namespace is required");
  }
  if (!descriptor.execution_safety_class) {
    throw new Error("execution_safety_class is required");
  }
  if (
    descriptor.fixed_arguments_json !== undefined &&
    (typeof descriptor.fixed_arguments_json !== "object" ||
      descriptor.fixed_arguments_json === null ||
      Array.isArray(descriptor.fixed_arguments_json))
  ) {
    throw new Error("fixed_arguments_json must be an object when provided");
  }
  assertSchemaDescriptions(descriptor.input_schema_json, "input_schema_json", true);
  return descriptor;
}

export function assertCapabilityCatalog(catalog: CapabilityCatalog): CapabilityCatalog {
  const ids = new Set<string>();
  const toolNames = new Set<string>();
  for (const capability of catalog.capabilities) {
    assertCapabilityDescriptor(capability);
    if (ids.has(capability.capability_id)) {
      throw new Error(`duplicate capability_id: ${capability.capability_id}`);
    }
    ids.add(capability.capability_id);
    if (toolNames.has(capability.tool_name)) {
      throw new Error(`duplicate tool_name: ${capability.tool_name}`);
    }
    toolNames.add(capability.tool_name);
  }
  return catalog;
}

export function assertInvocation(invocation: ToolInvocation): ToolInvocation {
  if (!invocation.invocation_id) {
    throw new Error("invocation_id is required");
  }
  if (!invocation.tool_surface_id) {
    throw new Error("tool_surface_id is required");
  }
  if (!invocation.capability_id) {
    throw new Error("capability_id is required");
  }
  return invocation;
}
