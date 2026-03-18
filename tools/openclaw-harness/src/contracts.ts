export type CapabilityExecutionMode = "sync" | "background" | "approval_gated";

export type CapabilityDescriptor = {
  capability_id: string;
  tool_surface_id: string;
  capability_kind: string;
  owner_plugin_id: string;
  tool_name: string;
  description: string;
  input_schema_json: Record<string, unknown>;
  output_contract: string;
  side_effect_class: string;
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
  return descriptor;
}

export function assertCapabilityCatalog(catalog: CapabilityCatalog): CapabilityCatalog {
  const ids = new Set<string>();
  for (const capability of catalog.capabilities) {
    assertCapabilityDescriptor(capability);
    if (ids.has(capability.capability_id)) {
      throw new Error(`duplicate capability_id: ${capability.capability_id}`);
    }
    ids.add(capability.capability_id);
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
