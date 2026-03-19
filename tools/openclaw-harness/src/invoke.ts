import type { CapabilityCatalog, CapabilityDescriptor, ToolInvocation } from "./contracts.js";
import { assertInvocation } from "./contracts.js";

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
