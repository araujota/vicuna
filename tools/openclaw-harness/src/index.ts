import { buildCatalog } from "./catalog.js";
import { resolveInvocation } from "./invoke.js";

export * from "./contracts.js";
export * from "./catalog.js";
export * from "./invoke.js";

if (import.meta.url === `file://${process.argv[1]}`) {
  const command = process.argv[2] ?? "catalog";
  if (command === "catalog") {
    process.stdout.write(`${JSON.stringify(buildCatalog(), null, 2)}\n`);
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
