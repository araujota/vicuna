import fs from "node:fs";
import { fileURLToPath } from "node:url";

function realpathOrOriginal(value: string): string {
  try {
    return fs.realpathSync(value);
  } catch {
    return value;
  }
}

export function isCliEntrypoint(importMetaUrl: string, argv1: string | undefined): boolean {
  if (!argv1) {
    return false;
  }
  const modulePath = realpathOrOriginal(fileURLToPath(importMetaUrl));
  const invokedPath = realpathOrOriginal(argv1);
  return modulePath === invokedPath;
}
