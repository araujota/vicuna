import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { handleSkills, resolveSkillsConfig } from "../src/skills.js";

test("resolveSkillsConfig prefers explicit env and falls back to host-shell-root skills", () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-skills-config-"));
  const secretsPath = path.join(tempRoot, "openclaw-tool-secrets.json");
  fs.writeFileSync(secretsPath, "{}\n", "utf8");

  const previousSkillsDir = process.env.VICUNA_SKILLS_DIR;
  const previousHostShellRoot = process.env.VICUNA_HOST_SHELL_ROOT;
  try {
    delete process.env.VICUNA_SKILLS_DIR;
    process.env.VICUNA_HOST_SHELL_ROOT = "/srv/vicuna-home";
    assert.equal(resolveSkillsConfig(secretsPath).skillsDir, "/srv/vicuna-home/skills");

    process.env.VICUNA_SKILLS_DIR = "/tmp/vicuna-skills";
    assert.equal(resolveSkillsConfig(secretsPath).skillsDir, "/tmp/vicuna-skills");
  } finally {
    if (previousSkillsDir === undefined) {
      delete process.env.VICUNA_SKILLS_DIR;
    } else {
      process.env.VICUNA_SKILLS_DIR = previousSkillsDir;
    }
    if (previousHostShellRoot === undefined) {
      delete process.env.VICUNA_HOST_SHELL_ROOT;
    } else {
      process.env.VICUNA_HOST_SHELL_ROOT = previousHostShellRoot;
    }
  }
});

test("handleSkills reads a markdown skill file by name", async () => {
  const skillsDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-skills-read-"));
  fs.writeFileSync(path.join(skillsDir, "deploy-worker.md"), "# Deploy Worker\n\nUse the deploy path.\n", "utf8");

  const response = await handleSkills({ action: "read", name: "deploy-worker" }, { skillsDir });
  assert.equal(response.ok, true);
  assert.equal(response.name, "deploy-worker");
  assert.match(String(response.path ?? ""), /deploy-worker\.md$/);
  assert.match(String(response.content ?? ""), /Deploy Worker/);
});

test("handleSkills creates a skill file and rejects accidental overwrite", async () => {
  const skillsDir = fs.mkdtempSync(path.join(os.tmpdir(), "vicuna-skills-create-"));

  const createResponse = await handleSkills({
    action: "create",
    name: "Deploy Worker",
    content: "# Deploy Worker\n\nUse the deploy path.\n",
  }, { skillsDir });
  assert.equal(createResponse.ok, true);
  assert.equal(createResponse.created, true);
  assert.deepEqual(fs.readdirSync(skillsDir), ["deploy-worker.md"]);

  await assert.rejects(
    () => handleSkills({
      action: "create",
      name: "Deploy Worker",
      content: "# Deploy Worker\n\nUpdated.\n",
    }, { skillsDir }),
    /already exists/i,
  );

  const updateResponse = await handleSkills({
    action: "create",
    name: "Deploy Worker",
    content: "# Deploy Worker\n\nUpdated.\n",
    overwrite: true,
  }, { skillsDir });
  assert.equal(updateResponse.ok, true);
  assert.equal(updateResponse.updated, true);
  assert.match(fs.readFileSync(path.join(skillsDir, "deploy-worker.md"), "utf8"), /Updated\./);
});
