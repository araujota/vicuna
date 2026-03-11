# Supermemory Self-Hosting Bootstrap

This directory contains the Vicuña-side bootstrap for the publicly reproducible
Supermemory MCP worker deployment path.

## What This Does

`bootstrap-mcp-worker.sh` prepares a local deployable copy of upstream
`supermemoryai/supermemory/apps/mcp`, rewrites the Wrangler config so it is
deployable on a generic Cloudflare account, installs the minimum runtime
dependencies with `npm`, stages the checked-in `mcp-app.html` into `dist/`, and
can optionally run a Wrangler dry run or deploy.

## What This Does Not Do

It does **not** fully self-host the entire Supermemory API stack.

Upstream currently documents full API self-hosting in
`apps/docs/deployment/self-hosting.mdx` as an **enterprise-only** workflow that
requires a deployment package from the Supermemory team, plus additional
database and provider infrastructure. This repo can bootstrap the public-source
MCP worker slice now, and that worker can later point at a future private API
via `API_URL`.

## Requirements

- `git`
- `node`
- `npm`
- Wrangler auth

Wrangler resolution order:

1. `wrangler` on `PATH`
2. `WRANGLER_BIN`
3. the machine-local fallback path discovered during implementation:
   `/Users/tyleraraujo/arcagent/worker/cloudflare/node_modules/wrangler/bin/wrangler.js`

## Usage

Prepare only:

```bash
tools/supermemory-selfhost/bootstrap-mcp-worker.sh prepare
```

Prepare and dry-run deploy:

```bash
tools/supermemory-selfhost/bootstrap-mcp-worker.sh dry-run
```

Prepare and deploy:

```bash
tools/supermemory-selfhost/bootstrap-mcp-worker.sh deploy
```

Override the target API URL and worker name:

```bash
tools/supermemory-selfhost/bootstrap-mcp-worker.sh dry-run \
  --api-url https://api.example.internal \
  --worker-name vicuna-supermemory-mcp-private
```

If the machine is low on disk, you can either free space or prepare the
workspace without installing dependencies:

```bash
tools/supermemory-selfhost/bootstrap-mcp-worker.sh prepare --skip-install
```

The install preflight defaults to requiring `512 MiB` free and can be adjusted
with `SUPERMEMORY_MCP_MIN_FREE_MB`.

## Generated Artifacts

By default the script writes to:

- upstream cache: `~/.cache/supermemory-upstream`
- generated worker workspace:
  `/Users/tyleraraujo/vicuna/tools/supermemory-selfhost/workdir/mcp-worker`

Inside the generated workspace, the key local artifact is:

- `wrangler.vicuna.jsonc`

This config intentionally:

- preserves the upstream Durable Object binding and migration
- removes the upstream Vite build requirement by staging `mcp-app.html`
  directly into `dist/`
- removes upstream `mcp.supermemory.ai` custom-domain routing
- sets `workers_dev` deployment semantics for a generic Cloudflare account
- keeps `API_URL` explicit for future private/self-hosted API routing
