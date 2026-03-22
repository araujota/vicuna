# Contract: `sonarr` Tool

- Tool name: `sonarr`
- Capability ID: `openclaw.servarr.sonarr`
- Dispatch backend: `legacy_bash`
- Owner plugin: `openclaw-servarr`

Required parameter:

- `action`

Action values:

- `system_status`
- `queue`
- `calendar`
- `root_folders`
- `quality_profiles`
- `list_series`
- `lookup_series`
- `add_series`

The schema must include descriptions for every exposed parameter and must be emitted through the external OpenClaw runtime catalog.
