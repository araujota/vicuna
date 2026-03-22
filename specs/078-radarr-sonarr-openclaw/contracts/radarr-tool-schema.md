# Contract: `radarr` Tool

- Tool name: `radarr`
- Capability ID: `openclaw.servarr.radarr`
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
- `list_movies`
- `lookup_movie`
- `add_movie`

The schema must include descriptions for every exposed parameter and must be emitted through the external OpenClaw runtime catalog.
