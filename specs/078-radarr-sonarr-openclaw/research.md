# Research: Radarr and Sonarr OpenClaw Tools

## Local Audit

- The runtime already has one authoritative tool surface: `server_openclaw_fabric`.
- Builtin tools are registered in `tools/server/server-openclaw-fabric.cpp`.
- External tools are emitted through the TypeScript harness runtime catalog in `tools/openclaw-harness/src/catalog.ts` and loaded by `server_openclaw_fabric::load_external_catalog(...)`.
- The existing external pattern is Tavily `web_search`, which enters the same capability contract, same XML guidance, and same dispatch path as builtin tools.
- The host currently stores OpenClaw secrets in `/home/tyler-araujo/Projects/vicuna/.cache/vicuna/openclaw-tool-secrets.json`, but only Tavily is configured there today.

## Host and Network Findings

- From the Vicuña host on the LAN, `10.0.0.218` exposes:
  - `5000` / `5001`, consistent with a NAS web interface
  - `7878`, returning `401 Unauthorized` from Kestrel, consistent with Radarr
  - `8989`, returning `401 Unauthorized` from Kestrel, consistent with Sonarr
- No existing Radarr/Sonarr API key or base-URL config is present in the OpenClaw secrets file or `/etc/vicuna/vicuna.env`.

## GitHub Primary-Source Research

### Radarr

Primary source: `Radarr/Radarr`, `src/Radarr.Api.V3/openapi.json`

Key findings from the official OpenAPI file:

- Default server hostpath is `localhost:7878`
- Official auth schemes are:
  - `X-Api-Key` header
  - `apikey` query parameter
- Relevant official endpoints:
  - `GET /api/v3/system/status`
  - `GET /api/v3/queue`
  - `GET /api/v3/rootfolder`
  - `GET /api/v3/qualityprofile`
  - `GET /api/v3/movie`
  - `GET /api/v3/movie/lookup?term=...`
  - `POST /api/v3/movie`
  - `GET /api/v3/calendar`
- `POST /api/v3/movie` accepts a `MovieResource`
- `MovieResource` includes fields needed for add flows such as `qualityProfileId`, `rootFolderPath`, `monitored`, `minimumAvailability`, `tags`, and `addOptions`
- `AddMovieOptions` includes:
  - `monitor`
  - `searchForMovie`
  - `addMethod`

### Sonarr

Primary source: `Sonarr/Sonarr`, `src/Sonarr.Api.V3/openapi.json`

Key findings from the official OpenAPI file:

- Default server hostpath is `localhost:8989`
- Official auth schemes are:
  - `X-Api-Key` header
  - `apikey` query parameter
- Relevant official endpoints:
  - `GET /api/v3/system/status`
  - `GET /api/v3/queue`
  - `GET /api/v3/rootfolder`
  - `GET /api/v3/qualityprofile`
  - `GET /api/v3/series`
  - `GET /api/v3/series/lookup?term=...`
  - `POST /api/v3/series`
  - `GET /api/v3/calendar`
- `POST /api/v3/series` accepts a `SeriesResource`
- `SeriesResource` includes fields needed for add flows such as `qualityProfileId`, `rootFolderPath`, `seasonFolder`, `monitored`, `monitorNewItems`, `seriesType`, `tags`, and `addOptions`
- `AddSeriesOptions` includes:
  - `monitor`
  - `searchForMissingEpisodes`
  - `searchForCutoffUnmetEpisodes`

## Design Conclusions

- Radarr and Sonarr should be added as external OpenClaw capabilities in the TypeScript runtime catalog, not as server-only builtins. That keeps one authoritative tool fabric.
- The correct surface is one `radarr` tool and one `sonarr` tool, each with an `action` enum over a narrow set of high-value API operations. This is smaller and more model-usable than mirroring the full upstream APIs.
- Wrapper scripts should perform:
  - configuration resolution from secrets plus fixed LAN defaults
  - action validation
  - upstream HTTP execution
  - lookup-result-assisted request construction for add flows
  - typed result/error shaping
- The tools should remain visible in the runtime catalog even if API keys are not yet configured, but the wrappers must fail explicitly with typed auth/configuration errors in that case.
- Server dispatch should reuse the existing `legacy_bash` wrapper model, with explicit command construction in `server-context.cpp`, just like Tavily.
