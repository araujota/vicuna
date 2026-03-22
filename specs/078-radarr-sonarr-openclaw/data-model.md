# Data Model: Radarr and Sonarr OpenClaw Tools

## OpenClaw Tool Secrets

```json
{
  "tools": {
    "radarr": {
      "base_url": "http://10.0.0.218:7878",
      "api_key": "optional-api-key"
    },
    "sonarr": {
      "base_url": "http://10.0.0.218:8989",
      "api_key": "optional-api-key"
    }
  }
}
```

Rules:

- `base_url` defaults to the LAN NAS address for each service if omitted.
- `api_key` is optional at catalog time but required for successful upstream calls.

## Radarr Tool Request

- `action`: enum
  - `system_status`
  - `queue`
  - `calendar`
  - `root_folders`
  - `quality_profiles`
  - `list_movies`
  - `lookup_movie`
  - `add_movie`
- `term`: optional string used for `lookup_movie` and `add_movie`
- `tmdb_id`: optional integer used to disambiguate a lookup result for `add_movie`
- `start`: optional ISO datetime for `calendar`
- `end`: optional ISO datetime for `calendar`
- `include_unmonitored`: optional boolean for `calendar`
- `root_folder_path`: optional string, required for `add_movie`
- `quality_profile_id`: optional integer, required for `add_movie`
- `monitored`: optional boolean for `add_movie`
- `minimum_availability`: optional enum for `add_movie`
- `monitor`: optional enum for `add_movie.addOptions.monitor`
- `search_for_movie`: optional boolean for `add_movie.addOptions.searchForMovie`
- `tags`: optional integer array for `add_movie`

## Sonarr Tool Request

- `action`: enum
  - `system_status`
  - `queue`
  - `calendar`
  - `root_folders`
  - `quality_profiles`
  - `list_series`
  - `lookup_series`
  - `add_series`
- `term`: optional string used for `lookup_series` and `add_series`
- `tvdb_id`: optional integer used to disambiguate a lookup result for `add_series`
- `tmdb_id`: optional integer used to disambiguate a lookup result for `add_series`
- `start`: optional ISO datetime for `calendar`
- `end`: optional ISO datetime for `calendar`
- `include_unmonitored`: optional boolean for `calendar`
- `root_folder_path`: optional string, required for `add_series`
- `quality_profile_id`: optional integer, required for `add_series`
- `monitored`: optional boolean for `add_series`
- `season_folder`: optional boolean for `add_series`
- `series_type`: optional enum for `add_series`
- `monitor_new_items`: optional enum for `add_series`
- `monitor`: optional enum for `add_series.addOptions.monitor`
- `search_for_missing_episodes`: optional boolean for `add_series.addOptions.searchForMissingEpisodes`
- `search_for_cutoff_unmet_episodes`: optional boolean for `add_series.addOptions.searchForCutoffUnmetEpisodes`
- `tags`: optional integer array for `add_series`

## Tool Observation Shape

```json
{
  "service": "radarr",
  "action": "lookup_movie",
  "base_url": "http://10.0.0.218:7878",
  "ok": true,
  "request": {
    "method": "GET",
    "path": "/api/v3/movie/lookup"
  },
  "data": []
}
```

Error shape:

```json
{
  "service": "sonarr",
  "action": "add_series",
  "ok": false,
  "error": {
    "kind": "missing_api_key",
    "message": "missing Sonarr API key in OpenClaw tool secrets"
  }
}
```
