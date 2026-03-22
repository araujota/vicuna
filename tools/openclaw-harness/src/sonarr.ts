import {
  chooseLookupCandidate,
  optionalBoolean,
  optionalInteger,
  optionalIntegerArray,
  optionalString,
  requireInteger,
  requireString,
  runServarrCli,
  servarrRequestJson,
  successEnvelope,
  type ServarrCliContext,
  type ServarrInvocation,
} from "./servarr.js";

function calendarQuery(payload: ServarrInvocation) {
  return {
    start: optionalString(payload, "start"),
    end: optionalString(payload, "end"),
    unmonitored: optionalBoolean(payload, "include_unmonitored"),
  };
}

async function handleSonarr(context: ServarrCliContext) {
  const { payload, config } = context;
  const action = String(payload.action);

  if (action === "system_status") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/system/status");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/system/status", undefined, data);
  }
  if (action === "queue") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/queue");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/queue", undefined, data);
  }
  if (action === "calendar") {
    const query = calendarQuery(payload);
    const data = await servarrRequestJson(config, "GET", "/api/v3/calendar", query);
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/calendar", query, data);
  }
  if (action === "root_folders") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/rootfolder");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/rootfolder", undefined, data);
  }
  if (action === "quality_profiles") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/qualityprofile");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/qualityprofile", undefined, data);
  }
  if (action === "list_series") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/series");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/series", undefined, data);
  }
  if (action === "lookup_series") {
    const term = requireString(payload, "term", "term is required for lookup_series");
    const query = { term };
    const data = await servarrRequestJson(config, "GET", "/api/v3/series/lookup", query);
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/series/lookup", query, data);
  }
  if (action === "add_series") {
    const term = requireString(payload, "term", "term is required for add_series");
    const rootFolderPath = requireString(payload, "root_folder_path", "root_folder_path is required for add_series");
    const qualityProfileId = requireInteger(payload, "quality_profile_id", "quality_profile_id is required for add_series");
    const lookupResults = await servarrRequestJson(config, "GET", "/api/v3/series/lookup", { term });
    if (!Array.isArray(lookupResults)) {
      throw new Error("Sonarr lookup_series did not return an array");
    }
    const tvdbId = optionalInteger(payload, "tvdb_id");
    const tmdbId = optionalInteger(payload, "tmdb_id");
    const candidate = chooseLookupCandidate<Record<string, unknown>>(lookupResults as Record<string, unknown>[], [
      (entry) => tvdbId !== undefined && entry.tvdbId === tvdbId,
      (entry) => tmdbId !== undefined && entry.tmdbId === tmdbId,
      () => true,
    ]);
    const requestBody: Record<string, unknown> = {
      ...candidate,
      rootFolderPath,
      qualityProfileId,
      monitored: optionalBoolean(payload, "monitored") ?? true,
      seasonFolder: optionalBoolean(payload, "season_folder") ?? true,
      seriesType: optionalString(payload, "series_type") ?? candidate.seriesType ?? "standard",
      monitorNewItems: optionalString(payload, "monitor_new_items") ?? candidate.monitorNewItems ?? "all",
      tags: optionalIntegerArray(payload, "tags") ?? candidate.tags ?? [],
      addOptions: {
        ...(typeof candidate.addOptions === "object" && candidate.addOptions ? (candidate.addOptions as Record<string, unknown>) : {}),
        monitor: optionalString(payload, "monitor") ?? "all",
        searchForMissingEpisodes: optionalBoolean(payload, "search_for_missing_episodes") ?? true,
        searchForCutoffUnmetEpisodes: optionalBoolean(payload, "search_for_cutoff_unmet_episodes") ?? false,
      }
    };
    delete requestBody.id;
    const data = await servarrRequestJson(config, "POST", "/api/v3/series", undefined, requestBody);
    return successEnvelope("sonarr", action, config.baseUrl, "POST", "/api/v3/series", { term }, data);
  }

  throw new Error(`unsupported Sonarr action: ${action}`);
}

void runServarrCli("sonarr", process.argv.slice(2), handleSonarr);
