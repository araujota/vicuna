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

async function handleRadarr(context: ServarrCliContext) {
  const { payload, config } = context;
  const action = String(payload.action);

  if (action === "system_status") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/system/status");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/system/status", undefined, data);
  }
  if (action === "queue") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/queue");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/queue", undefined, data);
  }
  if (action === "calendar") {
    const query = calendarQuery(payload);
    const data = await servarrRequestJson(config, "GET", "/api/v3/calendar", query);
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/calendar", query, data);
  }
  if (action === "root_folders") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/rootfolder");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/rootfolder", undefined, data);
  }
  if (action === "quality_profiles") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/qualityprofile");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/qualityprofile", undefined, data);
  }
  if (action === "list_movies") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/movie");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/movie", undefined, data);
  }
  if (action === "lookup_movie") {
    const term = requireString(payload, "term", "term is required for lookup_movie");
    const query = { term };
    const data = await servarrRequestJson(config, "GET", "/api/v3/movie/lookup", query);
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/movie/lookup", query, data);
  }
  if (action === "add_movie") {
    const term = requireString(payload, "term", "term is required for add_movie");
    const rootFolderPath = requireString(payload, "root_folder_path", "root_folder_path is required for add_movie");
    const qualityProfileId = requireInteger(payload, "quality_profile_id", "quality_profile_id is required for add_movie");
    const lookupResults = await servarrRequestJson(config, "GET", "/api/v3/movie/lookup", { term });
    if (!Array.isArray(lookupResults)) {
      throw new Error("Radarr lookup_movie did not return an array");
    }
    const tmdbId = optionalInteger(payload, "tmdb_id");
    const candidate = chooseLookupCandidate<Record<string, unknown>>(lookupResults as Record<string, unknown>[], [
      (entry) => tmdbId !== undefined && entry.tmdbId === tmdbId,
      () => true,
    ]);
    const requestBody: Record<string, unknown> = {
      ...candidate,
      rootFolderPath,
      qualityProfileId,
      monitored: optionalBoolean(payload, "monitored") ?? true,
      minimumAvailability: optionalString(payload, "minimum_availability") ?? candidate.minimumAvailability ?? "released",
      tags: optionalIntegerArray(payload, "tags") ?? candidate.tags ?? [],
      addOptions: {
        ...(typeof candidate.addOptions === "object" && candidate.addOptions ? (candidate.addOptions as Record<string, unknown>) : {}),
        monitor: optionalString(payload, "monitor") ?? "movieOnly",
        searchForMovie: optionalBoolean(payload, "search_for_movie") ?? false,
        addMethod: "manual"
      }
    };
    delete requestBody.id;
    const data = await servarrRequestJson(config, "POST", "/api/v3/movie", undefined, requestBody);
    return successEnvelope("radarr", action, config.baseUrl, "POST", "/api/v3/movie", { term }, data);
  }

  throw new Error(`unsupported Radarr action: ${action}`);
}

void runServarrCli("radarr", process.argv.slice(2), handleRadarr);
