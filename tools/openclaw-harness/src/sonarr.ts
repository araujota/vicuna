import {
  canonicalizeServarrAction,
  chooseLookupCandidate,
  FIXED_SONARR_QUALITY_PROFILE_ID,
  FIXED_SONARR_ROOT_FOLDER_PATH,
  optionalBoolean,
  optionalInteger,
  optionalIntegerArray,
  optionalStringArray,
  optionalString,
  requireString,
  runServarrCli,
  servarrRequestJson,
  ServarrToolError,
  successEnvelope,
  summarizeServarrCommand,
  summarizeServarrQualityProfiles,
  summarizeServarrRootFolders,
  summarizeServarrSystemStatus,
  summarizeSonarrCalendar,
  summarizeSonarrLookupResults,
  summarizeSonarrQueue,
  summarizeSonarrSeriesList,
  summarizeSonarrSeriesRecord,
  type ServarrCliContext,
  type ServarrInvocation,
} from "./servarr.js";

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function normalizeQueryText(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function seriesQueryAliases(record: Record<string, unknown>): string[] {
  const title = typeof record.title === "string" ? record.title.trim() : "";
  const year = typeof record.year === "number" ? String(record.year) : "";
  return [
    title,
    year ? `${title} ${year}` : "",
    year ? `${title} (${year})` : "",
  ].filter((value) => value.length > 0);
}

function formatSeriesLabel(record: Record<string, unknown>): string {
  const title = typeof record.title === "string" ? record.title : "Unknown series";
  const year = typeof record.year === "number" ? ` (${record.year})` : "";
  const id = typeof record.id === "number" ? ` [id=${record.id}]` : "";
  return `${title}${year}${id}`;
}

function queryMatchesSeries(record: Record<string, unknown>, term: string): boolean {
  const normalizedTerm = normalizeQueryText(term);
  if (!normalizedTerm) {
    return false;
  }

  const aliases = seriesQueryAliases(record).map(normalizeQueryText);
  if (aliases.includes(normalizedTerm)) {
    return true;
  }

  const termTokens = normalizedTerm.split(/\s+/).filter((token) => token.length > 0);
  if (termTokens.length === 0) {
    return false;
  }

  return aliases.some((alias) => termTokens.every((token) => alias.includes(token)));
}

function findExistingSeries(
  existingSeries: unknown,
  candidate: Record<string, unknown>
): Record<string, unknown> | undefined {
  if (!Array.isArray(existingSeries)) {
    return undefined;
  }

  const candidateTvdbId = typeof candidate.tvdbId === "number" ? candidate.tvdbId : undefined;
  const candidateTmdbId = typeof candidate.tmdbId === "number" ? candidate.tmdbId : undefined;
  const candidateTitle = typeof candidate.title === "string" ? candidate.title.trim().toLowerCase() : "";
  const candidateYear = typeof candidate.year === "number" ? candidate.year : undefined;

  return existingSeries.find((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return false;
    }
    if (candidateTvdbId !== undefined && record.tvdbId === candidateTvdbId) {
      return true;
    }
    if (candidateTmdbId !== undefined && record.tmdbId === candidateTmdbId) {
      return true;
    }
    if (!candidateTitle) {
      return false;
    }
    const title = typeof record.title === "string" ? record.title.trim().toLowerCase() : "";
    const year = typeof record.year === "number" ? record.year : undefined;
    return title === candidateTitle && (candidateYear === undefined || year === candidateYear);
  }) as Record<string, unknown> | undefined;
}

async function triggerSeriesSearch(
  context: ServarrCliContext,
  seriesId: number
) {
  const { config } = context;
  const requestBody = {
    name: "SeriesSearch",
    seriesId,
  };
  return await servarrRequestJson(config, "POST", "/api/v3/command", undefined, requestBody);
}

function calendarQuery(payload: ServarrInvocation) {
  return {
    start: optionalString(payload, "start"),
    end: optionalString(payload, "end"),
    unmonitored: optionalBoolean(payload, "include_unmonitored"),
  };
}

function resolveDeleteSeriesTarget(
  existingSeries: unknown,
  payload: ServarrInvocation
): Record<string, unknown> {
  if (!Array.isArray(existingSeries)) {
    throw new ServarrToolError("invalid_response", "Sonarr series list did not return an array");
  }

  const candidates = existingSeries
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => entry !== undefined);
  const seriesId = optionalInteger(payload, "series_id");
  if (seriesId !== undefined) {
    const directMatch = candidates.find((entry) => entry.id === seriesId);
    if (!directMatch) {
      throw new ServarrToolError("lookup_no_match", `no tracked Sonarr series matched id=${seriesId}`);
    }
    return directMatch;
  }

  const tvdbId = optionalInteger(payload, "tvdb_id");
  if (tvdbId !== undefined) {
    const tvdbMatches = candidates.filter((entry) => entry.tvdbId === tvdbId);
    if (tvdbMatches.length === 1) {
      return tvdbMatches[0];
    }
    if (tvdbMatches.length > 1) {
      throw new ServarrToolError("ambiguous_match", "multiple tracked Sonarr series matched the requested tvdb_id", {
        tvdb_id: tvdbId,
        matches: tvdbMatches.slice(0, 10).map(formatSeriesLabel),
      });
    }
  }

  const tmdbId = optionalInteger(payload, "tmdb_id");
  if (tmdbId !== undefined) {
    const tmdbMatches = candidates.filter((entry) => entry.tmdbId === tmdbId);
    if (tmdbMatches.length === 1) {
      return tmdbMatches[0];
    }
    if (tmdbMatches.length > 1) {
      throw new ServarrToolError("ambiguous_match", "multiple tracked Sonarr series matched the requested tmdb_id", {
        tmdb_id: tmdbId,
        matches: tmdbMatches.slice(0, 10).map(formatSeriesLabel),
      });
    }
  }

  const term = requireString(payload, "term", "term or series_id is required for delete_series");
  const matches = candidates.filter((entry) => queryMatchesSeries(entry, term));
  if (matches.length === 1) {
    return matches[0];
  }
  if (matches.length > 1) {
    throw new ServarrToolError("ambiguous_match", "multiple tracked Sonarr series matched the requested title", {
      term,
      matches: matches.slice(0, 10).map(formatSeriesLabel),
    });
  }

  throw new ServarrToolError("lookup_no_match", `no tracked Sonarr series matched '${term}'`);
}

function collectDeleteSeriesTargets(
  existingSeries: unknown,
  payload: ServarrInvocation
): Record<string, unknown>[] {
  const targets: Record<string, unknown>[] = [];
  const seenIds = new Set<number>();

  const pushTarget = (target: Record<string, unknown>) => {
    const seriesId = typeof target.id === "number" ? target.id : undefined;
    if (seriesId === undefined || seenIds.has(seriesId)) {
      return;
    }
    seenIds.add(seriesId);
    targets.push(target);
  };

  const seriesIds = optionalIntegerArray(payload, "series_ids") ?? [];
  for (const seriesId of seriesIds) {
    pushTarget(resolveDeleteSeriesTarget(existingSeries, { action: "delete_series", series_id: seriesId }));
  }

  if (
    optionalInteger(payload, "series_id") !== undefined ||
    optionalInteger(payload, "tvdb_id") !== undefined ||
    optionalInteger(payload, "tmdb_id") !== undefined
  ) {
    pushTarget(resolveDeleteSeriesTarget(existingSeries, payload));
  }

  const terms = optionalStringArray(payload, "terms") ?? [];
  for (const term of terms) {
    pushTarget(resolveDeleteSeriesTarget(existingSeries, { action: "delete_series", term }));
  }

  if (optionalString(payload, "term")) {
    pushTarget(resolveDeleteSeriesTarget(existingSeries, payload));
  }

  if (targets.length === 0) {
    throw new ServarrToolError(
      "missing_argument",
      "delete_series requires series_id, series_ids, term, terms, tvdb_id, or tmdb_id"
    );
  }

  return targets;
}

export async function handleSonarr(context: ServarrCliContext) {
  const { payload, config } = context;
  const action = String(payload.action);
  const resolvedAction = canonicalizeServarrAction("sonarr", action);

  if (resolvedAction === "list_downloaded_series" || resolvedAction === "list_series") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/series");
    return successEnvelope("sonarr", action, config.baseUrl, "GET", "/api/v3/series", undefined, summarizeSonarrSeriesList(data));
  }
  if (resolvedAction === "download_series") {
    const term = requireString(payload, "term", "term is required for download_series");
    const rootFolderPath = FIXED_SONARR_ROOT_FOLDER_PATH;
    const qualityProfileId = FIXED_SONARR_QUALITY_PROFILE_ID;
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
    const existingSeries = await servarrRequestJson(config, "GET", "/api/v3/series");
    const existing = findExistingSeries(existingSeries, candidate);
    if (existing && typeof existing.id === "number") {
      const command = await triggerSeriesSearch(context, existing.id);
      return successEnvelope("sonarr", action, config.baseUrl, "POST", "/api/v3/command", { term }, {
        mode: "existing_series_search_started",
        message: `Started a Sonarr search for '${typeof existing.title === "string" ? existing.title : term}'.`,
        series: summarizeSonarrSeriesRecord(existing),
        search_command: summarizeServarrCommand(command),
      });
    }
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
    return successEnvelope("sonarr", action, config.baseUrl, "POST", "/api/v3/series", { term }, {
      mode: "new_series_added_and_search_requested",
      message: `Added '${term}' to Sonarr and requested an immediate search.`,
      series: summarizeSonarrSeriesRecord(data),
    });
  }
  if (resolvedAction === "delete_series") {
    const existingSeries = await servarrRequestJson(config, "GET", "/api/v3/series");
    const query = {
      deleteFiles: optionalBoolean(payload, "delete_files") ?? true,
      addImportListExclusion: optionalBoolean(payload, "add_import_list_exclusion") ?? false,
    };
    const targets = collectDeleteSeriesTargets(existingSeries, payload);
    const deletedSeries: unknown[] = [];
    for (const target of targets) {
      const seriesId = typeof target.id === "number" ? target.id : undefined;
      if (seriesId === undefined) {
        throw new ServarrToolError("lookup_no_match", "matched Sonarr series did not include an id");
      }
      await servarrRequestJson(config, "DELETE", `/api/v3/series/${seriesId}`, query);
      deletedSeries.push(summarizeSonarrSeriesRecord(target));
    }
    return successEnvelope("sonarr", action, config.baseUrl, "DELETE", "/api/v3/series/:id", query, {
      mode: "series_deleted",
      message:
        deletedSeries.length === 1
          ? "Deleted 1 series from Sonarr."
          : `Deleted ${deletedSeries.length} series from Sonarr.`,
      deleted_count: deletedSeries.length,
      deleted_series: deletedSeries,
    });
  }

  throw new Error(`unsupported Sonarr action: ${action}`);
}

void runServarrCli("sonarr", process.argv.slice(2), handleSonarr);
