import {
  canonicalizeServarrAction,
  chooseLookupCandidate,
  FIXED_RADARR_QUALITY_PROFILE_ID,
  FIXED_RADARR_ROOT_FOLDER_PATH,
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
  summarizeRadarrMovieRecord,
  summarizeRadarrLookupResults,
  summarizeRadarrMovieList,
  summarizeRadarrQueue,
  summarizeRadarrCalendar,
  summarizeServarrCommand,
  summarizeServarrQualityProfiles,
  summarizeServarrRootFolders,
  summarizeServarrSystemStatus,
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

function movieQueryAliases(record: Record<string, unknown>): string[] {
  const title = typeof record.title === "string" ? record.title.trim() : "";
  const year = typeof record.year === "number" ? String(record.year) : "";
  return [
    title,
    year ? `${title} ${year}` : "",
    year ? `${title} (${year})` : "",
  ].filter((value) => value.length > 0);
}

function formatMovieLabel(record: Record<string, unknown>): string {
  const title = typeof record.title === "string" ? record.title : "Unknown movie";
  const year = typeof record.year === "number" ? ` (${record.year})` : "";
  const id = typeof record.id === "number" ? ` [id=${record.id}]` : "";
  return `${title}${year}${id}`;
}

function queryMatchesMovie(record: Record<string, unknown>, term: string): boolean {
  const normalizedTerm = normalizeQueryText(term);
  if (!normalizedTerm) {
    return false;
  }

  const aliases = movieQueryAliases(record).map(normalizeQueryText);
  if (aliases.includes(normalizedTerm)) {
    return true;
  }

  const termTokens = normalizedTerm.split(/\s+/).filter((token) => token.length > 0);
  if (termTokens.length === 0) {
    return false;
  }

  return aliases.some((alias) => termTokens.every((token) => alias.includes(token)));
}

function findExistingMovie(
  existingMovies: unknown,
  candidate: Record<string, unknown>
): Record<string, unknown> | undefined {
  if (!Array.isArray(existingMovies)) {
    return undefined;
  }

  const candidateTmdbId = typeof candidate.tmdbId === "number" ? candidate.tmdbId : undefined;
  const candidateImdbId = typeof candidate.imdbId === "string" ? candidate.imdbId.trim() : "";
  const candidateTitle = typeof candidate.title === "string" ? candidate.title.trim().toLowerCase() : "";
  const candidateYear = typeof candidate.year === "number" ? candidate.year : undefined;

  return existingMovies.find((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return false;
    }
    if (candidateTmdbId !== undefined && record.tmdbId === candidateTmdbId) {
      return true;
    }
    if (candidateImdbId && record.imdbId === candidateImdbId) {
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

async function triggerMovieSearch(
  context: ServarrCliContext,
  movieId: number
) {
  const { config } = context;
  const requestBody = {
    name: "MoviesSearch",
    movieIds: [movieId],
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

function resolveDeleteMovieTarget(
  existingMovies: unknown,
  payload: ServarrInvocation
): Record<string, unknown> {
  if (!Array.isArray(existingMovies)) {
    throw new ServarrToolError("invalid_response", "Radarr movie list did not return an array");
  }

  const candidates = existingMovies
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => entry !== undefined);
  const movieId = optionalInteger(payload, "movie_id");
  if (movieId !== undefined) {
    const directMatch = candidates.find((entry) => entry.id === movieId);
    if (!directMatch) {
      throw new ServarrToolError("lookup_no_match", `no tracked Radarr movie matched id=${movieId}`);
    }
    return directMatch;
  }

  const tmdbId = optionalInteger(payload, "tmdb_id");
  if (tmdbId !== undefined) {
    const tmdbMatches = candidates.filter((entry) => entry.tmdbId === tmdbId);
    if (tmdbMatches.length === 1) {
      return tmdbMatches[0];
    }
    if (tmdbMatches.length > 1) {
      throw new ServarrToolError("ambiguous_match", "multiple tracked Radarr movies matched the requested tmdb_id", {
        tmdb_id: tmdbId,
        matches: tmdbMatches.slice(0, 10).map(formatMovieLabel),
      });
    }
  }

  const term = requireString(payload, "term", "term or movie_id is required for delete_movie");
  const matches = candidates.filter((entry) => queryMatchesMovie(entry, term));
  if (matches.length === 1) {
    return matches[0];
  }
  if (matches.length > 1) {
    throw new ServarrToolError("ambiguous_match", "multiple tracked Radarr movies matched the requested title", {
      term,
      matches: matches.slice(0, 10).map(formatMovieLabel),
    });
  }

  throw new ServarrToolError("lookup_no_match", `no tracked Radarr movie matched '${term}'`);
}

function collectDeleteMovieTargets(
  existingMovies: unknown,
  payload: ServarrInvocation
): Record<string, unknown>[] {
  const targets: Record<string, unknown>[] = [];
  const seenIds = new Set<number>();

  const pushTarget = (target: Record<string, unknown>) => {
    const movieId = typeof target.id === "number" ? target.id : undefined;
    if (movieId === undefined || seenIds.has(movieId)) {
      return;
    }
    seenIds.add(movieId);
    targets.push(target);
  };

  const movieIds = optionalIntegerArray(payload, "movie_ids") ?? [];
  for (const movieId of movieIds) {
    pushTarget(resolveDeleteMovieTarget(existingMovies, { action: "delete_movies", movie_id: movieId }));
  }

  if (optionalInteger(payload, "movie_id") !== undefined || optionalInteger(payload, "tmdb_id") !== undefined) {
    pushTarget(resolveDeleteMovieTarget(existingMovies, payload));
  }

  const terms = optionalStringArray(payload, "terms") ?? [];
  for (const term of terms) {
    pushTarget(resolveDeleteMovieTarget(existingMovies, { action: "delete_movies", term }));
  }

  if (optionalString(payload, "term")) {
    pushTarget(resolveDeleteMovieTarget(existingMovies, payload));
  }

  if (targets.length === 0) {
    throw new ServarrToolError(
      "missing_argument",
      "delete_movies requires movie_id, movie_ids, term, terms, or tmdb_id"
    );
  }

  return targets;
}

export async function handleRadarr(context: ServarrCliContext) {
  const { payload, config } = context;
  const action = String(payload.action);
  const resolvedAction = canonicalizeServarrAction("radarr", action);

  if (resolvedAction === "list_downloaded_movies" || resolvedAction === "list_movies") {
    const data = await servarrRequestJson(config, "GET", "/api/v3/movie");
    return successEnvelope("radarr", action, config.baseUrl, "GET", "/api/v3/movie", undefined, summarizeRadarrMovieList(data));
  }
  if (resolvedAction === "download_movie") {
    const term = requireString(payload, "term", "term is required for download_movie");
    const rootFolderPath = FIXED_RADARR_ROOT_FOLDER_PATH;
    const qualityProfileId = FIXED_RADARR_QUALITY_PROFILE_ID;
    const lookupResults = await servarrRequestJson(config, "GET", "/api/v3/movie/lookup", { term });
    if (!Array.isArray(lookupResults)) {
      throw new Error("Radarr lookup_movie did not return an array");
    }
    const tmdbId = optionalInteger(payload, "tmdb_id");
    const candidate = chooseLookupCandidate<Record<string, unknown>>(lookupResults as Record<string, unknown>[], [
      (entry) => tmdbId !== undefined && entry.tmdbId === tmdbId,
      () => true,
    ]);
    const existingMovies = await servarrRequestJson(config, "GET", "/api/v3/movie");
    const existingMovie = findExistingMovie(existingMovies, candidate);
    if (existingMovie && typeof existingMovie.id === "number") {
      const command = await triggerMovieSearch(context, existingMovie.id);
      return successEnvelope("radarr", action, config.baseUrl, "POST", "/api/v3/command", { term }, {
        mode: "existing_movie_search_started",
        message: `Started a Radarr search for '${typeof existingMovie.title === "string" ? existingMovie.title : term}'.`,
        movie: summarizeRadarrMovieRecord(existingMovie),
        search_command: summarizeServarrCommand(command),
      });
    }
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
        searchForMovie: optionalBoolean(payload, "search_for_movie") ?? true,
        addMethod: "manual"
      }
    };
    delete requestBody.id;
    const data = await servarrRequestJson(config, "POST", "/api/v3/movie", undefined, requestBody);
    return successEnvelope("radarr", action, config.baseUrl, "POST", "/api/v3/movie", { term }, {
      mode: "new_movie_added_and_search_requested",
      message: `Added '${term}' to Radarr and requested an immediate search.`,
      movie: summarizeRadarrMovieRecord(data),
    });
  }
  if (resolvedAction === "delete_movie" || resolvedAction === "delete_movies") {
    const existingMovies = await servarrRequestJson(config, "GET", "/api/v3/movie");
    const query = {
      deleteFiles: optionalBoolean(payload, "delete_files") ?? true,
      addImportExclusion: optionalBoolean(payload, "add_import_exclusion") ?? false,
    };
    const targets = collectDeleteMovieTargets(existingMovies, payload);
    const deletedMovies: unknown[] = [];
    for (const target of targets) {
      const movieId = typeof target.id === "number" ? target.id : undefined;
      if (movieId === undefined) {
        throw new ServarrToolError("lookup_no_match", "matched Radarr movie did not include an id");
      }
      await servarrRequestJson(config, "DELETE", `/api/v3/movie/${movieId}`, query);
      deletedMovies.push(summarizeRadarrMovieRecord(target));
    }
    return successEnvelope("radarr", action, config.baseUrl, "DELETE", "/api/v3/movie/:id", query, {
      mode: "movies_deleted",
      message:
        deletedMovies.length === 1
          ? "Deleted 1 movie from Radarr."
          : `Deleted ${deletedMovies.length} movies from Radarr.`,
      deleted_count: deletedMovies.length,
      deleted_movies: deletedMovies,
    });
  }

  throw new Error(`unsupported Radarr action: ${action}`);
}

void runServarrCli("radarr", process.argv.slice(2), handleRadarr);
