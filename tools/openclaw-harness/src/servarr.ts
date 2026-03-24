import { Buffer } from "node:buffer";

import {
  DEFAULT_RADARR_BASE_URL,
  DEFAULT_SONARR_BASE_URL,
  loadToolSecrets,
  type OpenClawServarrToolId,
} from "./config.js";

export type ServarrService = OpenClawServarrToolId;

export type ServarrInvocation = {
  action: string;
  [key: string]: unknown;
};

export type ServarrServiceConfig = {
  service: ServarrService;
  baseUrl: string;
  apiKey?: string;
};

export type ServarrResponseEnvelope = {
  service: ServarrService;
  action: string;
  base_url: string;
  ok: boolean;
  request?: {
    method: string;
    path: string;
    query?: Record<string, unknown>;
  };
  data?: unknown;
  error?: {
    kind: string;
    message: string;
    details?: Record<string, unknown>;
  };
};

type JsonRecord = Record<string, unknown>;

export class ServarrToolError extends Error {
  readonly kind: string;
  readonly details?: Record<string, unknown>;

  constructor(kind: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.kind = kind;
    this.details = details;
  }
}

export type ServarrCliContext = {
  payload: ServarrInvocation;
  config: ServarrServiceConfig;
  service: ServarrService;
};

export const FIXED_RADARR_ROOT_FOLDER_PATH = "/movies";
export const FIXED_SONARR_ROOT_FOLDER_PATH = "/tv";
export const FIXED_RADARR_QUALITY_PROFILE_ID = 6;
export const FIXED_RADARR_QUALITY_PROFILE_NAME = "HD - 720p/1080p";
export const FIXED_SONARR_QUALITY_PROFILE_ID = 4;
export const FIXED_SONARR_QUALITY_PROFILE_NAME = "HD-1080p";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function defaultBaseUrlForService(service: ServarrService): string {
  return service === "radarr" ? DEFAULT_RADARR_BASE_URL : DEFAULT_SONARR_BASE_URL;
}

export function resolveServarrConfig(service: ServarrService, secretsPath: string): ServarrServiceConfig {
  const secrets = loadToolSecrets(secretsPath);
  const serviceSecrets = service === "radarr" ? secrets.tools?.radarr : secrets.tools?.sonarr;
  const baseUrl = normalizeBaseUrl(serviceSecrets?.base_url?.trim() || defaultBaseUrlForService(service));
  const apiKey = serviceSecrets?.api_key?.trim();
  return {
    service,
    baseUrl,
    apiKey: apiKey && apiKey.length > 0 ? apiKey : undefined
  };
}

export function parseCliInvocation(argv: string[]): { payload: ServarrInvocation; secretsPath: string } {
  let payloadBase64 = "";
  let secretsPath = "";
  for (const arg of argv) {
    if (arg.startsWith("--payload-base64=")) {
      payloadBase64 = arg.slice("--payload-base64=".length);
      continue;
    }
    if (arg.startsWith("--secrets-path=")) {
      secretsPath = arg.slice("--secrets-path=".length);
      continue;
    }
  }

  if (!payloadBase64) {
    throw new ServarrToolError("missing_payload", "missing required --payload-base64 argument");
  }
  if (!secretsPath) {
    throw new ServarrToolError("missing_secrets_path", "missing required --secrets-path argument");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(payloadBase64, "base64").toString("utf8"));
  } catch (error) {
    throw new ServarrToolError("invalid_payload", `failed to decode tool payload: ${(error as Error).message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new ServarrToolError("invalid_payload", "decoded tool payload must be a JSON object");
  }

  return {
    payload: payload as ServarrInvocation,
    secretsPath
  };
}

export function requireAction(payload: ServarrInvocation): string {
  const action = typeof payload.action === "string" ? payload.action.trim() : "";
  if (!action) {
    throw new ServarrToolError("missing_action", "tool payload requires a non-empty action");
  }
  return action;
}

export function canonicalizeServarrAction(service: ServarrService, action: string): string {
  if (action !== "inspect") {
    return action;
  }
  return service === "radarr" ? "list_movies" : "list_series";
}

export function optionalString(payload: ServarrInvocation, key: string): string | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ServarrToolError("invalid_argument", `${key} must be a string`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function optionalInteger(payload: ServarrInvocation, key: string): number | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ServarrToolError("invalid_argument", `${key} must be an integer`);
  }
  return value > 0 ? value : undefined;
}

export function optionalBoolean(payload: ServarrInvocation, key: string): boolean | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServarrToolError("invalid_argument", `${key} must be a boolean`);
  }
  return value;
}

export function optionalIntegerArray(payload: ServarrInvocation, key: string): number[] | undefined {
  const value = payload[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number" || !Number.isInteger(entry))) {
    throw new ServarrToolError("invalid_argument", `${key} must be an array of integers`);
  }
  return value as number[];
}

export function requireString(payload: ServarrInvocation, key: string, helpText?: string): string {
  const value = optionalString(payload, key);
  if (!value) {
    throw new ServarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function requireInteger(payload: ServarrInvocation, key: string, helpText?: string): number {
  const value = optionalInteger(payload, key);
  if (value === undefined) {
    throw new ServarrToolError("missing_argument", helpText ?? `${key} is required`);
  }
  return value;
}

export function successEnvelope(
  service: ServarrService,
  action: string,
  baseUrl: string,
  method: string,
  path: string,
  query: Record<string, unknown> | undefined,
  data: unknown
): ServarrResponseEnvelope {
  return {
    service,
    action,
    base_url: baseUrl,
    ok: true,
    request: {
      method,
      path,
      ...(query && Object.keys(query).length > 0 ? { query } : {})
    },
    data
  };
}

function asRecord(value: unknown): JsonRecord | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : undefined;
}

function recordString(record: JsonRecord | undefined, key: string): string | undefined {
  const value = record?.[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function recordBoolean(record: JsonRecord | undefined, key: string): boolean | undefined {
  const value = record?.[key];
  return typeof value === "boolean" ? value : undefined;
}

function recordInteger(record: JsonRecord | undefined, key: string): number | undefined {
  const value = record?.[key];
  return typeof value === "number" && Number.isInteger(value) ? value : undefined;
}

function recordNumber(record: JsonRecord | undefined, key: string): number | undefined {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function sortTitles(titles: string[]): string[] {
  return titles
    .map((title) => title.trim())
    .filter((title) => title.length > 0)
    .sort((left, right) => left.localeCompare(right));
}

export function summarizeServarrCommand(data: unknown): unknown {
  const record = asRecord(data);
  if (!record) {
    return data;
  }

  return {
    id: recordInteger(record, "id"),
    name: recordString(record, "name"),
    status: recordString(record, "status"),
    state: recordString(record, "state"),
    message: recordString(record, "message"),
    body: recordString(record, "body"),
  };
}

export function summarizeSonarrSeriesRecord(data: unknown): unknown {
  const record = asRecord(data);
  if (!record) {
    return data;
  }

  const statistics = asRecord(record.statistics);
  return {
    id: recordInteger(record, "id"),
    title: recordString(record, "title"),
    year: recordInteger(record, "year"),
    status: recordString(record, "status"),
    monitored: recordBoolean(record, "monitored") ?? false,
    downloaded: (recordInteger(statistics, "episodeFileCount") ?? 0) > 0,
    tvdb_id: recordInteger(record, "tvdbId"),
    tmdb_id: recordInteger(record, "tmdbId"),
    imdb_id: recordString(record, "imdbId"),
    path: recordString(record, "path"),
  };
}

export function summarizeSonarrSeriesList(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const titles: string[] = [];
  const items = data.flatMap((entry) => {
    const summary = summarizeSonarrSeriesRecord(entry);
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      return [];
    }
    const title = recordString(summary as JsonRecord, "title");
    if (!title) {
      return [];
    }
    return [summary];
  });
  let monitoredSeries = 0;
  let continuingSeries = 0;
  let endedSeries = 0;
  let downloadedSeries = 0;

  for (const entry of data) {
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const title = recordString(record, "title");
    if (title) {
      titles.push(title);
    }
    if (recordBoolean(record, "monitored")) {
      monitoredSeries += 1;
    }
    const status = recordString(record, "status");
    if (status === "continuing") {
      continuingSeries += 1;
    } else if (status === "ended") {
      endedSeries += 1;
    }
    const statistics = asRecord(record.statistics);
    if ((recordInteger(statistics, "episodeFileCount") ?? 0) > 0) {
      downloadedSeries += 1;
    }
  }

  return {
    total_series: data.length,
    monitored_series: monitoredSeries,
    continuing_series: continuingSeries,
    ended_series: endedSeries,
    downloaded_series: downloadedSeries,
    titles: sortTitles(titles),
    items,
  };
}

export function summarizeSonarrLookupResults(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items = data.flatMap((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return [];
    }
    const title = recordString(record, "title");
    if (!title) {
      return [];
    }

    return [{
      title,
      year: recordInteger(record, "year"),
      status: recordString(record, "status"),
      network: recordString(record, "network"),
      tvdb_id: recordInteger(record, "tvdbId"),
      tmdb_id: recordInteger(record, "tmdbId"),
      imdb_id: recordString(record, "imdbId"),
      monitored: recordBoolean(record, "monitored") ?? false,
    }];
  });

  return {
    result_count: items.length,
    titles: sortTitles(items.map((item) => item.title)),
    items,
  };
}

export function summarizeSonarrQueue(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items = data.flatMap((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return [];
    }
    const series = asRecord(record.series);
    const title = recordString(series, "title") ?? recordString(record, "title");
    if (!title) {
      return [];
    }

    return [{
      title,
      status: recordString(record, "status"),
      tracked_download_state: recordString(record, "trackedDownloadState"),
      protocol: recordString(record, "protocol"),
      season_number: recordInteger(record, "seasonNumber"),
      episode_count: Array.isArray(record.episodeIds) ? record.episodeIds.length : undefined,
      sizeleft: recordNumber(record, "sizeleft"),
      timeleft: recordString(record, "timeleft"),
    }];
  });

  return {
    queue_count: items.length,
    titles: sortTitles(items.map((item) => item.title)),
    items,
  };
}

export function summarizeRadarrMovieList(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const titles: string[] = [];
  const items = data.flatMap((entry) => {
    const summary = summarizeRadarrMovieRecord(entry);
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      return [];
    }
    const title = recordString(summary as JsonRecord, "title");
    if (!title) {
      return [];
    }
    return [summary];
  });
  let monitoredMovies = 0;
  let downloadedMovies = 0;
  let releasedMovies = 0;

  for (const entry of data) {
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const title = recordString(record, "title");
    if (title) {
      titles.push(title);
    }
    if (recordBoolean(record, "monitored")) {
      monitoredMovies += 1;
    }
    if (recordBoolean(record, "hasFile") || asRecord(record.movieFile)) {
      downloadedMovies += 1;
    }
    if (recordString(record, "minimumAvailability") === "released") {
      releasedMovies += 1;
    }
  }

  return {
    total_movies: data.length,
    monitored_movies: monitoredMovies,
    downloaded_movies: downloadedMovies,
    released_movies: releasedMovies,
    titles: sortTitles(titles),
    items,
  };
}

export function summarizeRadarrLookupResults(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items = data.flatMap((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return [];
    }
    const title = recordString(record, "title");
    if (!title) {
      return [];
    }

    return [{
      title,
      year: recordInteger(record, "year"),
      status: recordString(record, "status"),
      studio: recordString(record, "studio"),
      tmdb_id: recordInteger(record, "tmdbId"),
      imdb_id: recordString(record, "imdbId"),
      monitored: recordBoolean(record, "monitored") ?? false,
      minimum_availability: recordString(record, "minimumAvailability"),
    }];
  });

  return {
    result_count: items.length,
    titles: sortTitles(items.map((item) => item.title)),
    items,
  };
}

export function summarizeRadarrQueue(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items = data.flatMap((entry) => {
    const record = asRecord(entry);
    if (!record) {
      return [];
    }
    const movie = asRecord(record.movie);
    const title = recordString(movie, "title") ?? recordString(record, "title");
    if (!title) {
      return [];
    }

    return [{
      title,
      status: recordString(record, "status"),
      tracked_download_state: recordString(record, "trackedDownloadState"),
      protocol: recordString(record, "protocol"),
      quality: recordString(asRecord(record.quality), "name"),
      sizeleft: recordNumber(record, "sizeleft"),
      timeleft: recordString(record, "timeleft"),
    }];
  });

  return {
    queue_count: items.length,
    titles: sortTitles(items.map((item) => item.title)),
    items,
  };
}

export function summarizeRadarrMovieRecord(data: unknown): unknown {
  const record = asRecord(data);
  if (!record) {
    return data;
  }

  return {
    id: recordInteger(record, "id"),
    title: recordString(record, "title"),
    year: recordInteger(record, "year"),
    monitored: recordBoolean(record, "monitored") ?? false,
    downloaded: recordBoolean(record, "hasFile") || Boolean(asRecord(record.movieFile)),
    minimum_availability: recordString(record, "minimumAvailability"),
    tmdb_id: recordInteger(record, "tmdbId"),
    imdb_id: recordString(record, "imdbId"),
    path: recordString(record, "path"),
  };
}

export function errorEnvelope(
  service: ServarrService,
  action: string,
  baseUrl: string,
  error: unknown
): ServarrResponseEnvelope {
  if (error instanceof ServarrToolError) {
    return {
      service,
      action,
      base_url: baseUrl,
      ok: false,
      error: {
        kind: error.kind,
        message: error.message,
        ...(error.details ? { details: error.details } : {})
      }
    };
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    service,
    action,
    base_url: baseUrl,
    ok: false,
    error: {
      kind: "unexpected_error",
      message
    }
  };
}

function appendQuery(url: URL, query: Record<string, unknown> | undefined): void {
  if (!query) {
    return;
  }
  for (const [key, rawValue] of Object.entries(query)) {
    if (rawValue === undefined || rawValue === null || rawValue === "") {
      continue;
    }
    if (Array.isArray(rawValue)) {
      for (const entry of rawValue) {
        url.searchParams.append(key, String(entry));
      }
      continue;
    }
    url.searchParams.set(key, String(rawValue));
  }
}

export async function servarrRequestJson(
  config: ServarrServiceConfig,
  method: string,
  path: string,
  query?: Record<string, unknown>,
  body?: unknown
): Promise<unknown> {
  if (!config.apiKey) {
    throw new ServarrToolError(
      "missing_api_key",
      `missing ${config.service} API key in OpenClaw tool secrets`,
      { service: config.service }
    );
  }

  const url = new URL(path, `${config.baseUrl}/`);
  appendQuery(url, query);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: {
        accept: "application/json",
        "content-type": "application/json",
        "X-Api-Key": config.apiKey
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) })
    });
  } catch (error) {
    throw new ServarrToolError(
      "network_error",
      `failed to reach ${config.service}: ${(error as Error).message}`,
      { service: config.service, base_url: config.baseUrl }
    );
  }

  const responseText = await response.text();
  if (!response.ok) {
    const kind = response.status === 401 || response.status === 403 ? "authorization_failed" : "http_error";
    throw new ServarrToolError(
      kind,
      `${config.service} request failed with HTTP ${response.status}`,
      {
        status: response.status,
        body: responseText.slice(0, 400),
        path
      }
    );
  }

  if (!responseText.trim()) {
    return null;
  }

  try {
    return JSON.parse(responseText) as unknown;
  } catch (error) {
    throw new ServarrToolError(
      "invalid_json",
      `${config.service} returned non-JSON content`,
      { body: responseText.slice(0, 400), parse_error: (error as Error).message }
    );
  }
}

export function chooseLookupCandidate<T extends Record<string, unknown>>(
  candidates: T[],
  matchers: Array<(candidate: T) => boolean>
): T {
  for (const matcher of matchers) {
    const match = candidates.find(matcher);
    if (match) {
      return match;
    }
  }
  if (candidates.length === 0) {
    throw new ServarrToolError("lookup_no_match", "lookup returned no candidates");
  }
  return candidates[0];
}

export async function runServarrCli(
  service: ServarrService,
  argv: string[],
  handler: (context: ServarrCliContext) => Promise<ServarrResponseEnvelope>
): Promise<void> {
  let payload: ServarrInvocation = { action: "unknown" };
  let config: ServarrServiceConfig = {
    service,
    baseUrl: defaultBaseUrlForService(service),
    apiKey: undefined
  };
  let action = "unknown";

  try {
    const parsed = parseCliInvocation(argv);
    payload = parsed.payload;
    action = requireAction(payload);
    config = resolveServarrConfig(service, parsed.secretsPath);
    const response = await handler({ payload, config, service });
    process.stdout.write(`${JSON.stringify(response, null, 2)}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify(errorEnvelope(service, action, config.baseUrl, error), null, 2)}\n`);
  }
}
