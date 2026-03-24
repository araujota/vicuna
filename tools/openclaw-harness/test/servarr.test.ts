import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_RADARR_BASE_URL,
  DEFAULT_SONARR_BASE_URL,
} from "../src/config.js";
import {
  canonicalizeServarrAction,
  errorEnvelope,
  FIXED_RADARR_QUALITY_PROFILE_ID,
  FIXED_RADARR_ROOT_FOLDER_PATH,
  FIXED_SONARR_QUALITY_PROFILE_ID,
  FIXED_SONARR_ROOT_FOLDER_PATH,
  parseCliInvocation,
  resolveServarrConfig,
  servarrRequestJson,
  ServarrToolError,
  summarizeRadarrMovieList,
  summarizeSonarrSeriesList,
} from "../src/servarr.js";
import { handleRadarr } from "../src/radarr.js";
import { handleSonarr } from "../src/sonarr.js";

test("resolveServarrConfig falls back to LAN NAS defaults", () => {
  const originalFetch = globalThis.fetch;
  assert.equal(resolveServarrConfig("radarr", "/tmp/does-not-exist.json").baseUrl, DEFAULT_RADARR_BASE_URL);
  assert.equal(resolveServarrConfig("sonarr", "/tmp/does-not-exist.json").baseUrl, DEFAULT_SONARR_BASE_URL);
  globalThis.fetch = originalFetch;
});

test("parseCliInvocation decodes the base64 tool payload", () => {
  const encoded = Buffer.from(JSON.stringify({ action: "lookup_movie", term: "Alien" }), "utf8").toString("base64");
  const parsed = parseCliInvocation([`--payload-base64=${encoded}`, "--secrets-path=/tmp/secrets.json"]);
  assert.equal(parsed.secretsPath, "/tmp/secrets.json");
  assert.equal(parsed.payload.action, "lookup_movie");
  assert.equal(parsed.payload.term, "Alien");
});

test("servarrRequestJson returns a typed missing-api-key error before fetch", async () => {
  await assert.rejects(
    servarrRequestJson(
      {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: undefined
      },
      "GET",
      "/api/v3/system/status"
    ),
    (error: unknown) => error instanceof ServarrToolError && error.kind === "missing_api_key"
  );
});

test("errorEnvelope preserves typed Servarr errors", () => {
  const envelope = errorEnvelope(
    "sonarr",
    "add_series",
    DEFAULT_SONARR_BASE_URL,
    new ServarrToolError("authorization_failed", "Sonarr request failed", { status: 401 })
  );

  assert.equal(envelope.ok, false);
  assert.equal(envelope.error?.kind, "authorization_failed");
  assert.equal(envelope.error?.details?.status, 401);
});

test("canonicalizeServarrAction maps inspect to the service listing action", () => {
  assert.equal(canonicalizeServarrAction("radarr", "inspect"), "list_movies");
  assert.equal(canonicalizeServarrAction("sonarr", "inspect"), "list_series");
  assert.equal(canonicalizeServarrAction("sonarr", "lookup_series"), "lookup_series");
});

test("handleRadarr ignores caller-supplied path and quality profile for download_movie", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/movie/lookup")) {
      return new Response(
        JSON.stringify([
          {
            title: "Alien",
            tmdbId: 348,
            minimumAvailability: "released"
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    return new Response(JSON.stringify({
      id: 100,
      title: "Alien",
      year: 1979,
      monitored: true,
      hasFile: false,
      minimumAvailability: "released",
      tmdbId: 348,
      imdbId: "tt0078748",
      path: "/movies/Alien (1979)"
    }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "download_movie",
        term: "Alien",
        root_folder_path: "C:/Movies",
        quality_profile_id: 7
      }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_movie_add",
      movie: {
        id: 100,
        title: "Alien",
        year: 1979,
        monitored: true,
        downloaded: false,
        minimum_availability: "released",
        tmdb_id: 348,
        imdb_id: "tt0078748",
        path: "/movies/Alien (1979)",
      }
    });
    const postBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.rootFolderPath, FIXED_RADARR_ROOT_FOLDER_PATH);
    assert.equal(postBody.qualityProfileId, FIXED_RADARR_QUALITY_PROFILE_ID);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleRadarr triggers MoviesSearch for an already-tracked movie", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/movie/lookup")) {
      return new Response(
        JSON.stringify([
          {
            title: "Alien",
            tmdbId: 348,
            year: 1979,
            minimumAvailability: "released"
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    if (String(input).endsWith("/api/v3/movie")) {
      return new Response(
        JSON.stringify([
          {
            id: 55,
            title: "Alien",
            tmdbId: 348,
            year: 1979,
            monitored: true
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response(JSON.stringify({ id: 99, name: "MoviesSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "download_movie",
        term: "Alien",
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/command");
    assert.deepEqual(response.data, {
      mode: "existing_movie_search",
      existing_movie: {
        id: 55,
        title: "Alien",
        year: 1979,
        monitored: true,
        downloaded: false,
        minimum_availability: undefined,
        tmdb_id: 348,
        imdb_id: undefined,
        path: undefined,
      },
      command: {
        id: 99,
        name: "MoviesSearch",
        status: undefined,
        state: undefined,
        message: undefined,
        body: undefined,
      }
    });
    const postBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.name, "MoviesSearch");
    assert.deepEqual(postBody.movieIds, [55]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleRadarr deletes one tracked movie and removes files by default", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({ url, method });
    if (url.endsWith("/api/v3/movie")) {
      return new Response(
        JSON.stringify([
          {
            id: 55,
            title: "Alien",
            tmdbId: 348,
            year: 1979,
            monitored: true
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response("{}", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_movie",
        term: "Alien (1979)"
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/movie/55");
    assert.deepEqual(response.data, {
      mode: "tracked_movie_delete",
      deleted_movie: {
        id: 55,
        title: "Alien",
        year: 1979,
        monitored: true,
        downloaded: false,
        minimum_availability: undefined,
        tmdb_id: 348,
        imdb_id: undefined,
        path: undefined,
      }
    });
    assert.deepEqual(response.request?.query, {
      deleteFiles: true,
      addImportExclusion: false
    });
    assert.equal(fetchCalls.length, 2);
    assert.equal(fetchCalls[0].method, "GET");
    assert.equal(fetchCalls[1].method, "DELETE");
    assert.match(fetchCalls[1].url, /\/api\/v3\/movie\/55\?deleteFiles=true&addImportExclusion=false$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleRadarr ignores placeholder ids and falls back to title matching for delete_movie", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET" && url.endsWith("/api/v3/movie")) {
      return new Response(
        JSON.stringify([
          {
            id: 55,
            title: "Alien",
            tmdbId: 348,
            year: 1979,
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response("{}", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: {
        service: "radarr",
        baseUrl: DEFAULT_RADARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_movie",
        movie_id: 0,
        tmdb_id: 0,
        term: "Alien",
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/movie/55");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleRadarr rejects ambiguous tracked movie deletes", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify([
        {
          id: 55,
          title: "Alien",
          year: 1979
        },
        {
          id: 56,
          title: "Alien",
          year: 2003
        }
      ]),
      { status: 200, headers: { "content-type": "application/json" } }
    )) as typeof fetch;

  try {
    await assert.rejects(
      handleRadarr({
        service: "radarr",
        config: {
          service: "radarr",
          baseUrl: DEFAULT_RADARR_BASE_URL,
          apiKey: "test-key"
        },
        payload: {
          action: "delete_movie",
          term: "Alien"
        }
      }),
      (error: unknown) =>
        error instanceof ServarrToolError &&
        error.kind === "ambiguous_match" &&
        Array.isArray(error.details?.matches) &&
        error.details.matches.length === 2
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr ignores caller-supplied path and quality profile for download_series", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/series/lookup")) {
      return new Response(
        JSON.stringify([
          {
            title: "Severance",
            tvdbId: 366524,
            seriesType: "standard",
            monitorNewItems: "all"
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    return new Response(JSON.stringify({
      id: 200,
      title: "Severance",
      year: 2022,
      monitored: true,
      tvdbId: 366524,
      tmdbId: 95396,
      imdbId: "tt11280740",
      path: "/tv/Severance",
      statistics: {
        episodeFileCount: 0
      }
    }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: {
        service: "sonarr",
        baseUrl: DEFAULT_SONARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "download_series",
        term: "Severance",
        root_folder_path: "C:/TV",
        quality_profile_id: 3
      }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_series_add",
      series: {
        id: 200,
        title: "Severance",
        year: 2022,
        status: undefined,
        monitored: true,
        downloaded: false,
        tvdb_id: 366524,
        tmdb_id: 95396,
        imdb_id: "tt11280740",
        path: "/tv/Severance",
      }
    });
    const postBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.rootFolderPath, FIXED_SONARR_ROOT_FOLDER_PATH);
    assert.equal(postBody.qualityProfileId, FIXED_SONARR_QUALITY_PROFILE_ID);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr triggers SeriesSearch for an already-tracked series", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/series/lookup")) {
      return new Response(
        JSON.stringify([
          {
            title: "Severance",
            tvdbId: 366524,
            tmdbId: 95396,
            year: 2022,
            seriesType: "standard",
            monitorNewItems: "all"
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    if (String(input).endsWith("/api/v3/series")) {
      return new Response(
        JSON.stringify([
          {
            id: 77,
            title: "Severance",
            tvdbId: 366524,
            tmdbId: 95396,
            monitored: true
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response(JSON.stringify({ id: 123, name: "SeriesSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: {
        service: "sonarr",
        baseUrl: DEFAULT_SONARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "download_series",
        term: "Severance",
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/command");
    assert.deepEqual(response.data, {
      mode: "existing_series_search",
      existing_series: {
        id: 77,
        title: "Severance",
        year: undefined,
        status: undefined,
        monitored: true,
        downloaded: false,
        tvdb_id: 366524,
        tmdb_id: 95396,
        imdb_id: undefined,
        path: undefined,
      },
      command: {
        id: 123,
        name: "SeriesSearch",
        status: undefined,
        state: undefined,
        message: undefined,
        body: undefined,
      }
    });
    const postBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.name, "SeriesSearch");
    assert.equal(postBody.seriesId, 77);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr deletes one tracked series and removes files by default", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({ url, method });
    if (url.endsWith("/api/v3/series")) {
      return new Response(
        JSON.stringify([
          {
            id: 77,
            title: "Severance",
            tvdbId: 366524,
            tmdbId: 95396,
            year: 2022,
            monitored: true
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response("{}", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: {
        service: "sonarr",
        baseUrl: DEFAULT_SONARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_series",
        term: "Severance (2022)"
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/series/77");
    assert.deepEqual(response.data, {
      mode: "tracked_series_delete",
      deleted_series: {
        id: 77,
        title: "Severance",
        year: 2022,
        status: undefined,
        monitored: true,
        downloaded: false,
        tvdb_id: 366524,
        tmdb_id: 95396,
        imdb_id: undefined,
        path: undefined,
      }
    });
    assert.deepEqual(response.request?.query, {
      deleteFiles: true,
      addImportListExclusion: false
    });
    assert.equal(fetchCalls.length, 2);
    assert.equal(fetchCalls[0].method, "GET");
    assert.equal(fetchCalls[1].method, "DELETE");
    assert.match(fetchCalls[1].url, /\/api\/v3\/series\/77\?deleteFiles=true&addImportListExclusion=false$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr ignores placeholder ids and falls back to title matching for delete_series", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET" && url.endsWith("/api/v3/series")) {
      return new Response(
        JSON.stringify([
          {
            id: 22,
            title: "The Simpsons",
            year: 1989,
            tvdbId: 71663,
            tmdbId: 456,
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response("{}", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: {
        service: "sonarr",
        baseUrl: DEFAULT_SONARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_series",
        series_id: 0,
        tvdb_id: 0,
        tmdb_id: 0,
        term: "The Simpsons",
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v3/series/22");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr delete responses stay compact even for very large tracked series records", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET" && url.endsWith("/api/v3/series")) {
      return new Response(
        JSON.stringify([
          {
            id: 22,
            title: "The Simpsons",
            year: 1989,
            tvdbId: 71663,
            tmdbId: 456,
            imdbId: "tt0096697",
            monitored: true,
            path: "/tv/The Simpsons",
            statistics: {
              episodeFileCount: 757,
            },
            alternateTitles: Array.from({ length: 100 }, (_, index) => ({ title: `Alt ${index}` })),
            seasons: Array.from({ length: 40 }, (_, index) => ({ seasonNumber: index + 1 })),
            overview: "x".repeat(5000),
          }
        ]),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }
    return new Response("{}", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: {
        service: "sonarr",
        baseUrl: DEFAULT_SONARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_series",
        term: "The Simpsons",
      }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "tracked_series_delete",
      deleted_series: {
        id: 22,
        title: "The Simpsons",
        year: 1989,
        status: undefined,
        monitored: true,
        downloaded: true,
        tvdb_id: 71663,
        tmdb_id: 456,
        imdb_id: "tt0096697",
        path: "/tv/The Simpsons",
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr rejects ambiguous tracked series deletes", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify([
        {
          id: 77,
          title: "Shogun",
          year: 1980
        },
        {
          id: 78,
          title: "Shogun",
          year: 2024
        }
      ]),
      { status: 200, headers: { "content-type": "application/json" } }
    )) as typeof fetch;

  try {
    await assert.rejects(
      handleSonarr({
        service: "sonarr",
        config: {
          service: "sonarr",
          baseUrl: DEFAULT_SONARR_BASE_URL,
          apiKey: "test-key"
        },
        payload: {
          action: "delete_series",
          term: "Shogun"
        }
      }),
      (error: unknown) =>
        error instanceof ServarrToolError &&
        error.kind === "ambiguous_match" &&
        Array.isArray(error.details?.matches) &&
        error.details.matches.length === 2
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("summarizeSonarrSeriesList emits a compact id-aware library summary", () => {
  const summary = summarizeSonarrSeriesList([
    {
      id: 11,
      title: "Archer (2009)",
      year: 2009,
      status: "ended",
      monitored: true,
      tvdbId: 110381,
      tmdbId: 4629,
      imdbId: "tt1486217",
      statistics: {
        episodeFileCount: 10,
      },
      overview: "very long text that should not survive the summary",
    },
    {
      id: 12,
      title: "Severance",
      year: 2022,
      status: "continuing",
      monitored: false,
      tvdbId: 371980,
      tmdbId: 95396,
      imdbId: "tt11280740",
      statistics: {
        episodeFileCount: 0,
      },
    }
  ]) as Record<string, unknown>;

  assert.deepEqual(summary, {
    total_series: 2,
    monitored_series: 1,
    continuing_series: 1,
    ended_series: 1,
    downloaded_series: 1,
    titles: ["Archer (2009)", "Severance"],
    items: [
      {
        id: 11,
        title: "Archer (2009)",
        year: 2009,
        status: "ended",
        monitored: true,
        downloaded: true,
        tvdb_id: 110381,
        tmdb_id: 4629,
        imdb_id: "tt1486217",
        path: undefined,
      },
      {
        id: 12,
        title: "Severance",
        year: 2022,
        status: "continuing",
        monitored: false,
        downloaded: false,
        tvdb_id: 371980,
        tmdb_id: 95396,
        imdb_id: "tt11280740",
        path: undefined,
      },
    ],
  });
});

test("summarizeRadarrMovieList emits a compact id-aware library summary", () => {
  const summary = summarizeRadarrMovieList([
    {
      id: 55,
      title: "Alien",
      year: 1979,
      monitored: true,
      hasFile: true,
      minimumAvailability: "released",
      tmdbId: 348,
      imdbId: "tt0078748",
      overview: "very long text that should not survive the summary",
    },
    {
      id: 56,
      title: "Heat",
      year: 1995,
      monitored: false,
      hasFile: false,
      minimumAvailability: "announced",
      tmdbId: 949,
      imdbId: "tt0113277",
    }
  ]) as Record<string, unknown>;

  assert.deepEqual(summary, {
    total_movies: 2,
    monitored_movies: 1,
    downloaded_movies: 1,
    released_movies: 1,
    titles: ["Alien", "Heat"],
    items: [
      {
        id: 55,
        title: "Alien",
        year: 1979,
        monitored: true,
        downloaded: true,
        minimum_availability: "released",
        tmdb_id: 348,
        imdb_id: "tt0078748",
        path: undefined,
      },
      {
        id: 56,
        title: "Heat",
        year: 1995,
        monitored: false,
        downloaded: false,
        minimum_availability: "announced",
        tmdb_id: 949,
        imdb_id: "tt0113277",
        path: undefined,
      },
    ],
  });
});
