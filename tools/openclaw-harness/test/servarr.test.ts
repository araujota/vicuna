import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_RADARR_BASE_URL,
  DEFAULT_SONARR_BASE_URL,
} from "../src/config.js";
import {
  canonicalizeServarrAction,
  FIXED_RADARR_QUALITY_PROFILE_ID,
  FIXED_RADARR_ROOT_FOLDER_PATH,
  FIXED_SONARR_QUALITY_PROFILE_ID,
  FIXED_SONARR_ROOT_FOLDER_PATH,
  summarizeRadarrMovieList,
  summarizeSonarrSeriesList,
} from "../src/servarr.js";
import { handleRadarr } from "../src/radarr.js";
import { handleSonarr } from "../src/sonarr.js";

test("canonicalizeServarrAction maps inspect to downloaded-library actions", () => {
  assert.equal(canonicalizeServarrAction("radarr", "inspect"), "list_downloaded_movies");
  assert.equal(canonicalizeServarrAction("sonarr", "inspect"), "list_downloaded_series");
});

test("summarizeRadarrMovieList returns only downloaded movie summaries", () => {
  assert.deepEqual(
    summarizeRadarrMovieList([
      {
        id: 1,
        title: "Alien",
        year: 1979,
        hasFile: true,
        tmdbId: 348,
      },
      {
        id: 2,
        title: "Heat",
        year: 1995,
        hasFile: false,
        tmdbId: 949,
      }
    ]),
    {
      downloaded_movie_count: 1,
      items: [
        {
          id: 1,
          title: "Alien",
          year: 1979,
          downloaded: true,
          tmdb_id: 348,
          imdb_id: undefined,
          path: undefined,
        }
      ]
    }
  );
});

test("summarizeSonarrSeriesList returns only downloaded series summaries", () => {
  assert.deepEqual(
    summarizeSonarrSeriesList([
      {
        id: 10,
        title: "Severance",
        year: 2022,
        tvdbId: 366524,
        statistics: { episodeFileCount: 5 }
      },
      {
        id: 11,
        title: "New Show",
        year: 2026,
        tvdbId: 400000,
        statistics: { episodeFileCount: 0 }
      }
    ]),
    {
      downloaded_series_count: 1,
      items: [
        {
          id: 10,
          title: "Severance",
          year: 2022,
          downloaded: true,
          tvdb_id: 366524,
          tmdb_id: undefined,
          imdb_id: undefined,
          path: undefined,
        }
      ]
    }
  );
});

test("handleRadarr adds a new movie with fixed deployment settings", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/movie/lookup")) {
      return new Response(JSON.stringify([{ title: "Alien", tmdbId: 348, minimumAvailability: "released" }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (String(input).endsWith("/api/v3/movie")) {
      if ((init?.method ?? "GET") === "GET") {
        return new Response("[]", { status: 200, headers: { "content-type": "application/json" } });
      }
      return new Response(JSON.stringify({
        id: 100,
        title: "Alien",
        year: 1979,
        hasFile: false,
        tmdbId: 348,
        imdbId: "tt0078748",
        path: "/movies/Alien (1979)"
      }), { status: 201, headers: { "content-type": "application/json" } });
    }
    throw new Error(`unexpected fetch ${String(input)}`);
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: { service: "radarr", baseUrl: DEFAULT_RADARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_movie", term: "Alien" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_movie_added_and_search_requested",
      message: "Added 'Alien' to Radarr and requested an immediate search.",
      movie: {
        id: 100,
        title: "Alien",
        year: 1979,
        downloaded: false,
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

test("handleRadarr triggers MoviesSearch for an already tracked movie", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/movie/lookup")) {
      return new Response(JSON.stringify([{ title: "Alien", tmdbId: 348, year: 1979 }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (String(input).endsWith("/api/v3/movie")) {
      return new Response(JSON.stringify([{ id: 55, title: "Alien", tmdbId: 348, year: 1979 }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response(JSON.stringify({ id: 99, name: "MoviesSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: { service: "radarr", baseUrl: DEFAULT_RADARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_movie", term: "Alien" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "existing_movie_search_started",
      message: "Started a Radarr search for 'Alien'.",
      movie: {
        id: 55,
        title: "Alien",
        year: 1979,
        downloaded: false,
        tmdb_id: 348,
        imdb_id: undefined,
        path: undefined,
      },
      search_command: {
        id: 99,
        name: "MoviesSearch",
        status: undefined,
        state: undefined,
        message: undefined,
        body: undefined,
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleRadarr deletes multiple movies with compact outcomes", async () => {
  const originalFetch = globalThis.fetch;
  const deletedUrls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET") {
      return new Response(JSON.stringify([
        { id: 55, title: "Alien", year: 1979, hasFile: true, tmdbId: 348 },
        { id: 56, title: "Heat", year: 1995, hasFile: true, tmdbId: 949 }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    deletedUrls.push(url);
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;

  try {
    const response = await handleRadarr({
      service: "radarr",
      config: { service: "radarr", baseUrl: DEFAULT_RADARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "delete_movies", terms: ["Alien", "Heat"] }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "movies_deleted",
      message: "Deleted 2 movies from Radarr.",
      deleted_count: 2,
      deleted_movies: [
        {
          id: 55,
          title: "Alien",
          year: 1979,
          downloaded: true,
          tmdb_id: 348,
          imdb_id: undefined,
          path: undefined,
        },
        {
          id: 56,
          title: "Heat",
          year: 1995,
          downloaded: true,
          tmdb_id: 949,
          imdb_id: undefined,
          path: undefined,
        }
      ]
    });
    assert.equal(deletedUrls.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr adds a new series with fixed deployment settings", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v3/series/lookup")) {
      return new Response(JSON.stringify([{ title: "Severance", tvdbId: 366524, seriesType: "standard", monitorNewItems: "all" }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (String(input).endsWith("/api/v3/series")) {
      if ((init?.method ?? "GET") === "GET") {
        return new Response("[]", { status: 200, headers: { "content-type": "application/json" } });
      }
      return new Response(JSON.stringify({
        id: 200,
        title: "Severance",
        year: 2022,
        tvdbId: 366524,
        tmdbId: 95396,
        imdbId: "tt11280740",
        path: "/tv/Severance",
        statistics: { episodeFileCount: 0 }
      }), { status: 201, headers: { "content-type": "application/json" } });
    }
    throw new Error(`unexpected fetch ${String(input)}`);
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: { service: "sonarr", baseUrl: DEFAULT_SONARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_series", term: "Severance" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_series_added_and_search_requested",
      message: "Added 'Severance' to Sonarr and requested an immediate search.",
      series: {
        id: 200,
        title: "Severance",
        year: 2022,
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

test("handleSonarr triggers SeriesSearch for an already tracked series", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    if (String(input).includes("/api/v3/series/lookup")) {
      return new Response(JSON.stringify([{ title: "Severance", tvdbId: 366524, tmdbId: 95396, year: 2022 }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (String(input).endsWith("/api/v3/series")) {
      return new Response(JSON.stringify([{ id: 77, title: "Severance", tvdbId: 366524, tmdbId: 95396 }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response(JSON.stringify({ id: 123, name: "SeriesSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: { service: "sonarr", baseUrl: DEFAULT_SONARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_series", term: "Severance" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "existing_series_search_started",
      message: "Started a Sonarr search for 'Severance'.",
      series: {
        id: 77,
        title: "Severance",
        year: undefined,
        downloaded: false,
        tvdb_id: 366524,
        tmdb_id: 95396,
        imdb_id: undefined,
        path: undefined,
      },
      search_command: {
        id: 123,
        name: "SeriesSearch",
        status: undefined,
        state: undefined,
        message: undefined,
        body: undefined,
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleSonarr deletes multiple series with compact outcomes", async () => {
  const originalFetch = globalThis.fetch;
  const deletedUrls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET") {
      return new Response(JSON.stringify([
        { id: 22, title: "The Simpsons", year: 1989, tvdbId: 71663, tmdbId: 456, imdbId: "tt0096697", statistics: { episodeFileCount: 20 } },
        { id: 77, title: "Severance", year: 2022, tvdbId: 366524, tmdbId: 95396, statistics: { episodeFileCount: 5 } },
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    deletedUrls.push(url);
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;

  try {
    const response = await handleSonarr({
      service: "sonarr",
      config: { service: "sonarr", baseUrl: DEFAULT_SONARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "delete_series", terms: ["The Simpsons", "Severance"] }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "series_deleted",
      message: "Deleted 2 series from Sonarr.",
      deleted_count: 2,
      deleted_series: [
        {
          id: 22,
          title: "The Simpsons",
          year: 1989,
          downloaded: true,
          tvdb_id: 71663,
          tmdb_id: 456,
          imdb_id: "tt0096697",
          path: undefined,
        },
        {
          id: 77,
          title: "Severance",
          year: 2022,
          downloaded: true,
          tvdb_id: 366524,
          tmdb_id: 95396,
          imdb_id: undefined,
          path: undefined,
        }
      ]
    });
    assert.equal(deletedUrls.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
