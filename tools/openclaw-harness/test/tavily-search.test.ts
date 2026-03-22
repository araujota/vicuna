import test from "node:test";
import assert from "node:assert/strict";

import {
  buildTavilySearchRequest,
  sanitizeTavilyResponse,
  TAVILY_DEFAULT_CHUNKS_PER_SOURCE,
  TAVILY_DEFAULT_MAX_RESULTS,
  TAVILY_INCLUDE_RAW_CONTENT,
  TAVILY_MAX_MAX_RESULTS,
  TAVILY_MIN_MAX_RESULTS,
} from "../src/tavily-search.js";

test("Tavily request defaults are source-first and quality-oriented", () => {
  const request = buildTavilySearchRequest("best pizza in chicago", {
    queryUrl: "best%20pizza%20in%20chicago",
    secretsPath: "/tmp/secrets.json",
    topic: "general",
    searchDepth: "advanced",
    maxResults: 1,
    includeDomains: [],
    excludeDomains: [],
  });

  assert.equal(request.include_answer, false);
  assert.equal(request.include_raw_content, TAVILY_INCLUDE_RAW_CONTENT);
  assert.equal(request.chunks_per_source, TAVILY_DEFAULT_CHUNKS_PER_SOURCE);
  assert.equal(request.max_results, TAVILY_MIN_MAX_RESULTS);
});

test("Tavily request preserves richer filters", () => {
  const request = buildTavilySearchRequest("tesla stock price", {
    queryUrl: "tesla%20stock%20price",
    secretsPath: "/tmp/secrets.json",
    topic: "finance",
    searchDepth: "advanced",
    maxResults: 99,
    timeRange: "day",
    includeDomains: ["finance.yahoo.com", "google.com"],
    excludeDomains: ["reddit.com"],
    country: "united states",
  });

  assert.equal(request.topic, "finance");
  assert.equal(request.time_range, "day");
  assert.equal(request.max_results, TAVILY_MAX_MAX_RESULTS);
  assert.deepEqual(request.include_domains, ["finance.yahoo.com", "google.com"]);
  assert.deepEqual(request.exclude_domains, ["reddit.com"]);
  assert.equal(request.country, "united states");
});

test("Tavily response sanitization strips provider answer and preserves bounded evidence", () => {
  const request = buildTavilySearchRequest("current weather chicago", {
    queryUrl: "current%20weather%20chicago",
    secretsPath: "/tmp/secrets.json",
    topic: "general",
    searchDepth: "advanced",
    maxResults: TAVILY_DEFAULT_MAX_RESULTS,
    includeDomains: [],
    excludeDomains: [],
  });

  const response = sanitizeTavilyResponse(
    {
      query: "current weather chicago",
      answer: "Chicago is 52F right now",
      response_time: 1.2,
      results: [
        {
          title: "Weather source",
          url: "https://example.com/weather",
          content: "Short snippet",
          raw_content: "A".repeat(2000),
          score: 0.98,
          published_date: "2026-03-22",
        },
      ],
    },
    request
  );

  assert.equal("answer" in response, false);
  assert.equal(response.retrieval_policy?.source_first, true);
  assert.equal(response.retrieval_policy?.include_answer, false);
  assert.equal(response.results?.length, 1);
  assert.equal(response.results?.[0]?.raw_content?.length, 1200);
});
