import { loadToolSecrets } from "./config.js";

type TavilyApiSearchResult = {
  title?: string;
  url?: string;
  content?: string;
  raw_content?: string | null;
  score?: number;
  published_date?: string;
};

type TavilySearchResult = {
  title?: string;
  url?: string;
  excerpt?: string;
  score?: number;
  published_date?: string;
};

type TavilySearchResponse = {
  query?: string;
  response_time?: number;
  retrieval_policy?: {
    source_first: boolean;
    include_answer: boolean;
    include_raw_content: "markdown";
    chunks_per_source: number;
    search_depth: "basic" | "advanced";
    max_results: number;
    topic: "general" | "news" | "finance";
    time_range?: "day" | "week" | "month" | "year";
    include_domains?: string[];
    exclude_domains?: string[];
    country?: string;
  };
  results?: TavilySearchResult[];
};

type TavilyCliOptions = {
  queryUrl: string;
  secretsPath: string;
  topic: "general" | "news" | "finance";
  searchDepth: "basic" | "advanced";
  maxResults: number;
  timeRange?: "day" | "week" | "month" | "year";
  includeDomains: string[];
  excludeDomains: string[];
  country?: string;
};

export const TAVILY_DEFAULT_MAX_RESULTS = 5;
export const TAVILY_MIN_MAX_RESULTS = 3;
export const TAVILY_MAX_MAX_RESULTS = 8;
export const TAVILY_DEFAULT_CHUNKS_PER_SOURCE = 3;
export const TAVILY_INCLUDE_RAW_CONTENT: "markdown" = "markdown";

export type TavilySearchRequestBody = {
  query: string;
  topic: "general" | "news" | "finance";
  search_depth: "basic" | "advanced";
  max_results: number;
  include_answer: false;
  include_raw_content: "markdown";
  chunks_per_source: number;
  time_range?: "day" | "week" | "month" | "year";
  include_domains?: string[];
  exclude_domains?: string[];
  country?: string;
};

function parseStringList(rawValue: string): string[] {
  return rawValue
    .split(",")
    .map((item) => decodeURIComponent(item).trim())
    .filter((item) => item.length > 0);
}

function clampMaxResults(value: number | undefined): number {
  const fallback = TAVILY_DEFAULT_MAX_RESULTS;
  if (!Number.isFinite(value ?? NaN)) {
    return fallback;
  }
  return Math.max(TAVILY_MIN_MAX_RESULTS, Math.min(TAVILY_MAX_MAX_RESULTS, Math.trunc(value ?? fallback)));
}

function parseArgs(argv: string[]): TavilyCliOptions {
  let queryUrl = "";
  let secretsPath = "";
  let topic: TavilyCliOptions["topic"] = "general";
  let searchDepth: TavilyCliOptions["searchDepth"] = "advanced";
  let maxResults = TAVILY_DEFAULT_MAX_RESULTS;
  let timeRange: TavilyCliOptions["timeRange"];
  let includeDomains: string[] = [];
  let excludeDomains: string[] = [];
  let country: string | undefined;

  for (const arg of argv) {
    if (arg.startsWith("--query-url=")) {
      queryUrl = arg.slice("--query-url=".length);
      continue;
    }
    if (arg.startsWith("--secrets-path=")) {
      secretsPath = arg.slice("--secrets-path=".length);
      continue;
    }
    if (arg.startsWith("--topic=")) {
      const value = arg.slice("--topic=".length);
      if (value === "general" || value === "news" || value === "finance") {
        topic = value;
      }
      continue;
    }
    if (arg.startsWith("--search-depth=")) {
      const value = arg.slice("--search-depth=".length);
      if (value === "basic" || value === "advanced") {
        searchDepth = value;
      }
      continue;
    }
    if (arg.startsWith("--max-results=")) {
      const value = Number.parseInt(arg.slice("--max-results=".length), 10);
      if (Number.isFinite(value)) {
        maxResults = clampMaxResults(value);
      }
      continue;
    }
    if (arg.startsWith("--time-range=")) {
      const value = arg.slice("--time-range=".length);
      if (value === "day" || value === "week" || value === "month" || value === "year") {
        timeRange = value;
      }
      continue;
    }
    if (arg.startsWith("--include-domains=")) {
      includeDomains = parseStringList(arg.slice("--include-domains=".length));
      continue;
    }
    if (arg.startsWith("--exclude-domains=")) {
      excludeDomains = parseStringList(arg.slice("--exclude-domains=".length));
      continue;
    }
    if (arg.startsWith("--country=")) {
      const value = decodeURIComponent(arg.slice("--country=".length)).trim();
      if (value) {
        country = value;
      }
      continue;
    }
  }

  if (!queryUrl) {
    throw new Error("missing required --query-url argument");
  }
  if (!secretsPath) {
    throw new Error("missing required --secrets-path argument");
  }

  return {
    queryUrl,
    secretsPath,
    topic,
    searchDepth,
    maxResults,
    timeRange,
    includeDomains,
    excludeDomains,
    country
  };
}

function decodeQuery(queryUrl: string): string {
  const decoded = decodeURIComponent(queryUrl);
  if (!decoded.trim()) {
    throw new Error("decoded Tavily query is empty");
  }
  return decoded;
}

export function buildTavilySearchRequest(query: string, options: TavilyCliOptions): TavilySearchRequestBody {
  const request: TavilySearchRequestBody = {
    query,
    topic: options.topic,
    search_depth: options.searchDepth,
    max_results: clampMaxResults(options.maxResults),
    include_answer: false,
    include_raw_content: TAVILY_INCLUDE_RAW_CONTENT,
    chunks_per_source: TAVILY_DEFAULT_CHUNKS_PER_SOURCE
  };
  if (options.timeRange) {
    request.time_range = options.timeRange;
  }
  if (options.includeDomains.length > 0) {
    request.include_domains = options.includeDomains;
  }
  if (options.excludeDomains.length > 0) {
    request.exclude_domains = options.excludeDomains;
  }
  if (options.country) {
    request.country = options.country;
  }
  return request;
}

function boundedResults(results: TavilyApiSearchResult[] | undefined, maxResults: number): TavilySearchResult[] {
  return (results ?? []).slice(0, clampMaxResults(maxResults)).map((result) => {
    const excerptSource = result.content?.trim() || result.raw_content?.trim() || undefined;
    return {
      title: result.title,
      url: result.url,
      excerpt: excerptSource?.slice(0, 400),
      score: result.score,
      published_date: result.published_date
    };
  });
}

export function sanitizeTavilyResponse(
  payload: Omit<TavilySearchResponse, "results"> & { answer?: string; results?: TavilyApiSearchResult[] },
  request: TavilySearchRequestBody
): TavilySearchResponse {
  return {
    query: payload.query ?? request.query,
    response_time: payload.response_time,
    retrieval_policy: {
      source_first: true,
      include_answer: false,
      include_raw_content: request.include_raw_content,
      chunks_per_source: request.chunks_per_source,
      search_depth: request.search_depth,
      max_results: request.max_results,
      topic: request.topic,
      time_range: request.time_range,
      include_domains: request.include_domains,
      exclude_domains: request.exclude_domains,
      country: request.country
    },
    results: boundedResults(payload.results, request.max_results)
  };
}

async function searchTavily(
  query: string,
  secretsPath: string,
  requestBody: TavilySearchRequestBody
): Promise<TavilySearchResponse> {
  const apiKey = loadToolSecrets(secretsPath).tools?.tavily?.api_key?.trim();
  if (!apiKey) {
    throw new Error("missing Tavily API key in OpenClaw tool secrets");
  }

  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Tavily request failed (${response.status}): ${body.slice(0, 400)}`);
  }

  const payload = (await response.json()) as TavilySearchResponse & { answer?: string };
  return sanitizeTavilyResponse(payload, requestBody);
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const query = decodeQuery(options.queryUrl);
  const requestBody = buildTavilySearchRequest(query, options);
  const payload = await searchTavily(
    query,
    options.secretsPath,
    requestBody
  );
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
