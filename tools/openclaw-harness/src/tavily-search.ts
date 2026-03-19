import { loadToolSecrets } from "./config.js";

type TavilySearchResult = {
  title?: string;
  url?: string;
  content?: string;
  score?: number;
  published_date?: string;
};

type TavilySearchResponse = {
  answer?: string;
  query?: string;
  response_time?: number;
  results?: TavilySearchResult[];
};

type TavilyCliOptions = {
  queryUrl: string;
  secretsPath: string;
  topic: "general" | "news";
  searchDepth: "basic" | "advanced";
  maxResults: number;
};

function parseArgs(argv: string[]): TavilyCliOptions {
  let queryUrl = "";
  let secretsPath = "";
  let topic: TavilyCliOptions["topic"] = "general";
  let searchDepth: TavilyCliOptions["searchDepth"] = "advanced";
  let maxResults = 5;

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
      if (value === "general" || value === "news") {
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
      if (Number.isFinite(value) && value >= 1 && value <= 10) {
        maxResults = value;
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
    maxResults
  };
}

function decodeQuery(queryUrl: string): string {
  const decoded = decodeURIComponent(queryUrl);
  if (!decoded.trim()) {
    throw new Error("decoded Tavily query is empty");
  }
  return decoded;
}

function boundedResults(results: TavilySearchResult[] | undefined): TavilySearchResult[] {
  return (results ?? []).slice(0, 5).map((result) => ({
    title: result.title,
    url: result.url,
    content: result.content?.slice(0, 600),
    score: result.score,
    published_date: result.published_date
  }));
}

async function searchTavily(
  query: string,
  secretsPath: string,
  topic: TavilyCliOptions["topic"],
  searchDepth: TavilyCliOptions["searchDepth"],
  maxResults: number
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
    body: JSON.stringify({
      query,
      topic,
      search_depth: searchDepth,
      max_results: maxResults,
      include_answer: true
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Tavily request failed (${response.status}): ${body.slice(0, 400)}`);
  }

  const payload = (await response.json()) as TavilySearchResponse;
  return {
    query: payload.query ?? query,
    answer: payload.answer,
    response_time: payload.response_time,
    results: boundedResults(payload.results)
  };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const query = decodeQuery(options.queryUrl);
  const payload = await searchTavily(
    query,
    options.secretsPath,
    options.topic,
    options.searchDepth,
    options.maxResults
  );
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
