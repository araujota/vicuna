import test from "node:test";
import assert from "node:assert/strict";

import { DEFAULT_CHAPTARR_BASE_URL } from "../src/config.js";
import {
  chaptarrRequestJson,
  ChaptarrToolError,
  errorEnvelope,
  FIXED_CHAPTARR_ROOT_FOLDER_PATH,
  parseCliInvocation,
  resolveChaptarrConfig,
} from "../src/chaptarr-client.js";
import { handleChaptarr } from "../src/chaptarr.js";

test("resolveChaptarrConfig falls back to the LAN NAS default", () => {
  assert.equal(resolveChaptarrConfig("/tmp/does-not-exist.json").baseUrl, DEFAULT_CHAPTARR_BASE_URL);
});

test("parseCliInvocation decodes the base64 Chaptarr payload", () => {
  const encoded = Buffer.from(JSON.stringify({ action: "author_lookup", term: "Tolkien" }), "utf8").toString("base64");
  const parsed = parseCliInvocation([`--payload-base64=${encoded}`, "--secrets-path=/tmp/secrets.json"]);
  assert.equal(parsed.secretsPath, "/tmp/secrets.json");
  assert.equal(parsed.payload.action, "author_lookup");
  assert.equal(parsed.payload.term, "Tolkien");
});

test("chaptarrRequestJson returns a typed missing-api-key error before fetch", async () => {
  await assert.rejects(
    chaptarrRequestJson(
      {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: undefined
      },
      "GET",
      "/api/v1/system/status"
    ),
    (error: unknown) => error instanceof ChaptarrToolError && error.kind === "missing_api_key"
  );
});

test("errorEnvelope preserves typed Chaptarr errors", () => {
  const envelope = errorEnvelope(
    "add_author",
    DEFAULT_CHAPTARR_BASE_URL,
    new ChaptarrToolError("authorization_failed", "Chaptarr request failed", { status: 401 })
  );

  assert.equal(envelope.ok, false);
  assert.equal(envelope.error?.kind, "authorization_failed");
  assert.equal(envelope.error?.details?.status, 401);
});

test("handleChaptarr maps inspect to author listing", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET"
    });
    return new Response("[]", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: { action: "inspect" }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/author");
    assert.equal(fetchCalls.length, 1);
    assert.match(fetchCalls[0].url, /\/api\/v1\/author$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr filters generic search results down to ebook-capable book options", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET"
    });
    return new Response(JSON.stringify([
      {
        author: {
          id: 12,
          authorName: "J.R.R. Tolkien",
          foreignAuthorId: "hc:656983",
          monitored: true,
          titleSlug: "jrr-tolkien",
        }
      },
      {
        book: {
          id: 77,
          title: "The Hobbit (Audio)",
          foreignBookId: "hc:41",
          hardcoverBookId: "hc:41",
          mediaType: "audiobook",
          monitored: false,
          author: {
            id: 12,
            authorName: "J.R.R. Tolkien"
          },
          bookFiles: [
            {
              path: "/books/the-hobbit.m4b"
            }
          ]
        }
      },
      {
        book: {
          id: 91,
          title: "The Hobbit",
          foreignBookId: "hc:42",
          hardcoverBookId: "hc:42",
          monitored: true,
          editions: [
            {
              foreignEditionId: "edition-epub",
              isEbook: true,
              format: "EPUB"
            }
          ],
          bookFiles: [
            {
              path: "/books/the-hobbit.epub"
            }
          ],
          author: {
            id: 12,
            authorName: "J.R.R. Tolkien"
          }
        }
      }
    ]), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: { action: "search", term: "Tolkien", provider: "hardcover" }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/search");
    assert.deepEqual(response.request?.query, { term: "Tolkien", provider: "hardcover" });
    assert.deepEqual(response.data, {
      raw_result_count: 3,
      raw_book_result_count: 2,
      result_count: 2,
      author_result_count: 1,
      book_result_count: 1,
      authors: [
        {
          id: 12,
          author_name: "J.R.R. Tolkien",
          foreign_author_id: "hc:656983",
          monitored: true,
          path: undefined,
          title_slug: "jrr-tolkien",
          last_selected_media_type: undefined,
        }
      ],
      books: [
        {
          id: 91,
          title: "The Hobbit",
          author_name: "J.R.R. Tolkien",
          author_id: 12,
          foreign_book_id: "hc:42",
          foreign_edition_id: undefined,
          hardcover_book_id: "hc:42",
          monitored: true,
          media_type: undefined,
          path: undefined,
          has_files: false,
          edition_count: 1,
          ebook_edition_count: 1,
          ebook_formats: ["EPUB"],
          ebook_file_extensions: ["epub"],
          ebook_option_count: 2,
        }
      ],
    });
    assert.equal(fetchCalls.length, 1);
    assert.match(fetchCalls[0].url, /\/api\/v1\/search\?term=Tolkien&provider=hardcover$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr shapes book lookup requests", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET"
    });
    return new Response("[]", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: { action: "book_lookup", term: "The Hobbit" }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/book/lookup");
    assert.deepEqual(response.request?.query, { term: "The Hobbit" });
    assert.equal(fetchCalls.length, 1);
    assert.match(fetchCalls[0].url, /\/api\/v1\/book\/lookup\?term=The\+Hobbit$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr signals when book lookup finds only audiobook matches", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(JSON.stringify([
      {
        id: 88,
        title: "The Hobbit (Audio)",
        foreignBookId: "hc:41",
        mediaType: "audiobook",
        bookFiles: [
          {
            path: "/books/the-hobbit.m4b"
          }
        ],
        author: {
          authorName: "J.R.R. Tolkien"
        }
      }
    ]), {
      status: 200,
      headers: { "content-type": "application/json" }
    })) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: { action: "book_lookup", term: "The Hobbit" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      raw_book_count: 1,
      total_books: 0,
      titles: [],
      no_ebook_match: true,
      message: "No ebook-capable book matches were found.",
      items: [],
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr fetches only ebook profiles when media_type is omitted", async () => {
  const originalFetch = globalThis.fetch;
  const fetchUrls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request) => {
    fetchUrls.push(String(input));
    return new Response("[]", {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: { action: "quality_profiles" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.request?.query, { mediaType: "ebook" });
    assert.equal(fetchUrls.length, 1);
    assert.match(fetchUrls[0], /mediaType=ebook/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr turns add_author into immediate ebook acquisition", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v1/search")) {
      return new Response(
        JSON.stringify([
          {
            foreignId: "hc:656983",
            author: {
              authorName: "J.R.R. Tolkien",
              foreignAuthorId: "hc:656983",
              titleSlug: "jrr-tolkien",
              monitored: false,
              tags: [1],
              addOptions: {
                monitor: "none",
                monitored: false,
                searchForMissingBooks: false
              }
            }
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    if (String(input).endsWith("/api/v1/author")) {
      return new Response(JSON.stringify({
        id: 12,
        authorName: "J.R.R. Tolkien",
        foreignAuthorId: "hc:656983",
        monitored: true,
        path: "/books/jrr-tolkien",
        titleSlug: "jrr-tolkien",
        lastSelectedMediaType: "ebook"
      }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response(JSON.stringify({ id: 33, name: "AuthorSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "add_author",
        term: "Tolkien",
        foreign_author_id: "hc:656983",
        monitored: false,
        search_for_missing_books: false,
        tags: [4, 9]
      }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_author_search",
      author: {
        id: 12,
        author_name: "J.R.R. Tolkien",
        foreign_author_id: "hc:656983",
        monitored: true,
        path: "/books/jrr-tolkien",
        title_slug: "jrr-tolkien",
        last_selected_media_type: "ebook",
      },
      command: {
        id: 33,
        name: "AuthorSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
    assert.equal(fetchCalls.length, 3);
    assert.match(fetchCalls[0].url, /\/api\/v1\/search\?term=Tolkien&provider=hardcover$/);
    const postBody = JSON.parse(fetchCalls[1].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.rootFolderPath, FIXED_CHAPTARR_ROOT_FOLDER_PATH);
    assert.equal(postBody.qualityProfileId, 1);
    assert.equal(postBody.ebookQualityProfileId, 1);
    assert.equal(postBody.metadataProfileId, 2);
    assert.equal(postBody.ebookMetadataProfileId, 2);
    assert.equal(postBody.path, `${FIXED_CHAPTARR_ROOT_FOLDER_PATH}/jrr-tolkien`);
    assert.equal(postBody.lastSelectedMediaType, "ebook");
    assert.equal(postBody.monitored, true);
    assert.equal(postBody.monitorNewItems, "all");
    assert.deepEqual(postBody.tags, [4, 9]);
    assert.deepEqual(postBody.addOptions, {
      monitor: "all",
      monitored: true,
      searchForMissingBooks: true
    });
    const commandBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(fetchCalls[2].method, "POST");
    assert.equal(commandBody.name, "AuthorSearch");
    assert.equal(commandBody.authorId, 12);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr repairs an existing tracked author before triggering AuthorSearch", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({
      url,
      method,
      body: typeof init?.body === "string" ? init.body : null
    });
    if (url.includes("/api/v1/search")) {
      return new Response(JSON.stringify([
        {
          foreignId: "hc:656983",
          author: {
            id: 12,
            authorName: "J.R.R. Tolkien",
            foreignAuthorId: "hc:656983",
            titleSlug: "jrr-tolkien",
            monitored: false,
            lastSelectedMediaType: "audiobook"
          }
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/author/12") && method === "GET") {
      return new Response(JSON.stringify({
        id: 12,
        authorName: "J.R.R. Tolkien",
        titleSlug: "jrr-tolkien",
        monitored: false,
        lastSelectedMediaType: "audiobook"
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/author/12") && method === "PUT") {
      return new Response(init?.body as BodyInit, {
        status: 202,
        headers: { "content-type": "application/json" }
      });
    }
    if (url.endsWith("/api/v1/command")) {
      return new Response(JSON.stringify({ id: 33, name: "AuthorSearch" }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "add_author",
        term: "Tolkien",
        foreign_author_id: "hc:656983"
      }
    });

    assert.equal(response.ok, true);
    assert.equal(fetchCalls.length, 4);
    assert.deepEqual(response.data, {
      mode: "existing_author_search",
      existing_author_id: 12,
      author: {
        id: 12,
        author_name: "J.R.R. Tolkien",
        foreign_author_id: undefined,
        monitored: true,
        path: "/books/jrr-tolkien",
        title_slug: "jrr-tolkien",
        last_selected_media_type: "ebook",
      },
      command: {
        id: 33,
        name: "AuthorSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
    assert.equal(fetchCalls[1].method, "GET");
    assert.equal(fetchCalls[2].method, "PUT");
    const putBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(putBody.monitored, true);
    assert.equal(putBody.lastSelectedMediaType, "ebook");
    assert.equal(putBody.ebookQualityProfileId, 1);
    assert.equal(putBody.metadataProfileId, 2);
    assert.equal(putBody.ebookMetadataProfileId, 2);
    const commandBody = JSON.parse(fetchCalls[3].body ?? "{}") as Record<string, unknown>;
    assert.equal(commandBody.name, "AuthorSearch");
    assert.equal(commandBody.authorId, 12);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr turns add_book into immediate ebook acquisition", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? init.body : null
    });
    if (String(input).includes("/api/v1/search")) {
      return new Response(
        JSON.stringify([
          {
            foreignId: "622059",
            book: {
              title: "By the Horns",
              foreignBookId: "622059",
              hardcoverBookId: "hc:622059",
              monitored: false,
              anyEditionOk: false,
              addOptions: {
                searchForNewBook: false
              },
              author: {
                authorName: "Ruby Dixon",
                foreignAuthorId: "hc:252750",
                titleSlug: "ruby-dixon",
                monitored: false
              },
              editions: [
                {
                  foreignEditionId: "edition-audio",
                  titleSlug: "edition-audio",
                  isEbook: false,
                  monitored: false
                },
                {
                  foreignEditionId: "edition-ebook",
                  titleSlug: "edition-ebook",
                  isEbook: true,
                  monitored: false
                }
              ]
            }
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    if (String(input).endsWith("/api/v1/book")) {
      return new Response(JSON.stringify({
        id: 55,
        title: "By the Horns",
        foreignBookId: "622059",
        hardcoverBookId: "hc:622059",
        monitored: true,
        mediaType: "ebook",
        author: {
          id: 1,
          authorName: "Ruby Dixon"
        }
      }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response(JSON.stringify({ id: 89, name: "BookSearch" }), {
      status: 201,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "add_book",
        term: "By the Horns",
        foreign_book_id: "hc:622059",
        foreign_edition_id: "edition-ebook",
        monitored: false,
        search_for_new_book: false,
      }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_book_search",
      book: {
        id: 55,
        title: "By the Horns",
        author_name: "Ruby Dixon",
        author_id: 1,
        foreign_book_id: "622059",
        foreign_edition_id: undefined,
        hardcover_book_id: "hc:622059",
        monitored: true,
        media_type: "ebook",
        path: undefined,
        has_files: false,
        edition_count: undefined,
        ebook_edition_count: undefined,
      },
      command: {
        id: 89,
        name: "BookSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
    assert.equal(fetchCalls.length, 3);
    assert.match(fetchCalls[0].url, /\/api\/v1\/search\?term=By\+the\+Horns&provider=hardcover$/);
    const postBody = JSON.parse(fetchCalls[1].body ?? "{}") as Record<string, unknown>;
    assert.equal(postBody.monitored, true);
    assert.equal(postBody.mediaType, "ebook");
    assert.equal(postBody.lastSelectedMediaType, "ebook");
    assert.equal(postBody.anyEditionOk, false);
    assert.equal(postBody.qualityProfileId, 1);
    assert.equal(postBody.ebookQualityProfileId, 1);
    assert.equal(postBody.metadataProfileId, 2);
    assert.equal(postBody.ebookMetadataProfileId, 2);
    assert.equal(postBody.audiobookMonitored, false);
    assert.equal(postBody.rootFolderPath, FIXED_CHAPTARR_ROOT_FOLDER_PATH);
    assert.deepEqual(postBody.addOptions, {
      searchForNewBook: true
    });
    assert.deepEqual(postBody.author, {
      authorName: "Ruby Dixon",
      foreignAuthorId: "hc:252750",
      titleSlug: "ruby-dixon",
      monitored: true,
      rootFolderPath: FIXED_CHAPTARR_ROOT_FOLDER_PATH,
      path: `${FIXED_CHAPTARR_ROOT_FOLDER_PATH}/ruby-dixon`,
      qualityProfileId: 1,
      ebookQualityProfileId: 1,
      metadataProfileId: 2,
      ebookMetadataProfileId: 2,
      lastSelectedMediaType: "ebook"
    });
    assert.deepEqual(postBody.editions, [
      {
        foreignEditionId: "edition-ebook",
        titleSlug: "edition-ebook",
        isEbook: true,
        monitored: true
      }
    ]);
    const commandBody = JSON.parse(fetchCalls[2].body ?? "{}") as Record<string, unknown>;
    assert.equal(fetchCalls[2].method, "POST");
    assert.equal(commandBody.name, "BookSearch");
    assert.deepEqual(commandBody.bookIds, [55]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr triggers BookSearch for an already-tracked add_book target", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string; body?: string | null }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({
      url,
      method,
      body: typeof init?.body === "string" ? init.body : null
    });
    if (url.includes("/api/v1/search")) {
      return new Response(
        JSON.stringify([
          {
            foreignId: "622059",
            book: {
              title: "By the Horns",
              foreignBookId: "622059",
              localBookId: 91,
              authorId: 1,
              author: {
                id: 1,
                authorName: "Ruby Dixon",
                foreignAuthorId: "hc:252750",
                titleSlug: "ruby-dixon"
              }
            }
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    if (url.endsWith("/api/v1/book/91") && method === "GET") {
      return new Response(JSON.stringify({
        id: 91,
        title: "By the Horns",
        authorId: 1,
        monitored: false,
        ebookMonitored: false,
        audiobookMonitored: false,
        mediaType: "audiobook",
        addOptions: {
          addType: "manual",
          searchForNewBook: false
        },
        author: {
          id: 1,
          authorName: "Ruby Dixon",
          foreignAuthorId: "hc:252750",
          titleSlug: "ruby-dixon",
          monitored: false,
          lastSelectedMediaType: "audiobook"
        },
        editions: [
          {
            foreignEditionId: "edition-audio",
            titleSlug: "edition-audio",
            isEbook: false,
            monitored: false
          },
          {
            foreignEditionId: "edition-ebook",
            titleSlug: "edition-ebook",
            isEbook: true,
            monitored: false
          }
        ]
      }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (url.endsWith("/api/v1/author/1") && method === "GET") {
      return new Response(JSON.stringify({
        id: 1,
        authorName: "Ruby Dixon",
        foreignAuthorId: "hc:252750",
        titleSlug: "ruby-dixon",
        monitored: false,
        lastSelectedMediaType: "audiobook"
      }), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if ((url.endsWith("/api/v1/author/1") || url.endsWith("/api/v1/book/91")) && method === "PUT") {
      return new Response(init?.body as BodyInit, {
        status: 202,
        headers: { "content-type": "application/json" }
      });
    }
    if (url.endsWith("/api/v1/book/monitor") && method === "PUT") {
      return new Response(JSON.stringify([
        {
          id: 91,
          monitored: true,
          ebookMonitored: true,
          audiobookMonitored: false
        }
      ]), {
        status: 202,
        headers: { "content-type": "application/json" }
      });
    }
    if (url.endsWith("/api/v1/command")) {
      return new Response(JSON.stringify({ id: 55, name: "BookSearch" }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "add_book",
        term: "By the Horns",
      }
    });

    assert.equal(response.ok, true);
    assert.equal(fetchCalls.length, 7);
    assert.deepEqual(response.data, {
      mode: "existing_book_search",
      existing_book_id: 91,
      author: {
        id: 1,
        author_name: "Ruby Dixon",
        foreign_author_id: "hc:252750",
        monitored: true,
        path: "/books/ruby-dixon",
        title_slug: "ruby-dixon",
        last_selected_media_type: "ebook",
      },
      book: {
        id: 91,
        title: "By the Horns",
        author_name: "Ruby Dixon",
        author_id: 1,
        foreign_book_id: undefined,
        foreign_edition_id: undefined,
        hardcover_book_id: undefined,
        monitored: true,
        media_type: "ebook",
        path: undefined,
        has_files: false,
        edition_count: 1,
        ebook_edition_count: 1,
        ebook_option_count: 1,
      },
      monitor_result: {
        total_books: 1,
        titles: [],
        items: [
          {
            id: 91,
            title: undefined,
            author_name: undefined,
            author_id: undefined,
            foreign_book_id: undefined,
            foreign_edition_id: undefined,
            hardcover_book_id: undefined,
            monitored: true,
            media_type: undefined,
            path: undefined,
            has_files: false,
            edition_count: undefined,
            ebook_edition_count: undefined,
          }
        ],
      },
      command: {
        id: 55,
        name: "BookSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
    assert.equal(fetchCalls[1].method, "GET");
    assert.equal(fetchCalls[2].method, "GET");
    assert.equal(fetchCalls[3].method, "PUT");
    assert.equal(fetchCalls[4].method, "PUT");
    assert.equal(fetchCalls[5].method, "PUT");
    assert.equal(fetchCalls[6].method, "POST");
    const authorPut = JSON.parse(fetchCalls[3].body ?? "{}") as Record<string, unknown>;
    assert.equal(authorPut.monitored, true);
    assert.equal(authorPut.lastSelectedMediaType, "ebook");
    const bookPut = JSON.parse(fetchCalls[4].body ?? "{}") as Record<string, unknown>;
    assert.equal(bookPut.mediaType, "ebook");
    assert.equal(bookPut.monitored, true);
    assert.equal(bookPut.ebookMonitored, true);
    const monitorBody = JSON.parse(fetchCalls[5].body ?? "{}") as Record<string, unknown>;
    assert.deepEqual(monitorBody.bookIds, [91]);
    assert.equal(monitorBody.monitored, true);
    const commandBody = JSON.parse(fetchCalls[6].body ?? "{}") as Record<string, unknown>;
    assert.equal(commandBody.name, "BookSearch");
    assert.deepEqual(commandBody.bookIds, [91]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr rejects audiobook-only download candidates", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    fetchCalls.push({
      url: String(input),
      method: init?.method ?? "GET"
    });
    return new Response(JSON.stringify([
      {
        foreignId: "hc:41",
        book: {
          title: "The Hobbit (Audio)",
          foreignBookId: "hc:41",
          mediaType: "audiobook",
          author: {
            authorName: "J.R.R. Tolkien",
            foreignAuthorId: "hc:656983",
            titleSlug: "jrr-tolkien",
          },
          bookFiles: [
            {
              path: "/books/the-hobbit.m4b"
            }
          ]
        }
      }
    ]), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    await assert.rejects(
      handleChaptarr({
        service: "chaptarr",
        config: {
          service: "chaptarr",
          baseUrl: DEFAULT_CHAPTARR_BASE_URL,
          apiKey: "test-key"
        },
        payload: {
          action: "download_book",
          term: "The Hobbit",
        }
      }),
      (error: unknown) =>
        error instanceof ChaptarrToolError &&
        error.kind === "no_ebook_match"
    );
    assert.equal(fetchCalls.length, 1);
    assert.match(fetchCalls[0].url, /\/api\/v1\/search\?term=The\+Hobbit&provider=hardcover$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr deletes one tracked book and removes files by default", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({ url, method });
    if (url.endsWith("/api/v1/book")) {
      return new Response(
        JSON.stringify([
          {
            id: "91",
            localBookId: "91",
            title: "By the Horns",
            foreignBookId: "622059",
            hardcoverBookId: "hc:622059",
            author: {
              authorName: "Ruby Dixon"
            }
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
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_book",
        term: "By the Horns Ruby Dixon"
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/book/91");
    assert.deepEqual(response.data, {
      mode: "tracked_book_delete",
      deleted_book: {
        id: 91,
        title: "By the Horns",
        author_name: "Ruby Dixon",
        author_id: undefined,
        foreign_book_id: "622059",
        foreign_edition_id: undefined,
        hardcover_book_id: "hc:622059",
        monitored: false,
        media_type: undefined,
        path: undefined,
        has_files: false,
        edition_count: undefined,
        ebook_edition_count: undefined,
      }
    });
    assert.deepEqual(response.request?.query, {
      deleteFiles: true,
      addImportListExclusion: false
    });
    assert.equal(fetchCalls.length, 2);
    assert.equal(fetchCalls[0].method, "GET");
    assert.equal(fetchCalls[1].method, "DELETE");
    assert.match(fetchCalls[1].url, /\/api\/v1\/book\/91\?deleteFiles=true&addImportListExclusion=false$/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr rejects ambiguous tracked book deletes", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify([
        {
          id: 91,
          title: "Foundation",
          author: {
            authorName: "Isaac Asimov"
          }
        },
        {
          id: 92,
          title: "Foundation",
          author: {
            authorName: "Isaac Asimov"
          }
        }
      ]),
      { status: 200, headers: { "content-type": "application/json" } }
    )) as typeof fetch;

  try {
    await assert.rejects(
      handleChaptarr({
        service: "chaptarr",
        config: {
          service: "chaptarr",
          baseUrl: DEFAULT_CHAPTARR_BASE_URL,
          apiKey: "test-key"
        },
        payload: {
          action: "delete_book",
          term: "Foundation"
        }
      }),
      (error: unknown) =>
        error instanceof ChaptarrToolError &&
        error.kind === "ambiguous_match" &&
        Array.isArray(error.details?.matches) &&
        error.details.matches.length === 2
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr ignores placeholder ids and falls back to title matching for delete_book", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET" && url.endsWith("/api/v1/book")) {
      return new Response(
        JSON.stringify([
          {
            id: 91,
            localBookId: "0",
            title: "By the Horns",
            author: {
              authorName: "Ruby Dixon"
            }
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
    const response = await handleChaptarr({
      service: "chaptarr",
      config: {
        service: "chaptarr",
        baseUrl: DEFAULT_CHAPTARR_BASE_URL,
        apiKey: "test-key"
      },
      payload: {
        action: "delete_book",
        book_id: 0,
        term: "By the Horns Ruby Dixon"
      }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/book/91");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr diagnoses upstream V5 metadata failures for add_book", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("/api/v1/search")) {
      return new Response(
        JSON.stringify([
          {
            foreignId: "622059",
            book: {
              title: "By the Horns",
              foreignBookId: "622059",
              hardcoverBookId: "hc:622059",
              author: {
                authorName: "Ruby Dixon",
                foreignAuthorId: "hc:252750",
                titleSlug: "ruby-dixon",
              },
            }
          }
        ]),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    if (url.includes("/api/v1/log")) {
      return new Response(
        JSON.stringify({
          records: [
            {
              time: "2026-03-23T21:04:24Z",
              exception:
                "NzbDrone.Core.MetadataSource.BookInfo.BookInfoException: Failed to get author info from V5 API\\n" +
                "---> NzbDrone.Common.Http.HttpException: HTTP request failed: [525:525] [GET] at [https://api.chaptarr.com/api/v5/author?id=hc%3A252750]"
            }
          ]
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" }
        }
      );
    }
    return new Response(
      JSON.stringify({ message: "Failed to get author info from V5 API" }),
      {
        status: 503,
        headers: { "content-type": "application/json" }
      }
    );
  }) as typeof fetch;

  try {
    await assert.rejects(
      handleChaptarr({
        service: "chaptarr",
        config: {
          service: "chaptarr",
          baseUrl: DEFAULT_CHAPTARR_BASE_URL,
          apiKey: "test-key"
        },
        payload: {
          action: "add_book",
          term: "By the Horns",
        }
      }),
      (error: unknown) =>
        error instanceof ChaptarrToolError &&
        error.kind === "upstream_metadata_failure" &&
        error.details?.upstream_status === 525 &&
        error.details?.upstream_url === "https://api.chaptarr.com/api/v5/author?id=hc%3A252750"
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
