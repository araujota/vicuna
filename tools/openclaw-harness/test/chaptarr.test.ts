import test from "node:test";
import assert from "node:assert/strict";

import { DEFAULT_CHAPTARR_BASE_URL } from "../src/config.js";
import { handleChaptarr } from "../src/chaptarr.js";

test("handleChaptarr maps inspect to downloaded book listing", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request) => {
    fetchCalls.push(String(input));
    return new Response(JSON.stringify([
      {
        id: 91,
        title: "The Hobbit",
        hasFiles: true,
        foreignBookId: "hc:42",
        author: { authorName: "J.R.R. Tolkien" },
        bookFiles: [{ path: "/books/the-hobbit.epub" }]
      },
      {
        id: 92,
        title: "Untracked",
        hasFiles: false,
        author: { authorName: "Unknown" }
      }
    ]), {
      status: 200,
      headers: { "content-type": "application/json" }
    });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: { service: "chaptarr", baseUrl: DEFAULT_CHAPTARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "inspect" }
    });

    assert.equal(response.ok, true);
    assert.equal(response.request?.path, "/api/v1/book");
    assert.deepEqual(response.data, {
      matched_book_count: 2,
      downloaded_book_count: 1,
      items: [
        {
          id: 91,
          title: "The Hobbit",
          author_name: "J.R.R. Tolkien",
          downloaded: true,
          foreign_book_id: "hc:42",
          path: undefined,
          ebook_file_extensions: ["epub"],
        }
      ]
    });
    assert.equal(fetchCalls.length, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr adds a new book and requests an immediate search", async () => {
  const originalFetch = globalThis.fetch;
  const fetchCalls: Array<{ url: string; method: string }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    fetchCalls.push({ url, method });

    if (url.includes("/api/v1/search")) {
      return new Response(JSON.stringify([
        {
          book: {
            title: "The Hobbit",
            foreignBookId: "hc:42",
            editions: [{ foreignEditionId: "ed-1", isEbook: true, format: "EPUB" }],
            author: { id: 12, authorName: "J.R.R. Tolkien" }
          }
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/book") && method === "POST") {
      return new Response(JSON.stringify({
        id: 55,
        title: "The Hobbit",
        foreignBookId: "hc:42",
        hasFiles: false,
        author: { id: 12, authorName: "J.R.R. Tolkien" }
      }), { status: 201, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/command")) {
      return new Response(JSON.stringify({ id: 99, name: "BookSearch" }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: { service: "chaptarr", baseUrl: DEFAULT_CHAPTARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_book", term: "The Hobbit" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_book_added_and_search_requested",
      message: "Added 'The Hobbit' to Chaptarr and requested an immediate search.",
      book: {
        id: 55,
        title: "The Hobbit",
        author_name: "J.R.R. Tolkien",
        downloaded: false,
        foreign_book_id: "hc:42",
        path: undefined,
      },
      search_command: {
        id: 99,
        name: "BookSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr repairs an existing tracked book and triggers BookSearch", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";

    if (url.includes("/api/v1/search")) {
      return new Response(JSON.stringify([
        {
          book: {
            id: 55,
            title: "The Hobbit",
            foreignBookId: "hc:42",
            author: { id: 12, authorName: "J.R.R. Tolkien" },
            editions: [{ foreignEditionId: "ed-1", isEbook: true, format: "EPUB" }]
          }
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/book/55") && method === "GET") {
      return new Response(JSON.stringify({
        id: 55,
        title: "The Hobbit",
        foreignBookId: "hc:42",
        authorId: 12,
        author: { id: 12, authorName: "J.R.R. Tolkien" }
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/author/12") && method === "GET") {
      return new Response(JSON.stringify({
        id: 12,
        authorName: "J.R.R. Tolkien"
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/author/12") && method === "PUT") {
      return new Response(JSON.stringify({
        id: 12,
        authorName: "J.R.R. Tolkien"
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/book/55") && method === "PUT") {
      return new Response(JSON.stringify({
        id: 55,
        title: "The Hobbit",
        foreignBookId: "hc:42",
        author: { id: 12, authorName: "J.R.R. Tolkien" }
      }), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/book/monitor")) {
      return new Response(JSON.stringify([{ id: 55 }]), {
        status: 200,
        headers: { "content-type": "application/json" }
      });
    }
    if (url.endsWith("/api/v1/command")) {
      return new Response(JSON.stringify({ id: 100, name: "BookSearch" }), {
        status: 201,
        headers: { "content-type": "application/json" }
      });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: { service: "chaptarr", baseUrl: DEFAULT_CHAPTARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_book", term: "The Hobbit" }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "existing_book_search_started",
      message: "Started a Chaptarr search for 'The Hobbit'.",
      book: {
        id: 55,
        title: "The Hobbit",
        author_name: "J.R.R. Tolkien",
        downloaded: false,
        foreign_book_id: "hc:42",
        path: undefined,
      },
      search_command: {
        id: 100,
        name: "BookSearch",
        status: undefined,
        state: undefined,
        body: undefined,
        message: undefined,
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr reports queued legal-import fallback when direct ebook acquisition is unavailable", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";

    if (url.includes("/api/v1/search")) {
      return new Response(JSON.stringify([
        {
          book: {
            title: "The Hobbit",
            foreignBookId: "hc:42",
            mediaType: "audiobook",
            bookFiles: [{ path: "/books/the-hobbit.m4b" }],
            author: { id: 12, authorName: "J.R.R. Tolkien" }
          }
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.includes("/api/v1/book/lookup")) {
      return new Response(JSON.stringify([
        {
          title: "The Hobbit",
          foreignBookId: "hc:42",
          author: { id: 12, authorName: "J.R.R. Tolkien" }
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    if (url.endsWith("/api/v1/book") && method === "POST") {
      return new Response(JSON.stringify({
        id: 55,
        title: "The Hobbit",
        foreignBookId: "hc:42",
        author: { id: 12, authorName: "J.R.R. Tolkien" }
      }), { status: 201, headers: { "content-type": "application/json" } });
    }
    throw new Error(`unexpected fetch ${url}`);
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: { service: "chaptarr", baseUrl: DEFAULT_CHAPTARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "download_book", term: "The Hobbit" },
      legalImporterRunner: async () => ({
        status: "queued_for_scheduler",
        triggered_immediately: false,
        waited_for_completion: false,
        message: "Queued The Hobbit in Chaptarr's missing/wanted list for legal-importer pickup.",
      })
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "new_book_missing_fallback",
      book: {
        id: 55,
        title: "The Hobbit",
        author_name: "J.R.R. Tolkien",
        downloaded: false,
        foreign_book_id: "hc:42",
        path: undefined,
      },
      importer_status: undefined,
      message: "Queued The Hobbit in Chaptarr's missing/wanted list for legal-importer pickup.",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("handleChaptarr deletes multiple books with compact outcomes", async () => {
  const originalFetch = globalThis.fetch;
  const deletedUrls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    if (method === "GET") {
      return new Response(JSON.stringify([
        {
          id: 91,
          title: "The Hobbit",
          hasFiles: true,
          foreignBookId: "hc:42",
          author: { authorName: "J.R.R. Tolkien" },
          bookFiles: [{ path: "/books/the-hobbit.epub" }]
        },
        {
          id: 92,
          title: "The Silmarillion",
          hasFiles: true,
          foreignBookId: "hc:43",
          author: { authorName: "J.R.R. Tolkien" },
          bookFiles: [{ path: "/books/the-silmarillion.epub" }]
        }
      ]), { status: 200, headers: { "content-type": "application/json" } });
    }
    deletedUrls.push(url);
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;

  try {
    const response = await handleChaptarr({
      service: "chaptarr",
      config: { service: "chaptarr", baseUrl: DEFAULT_CHAPTARR_BASE_URL, apiKey: "test-key" },
      payload: { action: "delete_books", terms: ["The Hobbit", "The Silmarillion"] }
    });

    assert.equal(response.ok, true);
    assert.deepEqual(response.data, {
      mode: "books_deleted",
      message: "Deleted 2 books from Chaptarr.",
      deleted_count: 2,
      deleted_books: [
        {
          id: 91,
          title: "The Hobbit",
          author_name: "J.R.R. Tolkien",
          downloaded: true,
          foreign_book_id: "hc:42",
          path: undefined,
          ebook_file_extensions: ["epub"],
        },
        {
          id: 92,
          title: "The Silmarillion",
          author_name: "J.R.R. Tolkien",
          downloaded: true,
          foreign_book_id: "hc:43",
          path: undefined,
          ebook_file_extensions: ["epub"],
        }
      ]
    });
    assert.equal(deletedUrls.length, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
