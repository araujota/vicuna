import {
  canonicalizeChaptarrAction,
  chaptarrRequestJson,
  chooseLookupCandidate,
  ChaptarrToolError,
  FIXED_CHAPTARR_MEDIA_TYPE,
  FIXED_CHAPTARR_ROOT_FOLDER_PATH,
  optionalBoolean,
  optionalInteger,
  optionalIntegerArray,
  optionalString,
  requireString,
  resolveMediaTypes,
  runChaptarrCli,
  successEnvelope,
  type ChaptarrCliContext,
  type ChaptarrInvocation,
} from "./chaptarr-client.js";

const DEFAULT_ADD_AUTHOR_MONITOR = "all";
const DEFAULT_MONITOR_NEW_ITEMS = "all";
const DEFAULT_CHAPTARR_SEARCH_PROVIDER = "hardcover";
const DEFAULT_CHAPTARR_EBOOK_QUALITY_PROFILE_ID = 1;
const DEFAULT_CHAPTARR_EBOOK_METADATA_PROFILE_ID = 2;
const EBOOK_FILE_EXTENSIONS = new Set(["epub", "mobi", "azw", "azw3", "kfx", "pdf", "fb2", "djvu", "djv"]);
const AUDIOBOOK_FILE_EXTENSIONS = new Set(["mp3", "m4b", "m4a", "aac", "aax", "flac", "ogg", "opus", "wav", "wma"]);

type ChaptarrRecord = Record<string, unknown>;
type BookMediaAnalysis = {
  eligible: boolean;
  ebookEditions: ChaptarrRecord[];
  ebookFiles: ChaptarrRecord[];
  ebookFormats: string[];
  ebookFileExtensions: string[];
};

function cloneRecord(value: unknown): ChaptarrRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return JSON.parse(JSON.stringify(value)) as ChaptarrRecord;
}

function recordString(record: ChaptarrRecord, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function recordInteger(record: ChaptarrRecord, key: string): number | undefined {
  const value = record[key];
  return typeof value === "number" && Number.isInteger(value) ? value : undefined;
}

function recordIntegerLike(record: ChaptarrRecord, key: string): number | undefined {
  const value = record[key];
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }
  if (typeof value === "string" && /^[0-9]+$/.test(value.trim())) {
    return Number.parseInt(value.trim(), 10);
  }
  return undefined;
}

function recordBoolean(record: ChaptarrRecord, key: string): boolean | undefined {
  const value = record[key];
  return typeof value === "boolean" ? value : undefined;
}

function resolveTrackedAuthorId(record: ChaptarrRecord): number | undefined {
  return (
    recordIntegerLike(record, "id") ??
    recordIntegerLike(record, "authorId") ??
    recordIntegerLike(record, "localAuthorId")
  );
}

function resolveTrackedBookId(record: ChaptarrRecord): number | undefined {
  return (
    recordIntegerLike(record, "id") ??
    recordIntegerLike(record, "bookId") ??
    recordIntegerLike(record, "localBookId")
  );
}

function recordObject(record: ChaptarrRecord, key: string): ChaptarrRecord | undefined {
  const value = record[key];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return cloneRecord(value);
}

function cloneRecordArray(value: unknown): ChaptarrRecord[] {
  if (Array.isArray(value)) {
    return value
      .filter((entry) => entry && typeof entry === "object" && !Array.isArray(entry))
      .map((entry) => cloneRecord(entry));
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const nestedValue = (value as ChaptarrRecord).value;
    if (Array.isArray(nestedValue)) {
      return nestedValue
        .filter((entry) => entry && typeof entry === "object" && !Array.isArray(entry))
        .map((entry) => cloneRecord(entry));
    }
  }
  return [];
}

function recordArray(record: ChaptarrRecord, key: string): ChaptarrRecord[] {
  return cloneRecordArray(record[key]);
}

function sortStrings(values: string[]): string[] {
  return [...new Set(values
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
  )].sort((left, right) => left.localeCompare(right));
}

function recordPathExtension(record: ChaptarrRecord): string | undefined {
  const explicitExtension = recordString(record, "extension");
  if (explicitExtension) {
    return explicitExtension.toLowerCase().replace(/^\./, "");
  }
  const path = recordString(record, "path");
  if (!path) {
    return undefined;
  }
  const match = /\.([A-Za-z0-9]+)$/.exec(path);
  return match ? match[1].toLowerCase() : undefined;
}

function matchesFormatPattern(value: string | undefined, extensions: Set<string>): boolean {
  if (!value) {
    return false;
  }
  const normalized = value.toLowerCase();
  return [...extensions].some((extension) => normalized.includes(extension));
}

function isEbookEdition(edition: ChaptarrRecord): boolean {
  return (
    edition.isEbook === true ||
    recordString(edition, "mediaType") === FIXED_CHAPTARR_MEDIA_TYPE ||
    matchesFormatPattern(recordString(edition, "format"), EBOOK_FILE_EXTENSIONS)
  );
}

function isAudiobookEdition(edition: ChaptarrRecord): boolean {
  return (
    edition.isEbook === false ||
    recordString(edition, "mediaType") === "audiobook" ||
    matchesFormatPattern(recordString(edition, "format"), AUDIOBOOK_FILE_EXTENSIONS)
  );
}

function isEbookFile(file: ChaptarrRecord): boolean {
  const extension = recordPathExtension(file);
  return (
    recordBoolean(file, "isEbook") === true ||
    recordString(file, "mediaType") === FIXED_CHAPTARR_MEDIA_TYPE ||
    (extension !== undefined && EBOOK_FILE_EXTENSIONS.has(extension)) ||
    matchesFormatPattern(recordString(file, "format"), EBOOK_FILE_EXTENSIONS)
  );
}

function isAudiobookFile(file: ChaptarrRecord): boolean {
  const extension = recordPathExtension(file);
  return (
    recordBoolean(file, "isEbook") === false ||
    recordString(file, "mediaType") === "audiobook" ||
    (extension !== undefined && AUDIOBOOK_FILE_EXTENSIONS.has(extension)) ||
    matchesFormatPattern(recordString(file, "format"), AUDIOBOOK_FILE_EXTENSIONS)
  );
}

function analyzeBookMedia(book: ChaptarrRecord): BookMediaAnalysis {
  const editions = recordArray(book, "editions");
  const files = [
    ...recordArray(book, "bookFiles"),
    ...recordArray(book, "files"),
  ];
  const ebookEditions = editions.filter(isEbookEdition);
  const audiobookEditions = editions.filter(isAudiobookEdition);
  const ebookFiles = files.filter(isEbookFile);
  const audiobookFiles = files.filter(isAudiobookFile);
  const mediaType = recordString(book, "mediaType");
  const lastSelectedMediaType = recordString(book, "lastSelectedMediaType");
  const explicitEbook =
    mediaType === FIXED_CHAPTARR_MEDIA_TYPE ||
    lastSelectedMediaType === FIXED_CHAPTARR_MEDIA_TYPE ||
    book.ebookMonitored === true ||
    ebookEditions.length > 0 ||
    ebookFiles.length > 0;
  const explicitAudiobook =
    mediaType === "audiobook" ||
    lastSelectedMediaType === "audiobook" ||
    book.audiobookMonitored === true ||
    (editions.length > 0 && ebookEditions.length === 0 && audiobookEditions.length > 0) ||
    (files.length > 0 && ebookFiles.length === 0 && audiobookFiles.length > 0);
  const eligible = explicitEbook || (!explicitAudiobook && editions.length === 0 && files.length === 0);

  return {
    eligible,
    ebookEditions,
    ebookFiles,
    ebookFormats: sortStrings(ebookEditions.flatMap((edition) => {
      const format = recordString(edition, "format");
      return format ? [format] : [];
    })),
    ebookFileExtensions: sortStrings(ebookFiles.flatMap((file) => {
      const extension = recordPathExtension(file);
      return extension ? [extension] : [];
    })),
  };
}

function summarizeChaptarrCommandResult(data: unknown): unknown {
  const record = data && typeof data === "object" && !Array.isArray(data) ? cloneRecord(data) : undefined;
  if (!record) {
    return data;
  }

  return {
    id: recordIntegerLike(record, "id"),
    name: recordString(record, "name"),
    status: recordString(record, "status"),
    state: recordString(record, "state"),
    body: recordString(record, "body"),
    message: recordString(record, "message"),
  };
}

function summarizeChaptarrAuthorRecord(data: unknown): unknown {
  const record = data && typeof data === "object" && !Array.isArray(data) ? cloneRecord(data) : undefined;
  if (!record) {
    return data;
  }

  return {
    id: recordIntegerLike(record, "id") ?? recordIntegerLike(record, "authorId") ?? recordIntegerLike(record, "localAuthorId"),
    author_name: recordString(record, "authorName"),
    foreign_author_id: recordString(record, "foreignAuthorId"),
    monitored: record.monitored === true,
    path: recordString(record, "path"),
    title_slug: recordString(record, "titleSlug"),
    last_selected_media_type: recordString(record, "lastSelectedMediaType"),
  };
}

function summarizeChaptarrBookRecord(data: unknown): unknown {
  const record = data && typeof data === "object" && !Array.isArray(data) ? cloneRecord(data) : undefined;
  if (!record) {
    return data;
  }

  const author = recordObject(record, "author");
  const editions = recordArray(record, "editions");
  const media = analyzeBookMedia(record);
  return {
    id: recordIntegerLike(record, "id") ?? recordIntegerLike(record, "bookId") ?? recordIntegerLike(record, "localBookId"),
    title: recordString(record, "title"),
    author_name: recordString(author ?? {}, "authorName"),
    author_id: recordIntegerLike(record, "authorId") ?? recordIntegerLike(author ?? {}, "id"),
    foreign_book_id: recordString(record, "foreignBookId"),
    foreign_edition_id: recordString(record, "foreignEditionId"),
    hardcover_book_id: recordString(record, "hardcoverBookId"),
    monitored: record.monitored === true || record.ebookMonitored === true,
    media_type: recordString(record, "mediaType"),
    path: recordString(record, "path"),
    has_files: record.hasFiles === true,
    edition_count: editions.length > 0 ? editions.length : undefined,
    ebook_edition_count: editions.filter((edition) => edition.isEbook === true).length || undefined,
    ...(media.ebookFormats.length > 0 ? { ebook_formats: media.ebookFormats } : {}),
    ...(media.ebookFileExtensions.length > 0 ? { ebook_file_extensions: media.ebookFileExtensions } : {}),
    ...((media.ebookEditions.length + media.ebookFiles.length) > 0
      ? { ebook_option_count: media.ebookEditions.length + media.ebookFiles.length }
      : {}),
  };
}

function summarizeChaptarrSeriesRecord(data: unknown): unknown {
  const record = data && typeof data === "object" && !Array.isArray(data) ? cloneRecord(data) : undefined;
  if (!record) {
    return data;
  }

  const author = recordObject(record, "author");
  return {
    id: recordIntegerLike(record, "id") ?? recordIntegerLike(record, "seriesId"),
    title: recordString(record, "title"),
    author_name: recordString(author ?? {}, "authorName"),
    monitored: record.monitored === true,
    foreign_series_id: recordString(record, "foreignSeriesId"),
  };
}

function summarizeChaptarrAuthorList(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items: ChaptarrRecord[] = data.flatMap((entry) => {
    const summary = summarizeChaptarrAuthorRecord(entry);
    return summary && typeof summary === "object" && !Array.isArray(summary) ? [summary as ChaptarrRecord] : [];
  });

  return {
    total_authors: items.length,
    author_names: sortStrings(items.flatMap((item) => typeof item.author_name === "string" ? [item.author_name] : [])),
    items,
  };
}

function summarizeChaptarrBookList(
  data: unknown,
  options?: { signalNoEbookMatch?: boolean; message?: string }
): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const rawBookCount = data.filter((entry) => entry && typeof entry === "object" && !Array.isArray(entry)).length;
  const items: ChaptarrRecord[] = data.flatMap((entry) => {
    const record = entry && typeof entry === "object" && !Array.isArray(entry) ? cloneRecord(entry) : undefined;
    if (!record || !analyzeBookMedia(record).eligible) {
      return [];
    }
    const summary = summarizeChaptarrBookRecord(record);
    return summary && typeof summary === "object" && !Array.isArray(summary) ? [summary as ChaptarrRecord] : [];
  });

  return {
    ...(rawBookCount !== items.length ? { raw_book_count: rawBookCount } : {}),
    total_books: items.length,
    titles: sortStrings(items.flatMap((item) => typeof item.title === "string" ? [item.title] : [])),
    ...(options?.signalNoEbookMatch && rawBookCount > 0 && items.length === 0
      ? {
          no_ebook_match: true,
          message: options.message ?? "No ebook-capable book matches were found.",
        }
      : {}),
    items,
  };
}

function summarizeChaptarrSeriesList(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items: ChaptarrRecord[] = data.flatMap((entry) => {
    const summary = summarizeChaptarrSeriesRecord(entry);
    return summary && typeof summary === "object" && !Array.isArray(summary) ? [summary as ChaptarrRecord] : [];
  });

  return {
    total_series: items.length,
    titles: sortStrings(items.flatMap((item) => typeof item.title === "string" ? [item.title] : [])),
    items,
  };
}

function summarizeChaptarrSearchResults(
  data: unknown,
  options?: { signalNoEbookMatch?: boolean; message?: string }
): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const bookItems: ChaptarrRecord[] = [];
  const authorItems: ChaptarrRecord[] = [];
  let rawBookCount = 0;

  for (const entry of data) {
    const record = entry && typeof entry === "object" && !Array.isArray(entry) ? cloneRecord(entry) : undefined;
    if (!record) {
      continue;
    }
    if (record.book && typeof record.book === "object" && !Array.isArray(record.book)) {
      rawBookCount += 1;
      const bookRecord = cloneRecord(record.book);
      if (analyzeBookMedia(bookRecord).eligible) {
        bookItems.push(summarizeChaptarrBookRecord(bookRecord) as ChaptarrRecord);
      }
    }
    if (record.author && typeof record.author === "object" && !Array.isArray(record.author)) {
      authorItems.push(summarizeChaptarrAuthorRecord(record.author) as ChaptarrRecord);
    }
  }

  const resultCount = authorItems.length + bookItems.length;
  return {
    ...(data.length !== resultCount ? { raw_result_count: data.length } : {}),
    ...(rawBookCount !== bookItems.length ? { raw_book_result_count: rawBookCount } : {}),
    result_count: resultCount,
    author_result_count: authorItems.length,
    book_result_count: bookItems.length,
    ...(options?.signalNoEbookMatch && rawBookCount > 0 && bookItems.length === 0
      ? {
          no_ebook_match: true,
          message: options.message ?? "No ebook-capable book matches were found.",
        }
      : {}),
    authors: authorItems,
    books: bookItems,
  };
}

function summarizeChaptarrProfileList(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items: ChaptarrRecord[] = data.flatMap((entry) => {
    const record = entry && typeof entry === "object" && !Array.isArray(entry) ? cloneRecord(entry) : undefined;
    if (!record) {
      return [];
    }
    return [{
      id: recordIntegerLike(record, "id"),
      name: recordString(record, "name"),
      cutoff: recordString(record, "cutoff"),
      upgrade_allowed: record.upgradeAllowed === true,
    }];
  });

  return {
    profile_count: items.length,
    items,
  };
}

function summarizeChaptarrRootFolders(data: unknown): unknown {
  if (!Array.isArray(data)) {
    return data;
  }

  const items: ChaptarrRecord[] = data.flatMap((entry) => {
    const record = entry && typeof entry === "object" && !Array.isArray(entry) ? cloneRecord(entry) : undefined;
    if (!record) {
      return [];
    }
    return [{
      id: recordIntegerLike(record, "id"),
      path: recordString(record, "path"),
      accessible: record.accessible === true,
      default_quality_profile_id: recordIntegerLike(record, "defaultQualityProfileId"),
      default_metadata_profile_id: recordIntegerLike(record, "defaultMetadataProfileId"),
      default_monitor_option: recordString(record, "defaultMonitorOption"),
    }];
  });

  return {
    root_folder_count: items.length,
    items,
  };
}

function sanitizePathSegment(value: string): string {
  const normalized = value
    .trim()
    .replace(/[\\/]+/g, "-")
    .replace(/\s+/g, " ")
    .replace(/[<>:"|?*\u0000-\u001F]/g, "")
    .trim();
  return normalized.length > 0 ? normalized : "Unknown";
}

function normalizeQueryText(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter((token) => token.length >= 2);
}

function queryMatchesText(query: string, text: string | undefined): boolean {
  if (!text) {
    return false;
  }
  const queryTerms = normalizeQueryText(query);
  if (queryTerms.length === 0) {
    return false;
  }
  const normalizedText = ` ${normalizeQueryText(text).join(" ")} `;
  return queryTerms.every((term) => normalizedText.includes(` ${term} `));
}

function formatBookLabel(book: ChaptarrRecord): string {
  const title = recordString(book, "title") ?? "Unknown book";
  const localBookId = recordIntegerLike(book, "localBookId") ?? recordIntegerLike(book, "id");
  return localBookId !== undefined ? `${title} [id=${localBookId}]` : title;
}

function joinRootFolder(rootFolderPath: string, leaf: string): string {
  return `${rootFolderPath.replace(/\/+$/, "")}/${sanitizePathSegment(leaf)}`;
}

function pickAuthorFolder(author: ChaptarrRecord): string {
  return (
    recordString(author, "folder") ??
    recordString(author, "titleSlug") ??
    recordString(author, "authorName") ??
    "Unknown Author"
  );
}

function selectSearchAuthorCandidate(searchResults: unknown, term: string, foreignAuthorId: string | undefined): ChaptarrRecord {
  if (!Array.isArray(searchResults)) {
    throw new Error("Chaptarr search did not return an array");
  }
  const candidates = searchResults.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return [];
    }
    const record = entry as ChaptarrRecord;
    if (!record.author || typeof record.author !== "object" || Array.isArray(record.author)) {
      return [];
    }
    return [{
      searchEntry: cloneRecord(record),
      author: cloneRecord(record.author),
    }];
  });
  const selected = chooseLookupCandidate(candidates, [
    (candidate) =>
      foreignAuthorId !== undefined &&
      (recordString(candidate.author, "foreignAuthorId") === foreignAuthorId ||
        recordString(candidate.searchEntry, "foreignId") === foreignAuthorId),
    (candidate) => queryMatchesText(term, recordString(candidate.author, "authorName")),
    () => true,
  ]);
  return selected.author;
}

function selectSearchBookCandidate(
  searchResults: unknown,
  term: string,
  foreignBookId: string | undefined,
  foreignEditionId: string | undefined
): ChaptarrRecord {
  if (!Array.isArray(searchResults)) {
    throw new Error("Chaptarr search did not return an array");
  }
  const candidates = searchResults.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return [];
    }
    const record = entry as ChaptarrRecord;
    if (!record.book || typeof record.book !== "object" || Array.isArray(record.book)) {
      return [];
    }
    return [{
      searchEntry: cloneRecord(record),
      book: cloneRecord(record.book),
      media: analyzeBookMedia(cloneRecord(record.book)),
    }];
  });
  const ebookCandidates = candidates.filter((candidate) => candidate.media.eligible);
  if (ebookCandidates.length === 0 && candidates.length > 0) {
    throw new ChaptarrToolError(
      "no_ebook_match",
      "search result did not include an ebook-capable book match",
      { term, foreign_book_id: foreignBookId, foreign_edition_id: foreignEditionId }
    );
  }
  const selected = chooseLookupCandidate(ebookCandidates, [
    (candidate) =>
      foreignBookId !== undefined &&
      (recordString(candidate.book, "foreignBookId") === foreignBookId ||
        recordString(candidate.book, "hardcoverBookId") === foreignBookId ||
        recordString(candidate.searchEntry, "foreignId") === foreignBookId),
    (candidate) =>
      foreignEditionId !== undefined &&
      candidate.media.ebookEditions.some((edition) =>
        recordString(edition, "foreignEditionId") === foreignEditionId ||
        recordString(edition, "titleSlug") === foreignEditionId),
    (candidate) =>
      candidate.media.ebookEditions.length > 0 || candidate.media.ebookFiles.length > 0,
    (candidate) =>
      queryMatchesText(
        term,
        `${recordString(candidate.book, "title") ?? ""} ${
          typeof candidate.book.author === "object" &&
          candidate.book.author &&
          !Array.isArray(candidate.book.author)
            ? recordString(candidate.book.author as ChaptarrRecord, "authorName") ?? ""
            : ""
        }`
      ),
    () => true,
  ]);
  return selected.book;
}

function pickPreferredBookEditions(book: ChaptarrRecord, foreignEditionId: string | undefined, monitored: boolean): ChaptarrRecord[] {
  const allEditions = recordArray(book, "editions");
  const ebookEditions = allEditions.filter(isEbookEdition);
  const source = ebookEditions.length > 0 ? ebookEditions : allEditions;
  if (source.length === 0) {
    return [];
  }
  const selected = chooseLookupCandidate(source, [
    (edition) =>
      foreignEditionId !== undefined &&
      (recordString(edition, "foreignEditionId") === foreignEditionId ||
        recordString(edition, "titleSlug") === foreignEditionId),
    () => true,
  ]);
  return source.map((edition) => ({
    ...edition,
    monitored:
      monitored && (
        edition === selected ||
        (foreignEditionId === undefined && source.length === 1) ||
        (ebookEditions.length > 0 && selected.isEbook === true && edition.isEbook === true)
      ),
    isEbook: edition.isEbook === true,
  }));
}

async function triggerChaptarrCommand(
  config: ChaptarrCliContext["config"],
  requestBody: Record<string, unknown>
) {
  return await chaptarrRequestJson(config, "POST", "/api/v1/command", undefined, requestBody);
}

async function fetchAuthorRecord(
  config: ChaptarrCliContext["config"],
  authorId: number
): Promise<ChaptarrRecord> {
  const data = await chaptarrRequestJson(config, "GET", `/api/v1/author/${authorId}`);
  return cloneRecord(data);
}

async function updateAuthorRecord(
  config: ChaptarrCliContext["config"],
  authorId: number,
  author: ChaptarrRecord
) {
  return await chaptarrRequestJson(config, "PUT", `/api/v1/author/${authorId}`, undefined, author);
}

async function fetchBookRecord(
  config: ChaptarrCliContext["config"],
  bookId: number
): Promise<ChaptarrRecord> {
  const data = await chaptarrRequestJson(config, "GET", `/api/v1/book/${bookId}`);
  return cloneRecord(data);
}

async function updateBookRecord(
  config: ChaptarrCliContext["config"],
  bookId: number,
  book: ChaptarrRecord
) {
  return await chaptarrRequestJson(config, "PUT", `/api/v1/book/${bookId}`, undefined, book);
}

async function updateBookMonitorState(
  config: ChaptarrCliContext["config"],
  bookIds: number[],
  monitored: boolean
) {
  return await chaptarrRequestJson(config, "PUT", "/api/v1/book/monitor", undefined, {
    bookIds,
    monitored,
  });
}

function buildEbookAuthorRecord(author: ChaptarrRecord, monitored: boolean): ChaptarrRecord {
  const repaired = cloneRecord(author);
  repaired.path = joinRootFolder(FIXED_CHAPTARR_ROOT_FOLDER_PATH, pickAuthorFolder(repaired));
  repaired.monitored = monitored;
  repaired.lastSelectedMediaType = FIXED_CHAPTARR_MEDIA_TYPE;
  repaired.ebookQualityProfileId = DEFAULT_CHAPTARR_EBOOK_QUALITY_PROFILE_ID;
  repaired.metadataProfileId = DEFAULT_CHAPTARR_EBOOK_METADATA_PROFILE_ID;
  repaired.ebookMetadataProfileId = DEFAULT_CHAPTARR_EBOOK_METADATA_PROFILE_ID;
  repaired.audiobookMonitorExisting = 0;
  repaired.audiobookMonitorFuture = false;
  repaired.ebookMonitorExisting = 1;
  repaired.ebookMonitorFuture = true;
  repaired.monitorNewItems =
    recordString(repaired, "monitorNewItems") ?? DEFAULT_MONITOR_NEW_ITEMS;
  if (!Array.isArray(repaired.tags)) {
    repaired.tags = [];
  }
  if (!Array.isArray(repaired.audiobookTags)) {
    repaired.audiobookTags = [];
  }
  if (!Array.isArray(repaired.ebookTags)) {
    repaired.ebookTags = [];
  }
  return repaired;
}

function buildEbookBookRecord(
  book: ChaptarrRecord,
  repairedAuthor: ChaptarrRecord,
  foreignEditionId: string | undefined,
  monitored: boolean
): ChaptarrRecord {
  const repaired = cloneRecord(book);
  repaired.monitored = monitored;
  repaired.ebookMonitored = monitored;
  repaired.audiobookMonitored = false;
  repaired.mediaType = FIXED_CHAPTARR_MEDIA_TYPE;
  repaired.lastSelectedMediaType = FIXED_CHAPTARR_MEDIA_TYPE;
  repaired.anyEditionOk = false;
  repaired.addOptions = {
    ...(typeof repaired.addOptions === "object" && repaired.addOptions ? (repaired.addOptions as Record<string, unknown>) : {}),
    addType: "manual",
    searchForNewBook: true,
  };
  const editions = pickPreferredBookEditions(repaired, foreignEditionId, monitored);
  if (editions.length > 0) {
    repaired.editions = editions;
  }
  repaired.author = repairedAuthor;
  return repaired;
}

function resolveDeleteBookTarget(
  books: unknown,
  payload: ChaptarrInvocation
): ChaptarrRecord {
  if (!Array.isArray(books)) {
    throw new ChaptarrToolError("invalid_response", "Chaptarr book list did not return an array");
  }

  const candidates = books
    .filter((entry) => entry && typeof entry === "object" && !Array.isArray(entry))
    .map((entry) => cloneRecord(entry));
  const bookId = optionalInteger(payload, "book_id");
  if (bookId !== undefined) {
    const directMatch = candidates.find((entry) => recordIntegerLike(entry, "id") === bookId || recordIntegerLike(entry, "localBookId") === bookId);
    if (!directMatch) {
      throw new ChaptarrToolError("lookup_no_match", `no tracked Chaptarr book matched id=${bookId}`);
    }
    return directMatch;
  }

  const foreignBookId = optionalString(payload, "foreign_book_id");
  if (foreignBookId) {
    const foreignMatches = candidates.filter((entry) =>
      recordString(entry, "foreignBookId") === foreignBookId ||
      recordString(entry, "hardcoverBookId") === foreignBookId
    );
    if (foreignMatches.length === 1) {
      return foreignMatches[0];
    }
    if (foreignMatches.length > 1) {
      throw new ChaptarrToolError("ambiguous_match", "multiple tracked Chaptarr books matched the requested foreign_book_id", {
        foreign_book_id: foreignBookId,
        matches: foreignMatches.slice(0, 10).map(formatBookLabel),
      });
    }
  }

  const term = requireString(payload, "term", "term or book_id is required for delete_book");
  const matches = candidates.filter((entry) => {
    const author = recordObject(entry, "author");
    return queryMatchesText(term, recordString(entry, "title")) ||
      queryMatchesText(term, recordString(entry, "authorTitle")) ||
      queryMatchesText(term, recordString(entry, "seriesTitle")) ||
      queryMatchesText(term, `${recordString(entry, "title") ?? ""} ${recordString(author ?? {}, "authorName") ?? ""}`);
  });
  if (matches.length === 1) {
    return matches[0];
  }
  if (matches.length > 1) {
    throw new ChaptarrToolError("ambiguous_match", "multiple tracked Chaptarr books matched the requested title", {
      term,
      matches: matches.slice(0, 10).map(formatBookLabel),
    });
  }

  throw new ChaptarrToolError("lookup_no_match", `no tracked Chaptarr book matched '${term}'`);
}

async function augmentMetadataFailureIfPresent(
  config: ChaptarrCliContext["config"],
  action: string,
  path: string,
  error: unknown
): Promise<never> {
  if (
    !(error instanceof ChaptarrToolError) ||
    error.kind !== "http_error" ||
    error.details?.status !== 503 ||
    typeof error.details?.body !== "string" ||
    !error.details.body.includes("Failed to get author info from V5 API")
  ) {
    throw error;
  }

  try {
    const logPage = await chaptarrRequestJson(config, "GET", "/api/v1/log", {
      page: 1,
      pageSize: 10,
      sortKey: "Id",
      sortDirection: "descending",
    });
    if (logPage && typeof logPage === "object" && !Array.isArray(logPage) && Array.isArray((logPage as ChaptarrRecord).records)) {
      const record = ((logPage as ChaptarrRecord).records as unknown[]).find((entry) =>
        entry &&
        typeof entry === "object" &&
        !Array.isArray(entry) &&
        typeof (entry as ChaptarrRecord).exception === "string" &&
        ((entry as ChaptarrRecord).exception as string).includes("Failed to get author info from V5 API")
      ) as ChaptarrRecord | undefined;
      const exceptionText = typeof record?.exception === "string" ? record.exception : "";
      const upstreamMatch = /HTTP request failed: \[(\d+):[^\]]*\] \[GET\] at \[(https:\/\/[^\]]+)\]/.exec(exceptionText);
      if (upstreamMatch) {
        throw new ChaptarrToolError(
          "upstream_metadata_failure",
          `${action} failed because Chaptarr could not reach its upstream V5 metadata service`,
          {
            path,
            status: error.details.status,
            upstream_status: Number.parseInt(upstreamMatch[1], 10),
            upstream_url: upstreamMatch[2],
            log_time: typeof record?.time === "string" ? record.time : undefined,
            service_message: "Failed to get author info from V5 API",
          }
        );
      }
    }
  } catch (diagnosticError) {
    if (diagnosticError instanceof ChaptarrToolError && diagnosticError.kind === "upstream_metadata_failure") {
      throw diagnosticError;
    }
  }

  throw error;
}

function authorScopedQuery(payload: ChaptarrInvocation) {
  return {
    authorId: optionalInteger(payload, "author_id"),
  };
}

async function fetchMediaTypedData(
  context: ChaptarrCliContext,
  action: string,
  path: string
) {
  const mediaTypes = resolveMediaTypes(context.payload);
  const queryValue = mediaTypes.length === 1 ? mediaTypes[0] : mediaTypes;
  if (mediaTypes.length === 1) {
    const data = await chaptarrRequestJson(context.config, "GET", path, { mediaType: mediaTypes[0] });
    return successEnvelope(action, context.config.baseUrl, "GET", path, { mediaType: queryValue }, summarizeChaptarrProfileList(data));
  }

  const data: Record<string, unknown> = {};
  for (const mediaType of mediaTypes) {
    data[mediaType] = summarizeChaptarrProfileList(await chaptarrRequestJson(context.config, "GET", path, { mediaType }));
  }
  return successEnvelope(action, context.config.baseUrl, "GET", path, { mediaType: queryValue }, data);
}

export async function handleChaptarr(context: ChaptarrCliContext) {
  const { payload, config } = context;
  const action = String(payload.action);
  const resolvedAction = canonicalizeChaptarrAction(action);

  if (resolvedAction === "system_status") {
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/system/status");
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/system/status", undefined, data);
  }
  if (resolvedAction === "health") {
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/health");
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/health", undefined, data);
  }
  if (resolvedAction === "queue_status") {
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/queue/status");
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/queue/status", undefined, data);
  }
  if (resolvedAction === "root_folders") {
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/rootfolder");
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/rootfolder", undefined, summarizeChaptarrRootFolders(data));
  }
  if (resolvedAction === "quality_profiles") {
    return fetchMediaTypedData(context, action, "/api/v1/qualityprofile");
  }
  if (resolvedAction === "metadata_profiles") {
    return fetchMediaTypedData(context, action, "/api/v1/metadataprofile");
  }
  if (resolvedAction === "list_authors") {
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/author");
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/author", undefined, summarizeChaptarrAuthorList(data));
  }
  if (resolvedAction === "search") {
    const term = requireString(payload, "term", "term is required for search");
    const provider = optionalString(payload, "provider");
    const query = provider ? { term, provider } : { term };
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/search", query);
    return successEnvelope(
      action,
      config.baseUrl,
      "GET",
      "/api/v1/search",
      query,
      summarizeChaptarrSearchResults(data, { signalNoEbookMatch: true })
    );
  }
  if (resolvedAction === "author_lookup") {
    const term = requireString(payload, "term", "term is required for author_lookup");
    const query = { term };
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/author/lookup", query);
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/author/lookup", query, summarizeChaptarrAuthorList(data));
  }
  if (resolvedAction === "book_lookup") {
    const term = requireString(payload, "term", "term is required for book_lookup");
    const query = { term };
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/book/lookup", query);
    return successEnvelope(
      action,
      config.baseUrl,
      "GET",
      "/api/v1/book/lookup",
      query,
      summarizeChaptarrBookList(data, { signalNoEbookMatch: true })
    );
  }
  if (resolvedAction === "list_books") {
    const query = authorScopedQuery(payload);
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/book", query);
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/book", query, summarizeChaptarrBookList(data));
  }
  if (resolvedAction === "list_series") {
    const query = authorScopedQuery(payload);
    const data = await chaptarrRequestJson(config, "GET", "/api/v1/series", query);
    return successEnvelope(action, config.baseUrl, "GET", "/api/v1/series", query, summarizeChaptarrSeriesList(data));
  }
  if (resolvedAction === "add_author" || resolvedAction === "download_author") {
    const isDownloadAction = resolvedAction === "download_author";
    const term = requireString(payload, "term", "term is required for add_author");
    const qualityProfileId = DEFAULT_CHAPTARR_EBOOK_QUALITY_PROFILE_ID;
    const metadataProfileId = DEFAULT_CHAPTARR_EBOOK_METADATA_PROFILE_ID;
    const foreignAuthorId = optionalString(payload, "foreign_author_id");
    const monitored = true;
    const rootFolderPath = FIXED_CHAPTARR_ROOT_FOLDER_PATH;
    const searchResults = await chaptarrRequestJson(config, "GET", "/api/v1/search", {
      term,
      provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER,
    });
    const candidate = selectSearchAuthorCandidate(searchResults, term, foreignAuthorId);
      const existingAuthorId = resolveTrackedAuthorId(candidate);
    if (existingAuthorId !== undefined) {
      const existingAuthor = await fetchAuthorRecord(config, existingAuthorId);
      const repairedAuthor = buildEbookAuthorRecord(existingAuthor, monitored);
      const updatedAuthor = await updateAuthorRecord(config, existingAuthorId, repairedAuthor);
      const command = await triggerChaptarrCommand(config, {
        name: "AuthorSearch",
        authorId: existingAuthorId,
      });
      return successEnvelope(action, config.baseUrl, "POST", "/api/v1/command", { term, provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER }, {
        mode: "existing_author_search",
        existing_author_id: existingAuthorId,
        author: summarizeChaptarrAuthorRecord(updatedAuthor),
        command: summarizeChaptarrCommandResult(command),
      });
    }
    const authorPath = joinRootFolder(rootFolderPath, pickAuthorFolder(candidate));

    const requestBody: Record<string, unknown> = {
      ...candidate,
      rootFolderPath,
      path: authorPath,
      qualityProfileId,
      ebookQualityProfileId: qualityProfileId,
      metadataProfileId,
      ebookMetadataProfileId: metadataProfileId,
      monitored,
      monitorNewItems: isDownloadAction
        ? DEFAULT_MONITOR_NEW_ITEMS
        : optionalString(payload, "monitor_new_items") ?? candidate.monitorNewItems ?? DEFAULT_MONITOR_NEW_ITEMS,
      tags: optionalIntegerArray(payload, "tags") ?? candidate.tags ?? [],
      lastSelectedMediaType: FIXED_CHAPTARR_MEDIA_TYPE,
      addOptions: {
        ...(typeof candidate.addOptions === "object" && candidate.addOptions ? (candidate.addOptions as Record<string, unknown>) : {}),
        monitor: isDownloadAction
          ? DEFAULT_ADD_AUTHOR_MONITOR
          : optionalString(payload, "monitor") ?? DEFAULT_ADD_AUTHOR_MONITOR,
        monitored,
        searchForMissingBooks: true,
      }
    };
    delete requestBody.id;
    try {
      const data = await chaptarrRequestJson(config, "POST", "/api/v1/author", undefined, requestBody);
      const addedAuthor = cloneRecord(data);
      const addedAuthorId = resolveTrackedAuthorId(addedAuthor);
      if (addedAuthorId === undefined) {
        throw new ChaptarrToolError("invalid_response", "newly added Chaptarr author did not include an id for AuthorSearch");
      }
      const command = await triggerChaptarrCommand(config, {
        name: "AuthorSearch",
        authorId: addedAuthorId,
      });
      return successEnvelope(action, config.baseUrl, "POST", "/api/v1/author", { term, provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER }, {
        mode: "new_author_search",
        author: summarizeChaptarrAuthorRecord(addedAuthor),
        command: summarizeChaptarrCommandResult(command),
      });
    } catch (error) {
      return await augmentMetadataFailureIfPresent(config, action, "/api/v1/author", error);
    }
  }
  if (resolvedAction === "add_book" || resolvedAction === "download_book") {
    const isDownloadAction = resolvedAction === "download_book";
    const term = requireString(payload, "term", "term is required for add_book");
    const qualityProfileId = DEFAULT_CHAPTARR_EBOOK_QUALITY_PROFILE_ID;
    const metadataProfileId = DEFAULT_CHAPTARR_EBOOK_METADATA_PROFILE_ID;
    const foreignBookId = optionalString(payload, "foreign_book_id");
    const foreignEditionId = optionalString(payload, "foreign_edition_id");
    const monitored = true;
    const rootFolderPath = FIXED_CHAPTARR_ROOT_FOLDER_PATH;
    const searchResults = await chaptarrRequestJson(config, "GET", "/api/v1/search", {
      term,
      provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER,
    });
    const candidate = selectSearchBookCandidate(searchResults, term, foreignBookId, foreignEditionId);
    const existingBookId = resolveTrackedBookId(candidate);
    if (existingBookId !== undefined) {
      const existingBook = await fetchBookRecord(config, existingBookId);
      const existingBookAuthor =
        recordObject(existingBook, "author") ??
        recordObject(candidate, "author") ??
        {};
      const existingAuthorId =
        recordInteger(existingBook, "authorId") ??
        recordInteger(existingBookAuthor, "id") ??
        recordInteger(existingBookAuthor, "authorId");
      if (existingAuthorId === undefined) {
        throw new ChaptarrToolError("lookup_no_match", "existing tracked book did not include an author id");
      }
      const existingAuthor = await fetchAuthorRecord(config, existingAuthorId);
      const repairedAuthor = buildEbookAuthorRecord(existingAuthor, true);
      const updatedAuthor = await updateAuthorRecord(config, existingAuthorId, repairedAuthor);
      const repairedBook = buildEbookBookRecord(existingBook, cloneRecord(updatedAuthor), foreignEditionId, true);
      const updatedBook = await updateBookRecord(config, existingBookId, repairedBook);
      const monitorResult = await updateBookMonitorState(config, [existingBookId], true);
      const command = await triggerChaptarrCommand(config, {
        name: "BookSearch",
        bookIds: [existingBookId],
      });
      return successEnvelope(action, config.baseUrl, "POST", "/api/v1/command", { term, provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER }, {
        mode: "existing_book_search",
        existing_book_id: existingBookId,
        author: summarizeChaptarrAuthorRecord(updatedAuthor),
        book: summarizeChaptarrBookRecord(updatedBook),
        monitor_result: summarizeChaptarrBookList(monitorResult),
        command: summarizeChaptarrCommandResult(command),
      });
    }
    const author =
      typeof candidate.author === "object" && candidate.author && !Array.isArray(candidate.author)
        ? { ...(candidate.author as Record<string, unknown>) }
        : {};
    if (Object.keys(author).length === 0) {
      throw new ChaptarrToolError("lookup_no_match", "search result did not include an author for add_book");
    }
    const authorPath = joinRootFolder(rootFolderPath, pickAuthorFolder(author));
    const editions = pickPreferredBookEditions(candidate, foreignEditionId, monitored);
    const requestBody: Record<string, unknown> = {
      ...candidate,
      monitored,
      rootFolderPath,
      qualityProfileId,
      ebookQualityProfileId: qualityProfileId,
      metadataProfileId,
      ebookMetadataProfileId: metadataProfileId,
      mediaType: FIXED_CHAPTARR_MEDIA_TYPE,
      lastSelectedMediaType: FIXED_CHAPTARR_MEDIA_TYPE,
      ebookMonitored: monitored,
      audiobookMonitored: false,
      anyEditionOk: isDownloadAction ? false : optionalBoolean(payload, "any_edition_ok") ?? candidate.anyEditionOk ?? false,
      ...(editions.length > 0 ? { editions } : {}),
      author: {
        ...author,
        rootFolderPath,
        path: authorPath,
        qualityProfileId,
        ebookQualityProfileId: qualityProfileId,
        metadataProfileId,
        ebookMetadataProfileId: metadataProfileId,
        monitored,
        lastSelectedMediaType: FIXED_CHAPTARR_MEDIA_TYPE,
      },
      addOptions: {
        ...(typeof candidate.addOptions === "object" && candidate.addOptions ? (candidate.addOptions as Record<string, unknown>) : {}),
        searchForNewBook: true,
      }
    };
    delete requestBody.id;
    delete requestBody.localBookId;
    delete requestBody.hasFiles;
    delete requestBody.authorId;
    delete requestBody.remoteCover;
    try {
      const data = await chaptarrRequestJson(config, "POST", "/api/v1/book", undefined, requestBody);
      const addedBook = cloneRecord(data);
      const addedBookId = resolveTrackedBookId(addedBook);
      if (addedBookId === undefined) {
        throw new ChaptarrToolError("invalid_response", "newly added Chaptarr book did not include an id for BookSearch");
      }
      const command = await triggerChaptarrCommand(config, {
        name: "BookSearch",
        bookIds: [addedBookId],
      });
      return successEnvelope(action, config.baseUrl, "POST", "/api/v1/book", { term, provider: DEFAULT_CHAPTARR_SEARCH_PROVIDER }, {
        mode: "new_book_search",
        book: summarizeChaptarrBookRecord(addedBook),
        command: summarizeChaptarrCommandResult(command),
      });
    } catch (error) {
      return await augmentMetadataFailureIfPresent(config, action, "/api/v1/book", error);
    }
  }
  if (resolvedAction === "delete_book") {
    const books = await chaptarrRequestJson(config, "GET", "/api/v1/book");
    const target = resolveDeleteBookTarget(books, payload);
    const bookId = recordIntegerLike(target, "id") ?? recordIntegerLike(target, "localBookId");
    if (bookId === undefined) {
      throw new ChaptarrToolError("lookup_no_match", "matched Chaptarr book did not include an id");
    }
    const query = {
      deleteFiles: optionalBoolean(payload, "delete_files") ?? true,
      addImportListExclusion: optionalBoolean(payload, "add_import_list_exclusion") ?? false,
    };
    await chaptarrRequestJson(config, "DELETE", `/api/v1/book/${bookId}`, query);
    return successEnvelope(action, config.baseUrl, "DELETE", `/api/v1/book/${bookId}`, query, {
      mode: "tracked_book_delete",
      deleted_book: summarizeChaptarrBookRecord(target),
    });
  }

  throw new Error(`unsupported Chaptarr action: ${action}`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  void runChaptarrCli(process.argv.slice(2), handleChaptarr);
}
