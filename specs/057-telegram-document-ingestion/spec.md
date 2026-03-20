# Feature Specification: Telegram Document Ingestion And Supermemory Linking

**Feature Branch**: `057-telegram-document-ingestion (spec-only on current 049-host-build-bringup worktree)`  
**Created**: 2026-03-19  
**Status**: Draft  
**Input**: User description: "modify the telegram middleware to be able to parse PDF/DOC files sent via telegram message. we want to remove all markup/images/etc and just get the text, in both cases. files themselves should also be saved in supermemory and linked with their extracted text."

## User Scenarios & Testing

### User Story 1 - Telegram Documents Become Plain-Text Chat Turns (Priority: P1)

As a Telegram user, I need PDF and Word documents sent to the bridge to be
downloaded, converted into plain text, and forwarded to the Vicuña runtime as
user-readable text so the model can reason over the document contents instead
of seeing an unsupported file error.

**Why this priority**: The primary user-facing failure today is that documents
are rejected outright. Converting them into plain text is the minimum viable
behavior change.

**Independent Test**: A Telegram `message.document` update for a supported PDF,
DOC, or DOCX file produces a bounded plain-text transcript entry without
embedded markup, image placeholders, or binary content, and the bridge stops
replying with the current plain-text-only rejection.

**Acceptance Scenarios**:

1. **Given** a Telegram message containing a PDF document, **When** the bridge
   processes the update, **Then** it downloads the file, extracts plain text,
   appends that text to the chat transcript, and forwards the extracted text to
   the Vicuña runtime.
2. **Given** a Telegram message containing a DOC or DOCX document, **When** the
   bridge processes the update, **Then** it converts the file to plain text and
   forwards only text content rather than markup or formatting metadata.
3. **Given** a supported document with little or no extractable text, **When**
   extraction completes, **Then** the bridge returns a clear operator-visible
   failure to the Telegram chat instead of silently sending an empty turn.

### User Story 2 - Raw Files And Extracted Text Are Persisted In Supermemory (Priority: P1)

As a runtime operator, I need every supported Telegram document to be stored in
Supermemory both as the raw uploaded file and as a linked extracted-text
document so the original artifact and the text the model saw remain durable and
traceable.

**Why this priority**: The user explicitly wants both the file and its text in
Supermemory, and the pair must stay linked for later retrieval or inspection.

**Independent Test**: Processing a supported Telegram document issues a
Supermemory file upload for the source file, persists a second text document
for the extracted plain text, and assigns both a shared linkage key plus stable
Telegram metadata.

**Acceptance Scenarios**:

1. **Given** a supported Telegram document and a configured Supermemory API
   key, **When** the bridge completes ingestion, **Then** the raw file is
   uploaded to Supermemory as a file-backed document and the extracted text is
   added as a separate text-backed document.
2. **Given** the raw file and extracted text are stored, **When** their stored
   metadata is inspected, **Then** both records share a common linkage key and
   Telegram source identifiers that relate the pair to the originating chat and
   message.
3. **Given** the raw file upload succeeds but extracted-text persistence fails,
   **When** the bridge reports the failure, **Then** the operator-visible error
   clearly states that Supermemory persistence was only partially completed.

### User Story 3 - The Bridge Remains Explicit About Supported Formats And Host Requirements (Priority: P2)

As a maintainer, I need the document-ingestion path to expose explicit format
rules, host dependencies, and error behavior so the bridge remains inspectable
and debuggable rather than relying on hidden heuristics.

**Why this priority**: This feature crosses Telegram, local extraction tools,
and Supermemory. Clear runtime policy is necessary to keep the bridge
maintainable.

**Independent Test**: Unit tests cover format detection, text sanitization,
metadata/link generation, and failure paths for unsupported types or missing
host tooling, while docs explain the required environment and extraction path.

**Acceptance Scenarios**:

1. **Given** an unsupported Telegram document type, **When** the bridge
   receives it, **Then** it returns a clear unsupported-format reply instead of
   attempting opaque best-effort parsing.
2. **Given** DOC or DOCX extraction is requested on a host without the required
   conversion tool, **When** the bridge attempts extraction, **Then** it fails
   with a direct host-requirement error.
3. **Given** a supported document is extracted, **When** the plain text is
   normalized, **Then** formatting noise is collapsed into bounded plain text
   before being appended to the transcript.

## Edge Cases

- Telegram sends a `document` with a supported extension but a missing or
  misleading MIME type.
- The Telegram file metadata exists but `getFile` or file download fails.
- The extracted text is empty, whitespace only, or larger than the bridge's
  practical prompt budget.
- The source document contains images, tables, headers, or formatting artifacts
  that should not survive as markup in the extracted text.
- DOC/DOCX extraction is attempted on a host where `textutil` is unavailable.
- The Supermemory API key is absent, invalid, or upload/add requests partially
  fail.
- The same file is resent in the same or different chat and must still produce
  a stable linkage key without corrupting the local transcript.

## Requirements

### Functional Requirements

- **FR-001**: The Telegram bridge MUST accept inbound Telegram
  `message.document` updates for supported PDF, DOC, and DOCX files.
- **FR-002**: The bridge MUST continue to reject unsupported document formats
  with a clear user-visible reply.
- **FR-003**: The bridge MUST download supported Telegram files using the Bot
  API `getFile` flow and the returned file path.
- **FR-004**: PDF ingestion MUST extract plain text only, without retaining PDF
  markup, binary payloads, or image placeholders in the forwarded transcript.
- **FR-005**: DOC and DOCX ingestion MUST extract plain text only, without
  retaining Word markup, styling, or embedded images in the forwarded
  transcript.
- **FR-006**: The bridge MUST normalize extracted document text into bounded
  plain text suitable for transcript storage and runtime forwarding.
- **FR-007**: The bridge MUST append the extracted document text to the local
  per-chat transcript and send that text to the Vicuña runtime as the user turn
  for the originating Telegram message.
- **FR-008**: The bridge MUST persist the original uploaded file to
  Supermemory using the configured Supermemory API key.
- **FR-009**: The bridge MUST persist the extracted plain text to Supermemory
  as a separate text document.
- **FR-010**: The raw-file document and extracted-text document MUST share a
  deterministic linkage key plus source metadata that identifies the Telegram
  chat, message, file id, and file-unique id.
- **FR-011**: The bridge MUST assign a stable Supermemory container grouping
  for Telegram-originated documents so the raw file and extracted text can be
  queried together.
- **FR-012**: If Supermemory persistence fails fully or partially, the bridge
  MUST report that failure back to the originating Telegram chat and MUST NOT
  claim success.
- **FR-013**: The document-ingestion path MUST keep its extraction policy,
  supported-type rules, and linkage metadata explicit in bridge code and
  documentation.
- **FR-014**: The change MUST add targeted Node-based tests and bridge
  documentation covering format support, extraction normalization, Telegram file
  download flow, and Supermemory persistence/linking behavior.

### Key Entities

- **Telegram Document Descriptor**: Normalized metadata derived from
  `message.document`, including identifiers, file name, MIME type, size, and
  detected logical type.
- **Extracted Document Text**: The plain-text-only normalized content derived
  from a PDF, DOC, or DOCX file before it is appended to the chat transcript.
- **Supermemory Linkage Record**: The shared metadata surface that links a raw
  file document and its extracted-text document through a deterministic key and
  Telegram provenance fields.
- **Telegram Document Ingestion Result**: The explicit bridge result for one
  inbound document, including extracted text, raw upload id, text document id,
  and failure details when any stage fails.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Targeted bridge tests show that supported PDF, DOC, and DOCX
  document messages are converted into normalized plain-text transcript turns.
- **SC-002**: Targeted bridge tests show that unsupported document formats
  return a direct unsupported-format reply without reaching the runtime.
- **SC-003**: Targeted bridge tests show that raw file uploads and extracted
  text persistence use shared linkage metadata and Telegram provenance in
  Supermemory-bound requests.
- **SC-004**: Bridge documentation and validation steps describe the required
  `SUPERMEMORY_API_KEY` configuration and the DOC/DOCX host dependency needed
  for extraction.
