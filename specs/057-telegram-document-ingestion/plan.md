# Implementation Plan: Telegram Document Ingestion And Supermemory Linking

**Branch**: `057-telegram-document-ingestion (spec-only on current 049-host-build-bringup worktree)` | **Date**: 2026-03-19 | **Spec**: [/Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/spec.md](/Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/spec.md)
**Input**: Feature specification from [/Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/spec.md](/Users/tyleraraujo/vicuna/specs/057-telegram-document-ingestion/spec.md)

## Summary

Extend the Node-based Telegram bridge so Telegram document messages can ingest
PDF, DOC, and DOCX files. The bridge will download the Telegram file, extract
plain text, append that normalized text to the chat transcript, and persist
both the raw source file and the extracted text to Supermemory with a shared
linkage key and Telegram provenance metadata. PDF extraction will use a Node
dependency, while DOC/DOCX extraction will use the host's `textutil`
conversion tool to avoid a second parsing library and to preserve plain-text
output semantics.

## Technical Context

**Language/Version**: Node.js ESM (`tools/telegram-bridge/*.mjs`) plus Spec Kit markdown artifacts  
**Primary Dependencies**: Existing `supermemory` SDK, Telegram Bot API `getFile`, new `pdf-parse` package, host `textutil` for DOC/DOCX extraction  
**Storage**: Existing local bridge state JSON plus Supermemory document storage  
**Testing**: `node:test` suite in `tools/telegram-bridge/bridge.test.mjs`  
**Target Platform**: Local host bridge process on macOS/Linux with Telegram bot access and Supermemory credentials  
**Project Type**: Node middleware / integration utility  
**Performance Goals**: Keep ingestion synchronous per inbound Telegram message, avoid forwarding binary payloads, and keep extracted text bounded enough for transcript use  
**Constraints**: Explicit bridge policy, minimal new dependencies, clear host error reporting, no leakage of markup/images into transcript text  
**Scale/Scope**: `tools/telegram-bridge/`, `package.json`, `package-lock.json`, and `specs/057-telegram-document-ingestion/`

## Constitution Check

*GATE: Must pass before implementation. Re-check after design updates.*

- **Runtime Policy**: Pass. Format detection, extraction routing, and
  Supermemory persistence stay explicit in bridge code rather than hidden in a
  generic upload proxy.
- **Typed State**: Pass. The Node bridge will use explicit normalized document
  descriptors and linkage metadata objects rather than unstructured blobs.
- **Bounded Memory**: Pass with planned normalization. Extracted text will be
  normalized and bounded before transcript insertion; no binary payload is
  stored in the local transcript state.
- **Validation**: Pass with required work. Targeted Node tests must cover
  supported-type detection, extracted text normalization, Supermemory request
  shaping, and Telegram document handling.
- **Documentation & Scope**: Pass. The implementation updates bridge docs and
  adds one extraction dependency (`pdf-parse`) while reusing the host's
  built-in `textutil` for DOC/DOCX support.

## Project Structure

### Documentation (this feature)

```text
specs/057-telegram-document-ingestion/
├── contracts/
│   └── telegram-document-ingestion-contract.md
├── data-model.md
├── plan.md
├── quickstart.md
├── research.md
├── spec.md
└── tasks.md
```

### Source Code (repository root)

```text
package.json
package-lock.json

tools/telegram-bridge/
├── README.md
├── bridge.test.mjs
├── index.mjs
└── lib.mjs
```

**Structure Decision**: Keep document ingestion inside the existing Telegram
bridge module set. Add helper functions to `lib.mjs` for format detection,
metadata/link shaping, and extraction normalization, while `index.mjs` owns the
Telegram fetch/persist flow.

## Research Summary

- Telegram Bot API document messages provide a `document` object with `file_id`,
  `file_unique_id`, `file_name`, `mime_type`, and `file_size`, and raw bytes are
  retrieved by calling `getFile` and then downloading the returned `file_path`
  from the bot file endpoint.
- The current Supermemory TypeScript SDK exposes:
  - `documents.uploadFile({ file, fileType?, metadata?, containerTags? })`
  - `documents.update(id, { containerTag?, customId?, metadata? })`
  - `documents.add({ content, containerTag?, customId?, metadata? })`
- `pdf-parse` is a maintained Node 20+ PDF text extraction library with a
  direct `PDFParse(...).getText()` path suitable for plain-text extraction.
- macOS `textutil` can convert both `.doc` and `.docx` to `txt`, which keeps
  document plain-text extraction explicit while avoiding another parser
  dependency.

## Design Outline

### 1. Explicit Document Descriptor And Linkage Metadata

Introduce helper functions that normalize Telegram `document` payloads into:

- supported logical type: `pdf`, `doc`, `docx`, or `unsupported`
- a deterministic Supermemory linkage key derived from Telegram chat/message
  provenance and Telegram file identity
- flat metadata objects suitable for Supermemory SDK surfaces

This keeps detection and linkage inspectable and fully testable.

### 2. Download And Extraction Pipeline

Add a document-ingestion path to `handleTelegramMessage()`:

- `telegramRequest('getFile', { file_id })`
- download raw bytes from the Telegram file endpoint
- route by logical type:
  - PDF -> `pdf-parse`
  - DOC/DOCX -> `textutil -convert txt -stdout`
- normalize extracted text into transcript-safe plain text

Errors stay explicit and are surfaced back to Telegram.

### 3. Supermemory Pair Persistence

For each supported document:

1. upload the raw file with `documents.uploadFile(...)`
2. update the uploaded document with explicit container and metadata if needed
3. add the extracted text as a second Supermemory text document with matching
   linkage metadata

The shared linkage key plus Telegram provenance make the pair queryable and
traceable even though the raw upload and text add use different API shapes.

### 4. Transcript Forwarding Policy

The bridge will synthesize a user transcript turn that includes:

- a short file header with filename/type
- the normalized extracted plain text

This preserves continuity for the LLM while keeping formatting noise, markup,
and image placeholders out of the model-visible transcript.

### 5. Documentation And Validation

Update bridge docs to describe:

- supported file formats
- required `SUPERMEMORY_API_KEY`
- DOC/DOCX dependence on host `textutil`
- test and runtime validation steps

## Planned Phase Artifacts

- `research.md`: external Telegram/Supermemory/extraction decisions
- `data-model.md`: normalized document descriptor, linkage metadata, and
  ingestion result shapes
- `contracts/telegram-document-ingestion-contract.md`: Telegram document
  handling and Supermemory persistence contract
- `quickstart.md`: validation commands for bridge tests and document ingestion
- `tasks.md`: implementation backlog organized by user story

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Adding one PDF extraction dependency | The bridge needs local PDF-to-text extraction and no host PDF tool is installed | Shelling out to unavailable `pdftotext` would make the feature non-functional on this host |
| Using host `textutil` for DOC/DOCX extraction | It supports both `.doc` and `.docx` plain-text conversion with no extra package | Adding a second Word parser would increase dependency surface while still not covering legacy `.doc` cleanly |
