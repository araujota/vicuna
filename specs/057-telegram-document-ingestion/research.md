# Research: Telegram Document Ingestion And Supermemory Linking

## Decision 1: Use Telegram `getFile` + file endpoint for raw bytes

- **Decision**: Download Telegram documents by calling Bot API `getFile` with
  the inbound `document.file_id`, then fetch the returned `file_path` from the
  bot file endpoint.
- **Rationale**: This is the official Telegram document retrieval path and
  keeps raw file download explicit in the bridge.
- **Alternatives considered**:
  - Reuse only metadata from the inbound update: rejected because the update
    does not include raw bytes.
  - Forward the Telegram URL into Supermemory directly: rejected because the
    bridge still needs local plain-text extraction.

## Decision 2: Store a raw-file document and a separate extracted-text document

- **Decision**: Persist two Supermemory records per supported Telegram
  document: one uploaded raw file and one text document containing normalized
  extracted text.
- **Rationale**: The user explicitly requested both the file itself and the
  extracted text. Separate documents keep the artifact and model-visible text
  independently inspectable.
- **Alternatives considered**:
  - Store only the raw file and rely on Supermemory's internal extraction:
    rejected because the bridge still needs deterministic plain text for the
    Telegram transcript.
  - Store only the extracted text: rejected because it loses the original
    artifact.

## Decision 3: Link the pair through shared metadata and container grouping

- **Decision**: Build a deterministic linkage key and shared Telegram metadata,
  use a stable Telegram-derived container tag for the text document, and update
  the raw uploaded file document with matching container/metadata after upload.
- **Rationale**: `documents.uploadFile()` and `documents.add()` expose slightly
  different request shapes; shared metadata plus container grouping gives the
  bridge a stable cross-record join key without inventing a side database.
- **Alternatives considered**:
  - Depend only on Supermemory-assigned ids: rejected because the bridge needs
    a deterministic link across two different writes.
  - Build a local bridge-side linkage store: rejected as unnecessary state.

## Decision 4: Use `pdf-parse` for PDF extraction

- **Decision**: Add the `pdf-parse` Node package and use its `PDFParse(...).
  getText()` path for PDF text extraction.
- **Rationale**: No host PDF text tool is installed on this machine, and
  `pdf-parse` provides maintained Node 20+ extraction with direct plain-text
  output.
- **Alternatives considered**:
  - `pdftotext`: rejected because it is not installed on this host.
  - OCR-first extraction: rejected because the user asked only for text and the
    change should stay minimal.

## Decision 5: Use host `textutil` for DOC/DOCX extraction

- **Decision**: Convert `.doc` and `.docx` files to plain text with
  `textutil -convert txt -stdout`.
- **Rationale**: `textutil` is available on this host and can convert both DOC
  and DOCX to text while stripping rich formatting.
- **Alternatives considered**:
  - `mammoth`: rejected because it mainly targets `.docx`, would add another
    dependency, and still would not cleanly solve legacy `.doc`.
  - LibreOffice headless conversion: rejected because it is not a current
    repository/runtime dependency and would be heavier operationally.

## Decision 6: Keep transcript-visible text normalized and explicit

- **Decision**: Normalize extracted text by stripping empty noise, collapsing
  excessive blank lines, and prepending a short document header before storing
  it as the user turn.
- **Rationale**: The runtime needs context about the file source, but the
  transcript must remain plain text rather than raw markup or binary data.
- **Alternatives considered**:
  - Send raw extraction output verbatim: rejected because it can contain noisy
    spacing and poor readability.
  - Omit file headers entirely: rejected because transcript provenance becomes
    ambiguous when multiple documents are discussed.
