# Data Model: Telegram Document Ingestion And Supermemory Linking

## Telegram Document Descriptor

- **Purpose**: Normalize Telegram `message.document` metadata into an explicit
  bridge-owned shape before download or extraction.
- **Fields**:
  - `chatId`: string form of Telegram `chat.id`
  - `messageId`: string form of Telegram `message_id`
  - `fileId`: Telegram reusable file id
  - `fileUniqueId`: stable Telegram file-unique id
  - `fileName`: original file name if present, else synthesized fallback
  - `mimeType`: original MIME type if present
  - `fileSize`: numeric byte count if present
  - `logicalType`: one of `pdf`, `doc`, `docx`, `unsupported`
  - `extension`: lowercase file extension without dot

## Extracted Document Text

- **Purpose**: Hold the plain-text-only content derived from the raw file.
- **Fields**:
  - `text`: normalized extracted text
  - `characterCount`: post-normalization character count
  - `lineCount`: post-normalization line count
  - `extractionMethod`: `pdf-parse` or `textutil`

## Supermemory Linkage Record

- **Purpose**: Provide flat shared metadata for both the raw file and extracted
  text documents.
- **Fields**:
  - `linkKey`: deterministic join key for the raw/text pair
  - `containerTag`: stable Telegram-derived grouping key
  - `source`: literal `telegram_bridge`
  - `telegramChatId`
  - `telegramMessageId`
  - `telegramFileId`
  - `telegramFileUniqueId`
  - `telegramFileName`
  - `telegramMimeType`
  - `telegramLogicalType`
  - `contentKind`: `source_file` or `extracted_text`

## Telegram Document Ingestion Result

- **Purpose**: Represent the end-to-end bridge result for one document update.
- **Fields**:
  - `descriptor`: Telegram Document Descriptor
  - `extracted`: Extracted Document Text
  - `rawDocumentId`: Supermemory id for uploaded file document
  - `textDocumentId`: Supermemory id for extracted text document
  - `transcriptText`: user-visible plain-text turn appended to local state
  - `partialFailure`: boolean
  - `error`: operator-visible failure message when any stage fails
