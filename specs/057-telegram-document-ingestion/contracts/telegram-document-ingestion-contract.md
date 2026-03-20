# Contract: Telegram Document Ingestion And Supermemory Linking

## Inbound Telegram Contract

The bridge accepts Telegram updates whose `message` contains:

- `chat.id`
- `message_id`
- `document.file_id`
- optional `document.file_unique_id`
- optional `document.file_name`
- optional `document.mime_type`
- optional `document.file_size`

Unsupported updates continue through the existing text-only path or receive a
direct unsupported-format reply.

## Supported Logical Types

- `pdf`
  - matched by MIME `application/pdf` or `.pdf` extension
- `doc`
  - matched by legacy Word MIME types or `.doc` extension
- `docx`
  - matched by DOCX MIME type or `.docx` extension

The bridge prefers explicit matches on MIME type or file extension and does not
attempt generic binary sniffing beyond that scope.

## Download Contract

1. Call `getFile` with Telegram `file_id`
2. Expect a response containing `file_path`
3. Download bytes from:

```text
https://api.telegram.org/file/bot<TELEGRAM_BOT_TOKEN>/<file_path>
```

## Extraction Contract

- `pdf` -> local `pdf-parse` extraction
- `doc` / `docx` -> host `textutil -convert txt -stdout`

The extracted output is normalized into plain text only:

- trim leading/trailing whitespace
- collapse excessive blank lines
- drop empty noise-only output

## Supermemory Persistence Contract

### Raw file

1. Upload source bytes via `client.documents.uploadFile(...)`
2. Update the resulting document id via `client.documents.update(...)` so the
   raw file document gets explicit `containerTag`, `customId`, and shared flat
   metadata

### Extracted text

Persist a second document via `client.documents.add(...)` with:

- `content`: normalized extracted plain text
- `containerTag`: shared Telegram-derived container
- `customId`: deterministic extracted-text id
- `metadata`: shared linkage metadata plus `contentKind=extracted_text`

## Transcript Contract

The model-visible user turn is plain text only and follows this shape:

```text
[Document: <filename>]
[Type: <logical type>]

<normalized extracted text>
```

No markup, XML, image placeholders, or binary content may appear in the
transcript entry.
