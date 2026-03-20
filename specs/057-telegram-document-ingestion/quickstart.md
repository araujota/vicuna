# Quickstart: Telegram Document Ingestion And Supermemory Linking

## Prerequisites

- `TELEGRAM_BOT_TOKEN` set
- `SUPERMEMORY_API_KEY` set
- bridge host has `/usr/bin/textutil` available for DOC/DOCX extraction
- Node dependencies installed from repository root

## Validation

1. Install dependencies:

```bash
cd /Users/tyleraraujo/vicuna
npm install
```

2. Run targeted bridge tests:

```bash
cd /Users/tyleraraujo/vicuna
node --test tools/telegram-bridge/bridge.test.mjs
```

3. Start the bridge:

```bash
cd /Users/tyleraraujo/vicuna
npm run telegram-bridge:start
```

4. Send a PDF, DOC, or DOCX document to the Telegram bot.

5. Verify:

- the bridge no longer replies with the plain-text-only rejection
- the transcript-visible turn contains only plain text plus the document header
- the raw file and extracted text are both persisted in Supermemory with a
  shared linkage key in metadata

## Failure Checks

- Unset `SUPERMEMORY_API_KEY` and confirm the bridge reports Supermemory
  persistence failure directly to Telegram.
- Temporarily hide `textutil` from `PATH` and confirm DOC/DOCX ingestion fails
  with a direct host requirement error.
