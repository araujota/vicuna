# Contract: Staged Tool Loop JSON Shapes

## Family Selection

The provider receives only the offered families and must return exactly:

```json
{
  "family": "Telegram"
}
```

Rules:
- `family` must exactly match one offered family name.
- No back or complete sentinel is allowed at this stage.

## Method Selection

The provider receives only methods for the chosen family and must return exactly:

```json
{
  "method": "send_message"
}
```

Allowed sentinel values:

```json
{
  "method": "back"
}
```

```json
{
  "method": "complete"
}
```

Rules:
- Non-sentinel `method` values must exactly match one offered method name for the current family.
- `complete` ends the active loop and hands control back to the existing replay/idle pipeline.

## Payload Construction

The provider receives the typed contract for the chosen method and must return exactly one of:

```json
{
  "action": "submit",
  "payload": {
    "chat_id": "12345",
    "text": "Here is the update."
  }
}
```

```json
{
  "action": "back"
}
```

Rules:
- `payload` is required only when `action` is `submit`.
- The payload must validate against the currently offered method contract.
- Validation failure keeps the runtime at the payload stage and reports the error additively.
