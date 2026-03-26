# Data Model: Runtime Latency Hot Path Reduction

## DeepSeekSharedClient

- `base_url`: normalized provider base URL used to build the client
- `url_parts`: parsed `server_http_url`
- `client`: configured `httplib::Client`
- `mutex`: guards request execution if the shared client is reused across
  concurrent requests

## TelegramRuntimeToolCacheEntry

- `cache_key`: stable hash of the authoritative runtime tool payload or override
- `tools`: JSON array containing the authoritative runtime tools plus appended
  `telegram_relay`
- `loaded_from_override`: whether the entry came from
  `VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON`

## TelegramStagedCatalogCacheEntry

- `cache_key`: must match the tool cache entry that produced it
- `catalog`: `staged_tool_catalog` derived from the cached tool array

## BridgePollingPolicy

- `ask_outbox_idle_delay_ms`
- `self_emit_active_delay_ms`
- `self_emit_error_delay_ms`
- `watchdog_delay_ms`

All values remain explicit constants/config-derived values in bridge code.
