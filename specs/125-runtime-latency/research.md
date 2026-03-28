# Research: Runtime Latency Hot Path Reduction

## DeepSeek transport reuse

- Local code inspection shows the DeepSeek adapter currently constructs a fresh
  `httplib::Client` for every request in
  [/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp](/Users/tyleraraujo/vicuna/tools/server/server-deepseek.cpp).
- Upstream `cpp-httplib` documentation indicates the client supports keep-alive
  and is intended to be reused as a configured object rather than recreated for
  every call.
- Design implication: add one explicit configured client holder in the
  DeepSeek adapter and bind it to one parsed provider authority. Do not add a
  generic pool or hidden connection manager.

## Telegram runtime catalog caching

- Local server code shells out to the runtime tool harness on every
  bridge-scoped request via `runtime-tools`, then appends `telegram_relay`.
- The same request path then rebuilds family/method/contract structures from
  scratch even when the underlying tool payload is unchanged.
- Design implication: cache the authoritative runtime tool payload and the
  derived staged catalog as separate inspectable structures so cache hits avoid
  both the subprocess and the repeated metadata build.

## Bridge polling

- Local bridge code retains fixed `1000ms` sleeps in outbox polling and
  self-emit reconnect paths, plus a `5000ms` watchdog interval.
- These sleeps do not dominate total latency, but they add visible tail delay
  after the server has already completed work.
- Design implication: reduce those fixed sleeps substantially while leaving
  Telegram Bot API long polling unchanged.

## DeepSeek completion ceilings

- DeepSeek's chat-completions documentation treats `max_tokens` as the
  completion ceiling for the whole generated output, including thinking /
  reasoning output.
- The original DeepSeek-V3 release material advertised roughly `60 tokens/s`,
  but the current API docs do not publish a hard throughput guarantee for
  DeepSeek-V3.2.
- Design implication: keep the runtime token ceiling explicit and conservative.
  For this staged, provider-touch-heavy runtime, reduce the fixed cap from
  `1024` to `256` so selector turns and ordinary bridge turns cannot spend a
  full kilotoken budget per provider hop.
