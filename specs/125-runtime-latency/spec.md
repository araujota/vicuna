# Feature Specification: Runtime Latency Hot Path Reduction

**Feature Branch**: `125-runtime-latency`  
**Created**: 2026-03-25  
**Status**: Draft  
**Input**: User description: "reuse one deepseek http client forever if possible. tighten bridge polling intervals considerably. then implement caching of the telegram runtime tool catalog in memory, along with caching of the further method/payload contract stuff. declare temperature at 0.2 throughout the application on all turns."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Provider Requests Reuse Persistent Transport State (Priority: P1)

As an operator, I want DeepSeek requests to reuse one persistent HTTP client so
the staged runtime stops paying repeated connection setup cost on every family,
method, payload, and continuation turn.

**Why this priority**: Provider round trips dominate visible latency, and the
current adapter recreates a client for every outbound request.

**Independent Test**: Run provider-mode unit coverage and a direct adapter
smoke path that proves the runtime reuses one configured client instance
instead of rebuilding the transport for each request.

**Acceptance Scenarios**:

1. **Given** multiple outbound provider requests against the same DeepSeek base
   URL, **When** the runtime sends them sequentially, **Then** it reuses one
   configured HTTP client instance instead of constructing a fresh client for
   each request.
2. **Given** runtime configuration with auth, timeouts, and follow-redirect
   policy, **When** the persistent client is reused, **Then** those policies
   remain identical to the previous per-request behavior.

---

### User Story 2 - Telegram Turns Reuse Server-Owned Tool Metadata (Priority: P1)

As a runtime operator, I want the server to cache the Telegram runtime tool
catalog and the derived staged family/method/contract metadata in memory so
bridge-scoped turns stop shelling out and rebuilding staged metadata on every
message.

**Why this priority**: Bridge-scoped Telegram requests currently pay repeated
subprocess and JSON-shaping cost before even reaching the first provider turn.

**Independent Test**: Trigger repeated bridge-scoped catalog loads in unit
coverage and verify the subprocess path is only used on cache miss or explicit
input change while the staged prompts remain byte-for-byte equivalent.

**Acceptance Scenarios**:

1. **Given** repeated bridge-scoped Telegram requests with unchanged runtime
   tool definitions, **When** the server loads the tool surface, **Then** it
   serves the tool catalog and staged family/method/contract metadata from
   in-memory cache.
2. **Given** the authoritative runtime tool payload changes, **When** the next
   bridge-scoped request arrives, **Then** the cache invalidates and rebuilds
   from the new payload before prompt assembly proceeds.

---

### User Story 3 - Bridge Polling Detects Ready Work Faster (Priority: P2)

As a Telegram user, I want the bridge to poll its retained internal surfaces
more aggressively so completed server work appears with less trailing delay.

**Why this priority**: Polling sleeps are not the dominant bottleneck, but they
   still add avoidable tail latency after the server has already finished.

**Independent Test**: Run bridge tests and verify the new polling constants and
reconnect delays are reduced while long-poll Telegram update behavior remains
unchanged.

**Acceptance Scenarios**:

1. **Given** a queued outbox item, **When** the bridge loop is idle, **Then**
   the bridge checks for work on the tightened interval instead of waiting the
   previous one-second poll delay.
2. **Given** a dropped self-emit stream or transient polling error, **When**
   the bridge retries, **Then** it waits the new shorter reconnect interval and
   does not regress correctness.

---

### User Story 4 - Every Provider Turn Uses One Explicit Temperature (Priority: P1)

As an operator, I want every DeepSeek-bound turn to use an explicit
`temperature: 0.2` so direct requests, staged selectors, bridge-scoped turns,
and background work all run under one stable sampling policy.

**Why this priority**: The system is deliberately multi-stage and provider
touch-heavy, so leaving temperature implicit or caller-dependent introduces
unnecessary variation across those stages.

**Independent Test**: Run provider-mode unit coverage and verify every outbound
DeepSeek request body includes `temperature: 0.2`, regardless of request
surface.

**Acceptance Scenarios**:

1. **Given** a direct `/v1/chat/completions` request with no temperature,
   **When** the DeepSeek adapter builds the provider body, **Then** it includes
   `temperature: 0.2`.
2. **Given** a staged family, method, payload, bridge-scoped, or background
   provider turn, **When** the server emits the outbound provider request,
   **Then** it includes `temperature: 0.2`.

### Edge Cases

- What happens when the DeepSeek base URL changes at runtime? The runtime must
  not reuse a client configured for a different authority.
- What happens when `VICUNA_TELEGRAM_RUNTIME_TOOLS_JSON` is set? The cache must
  key off the override payload so test overrides and operator overrides remain
  inspectable.
- What happens when the cached staged metadata was built from an invalid tool
  payload? The server must surface the same validation error instead of keeping
  a poisoned cache entry.
- What happens when many requests concurrently ask for the Telegram catalog?
  Cache reads and refreshes must stay thread-safe and bounded.
- What happens when a caller supplies a different temperature? The runtime must
  still emit `temperature: 0.2` on DeepSeek requests so all turns share one
  explicit policy.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The DeepSeek adapter MUST reuse a persistent HTTP client for
  repeated requests to the same configured provider authority.
- **FR-002**: The persistent DeepSeek client MUST preserve the existing auth,
  timeout, redirect, and TLS behavior already configured for provider requests.
- **FR-003**: The server MUST cache the bridge-scoped Telegram runtime tool
  catalog in memory and reuse it across requests until the authoritative input
  changes.
- **FR-004**: The server MUST cache the staged family, method, and typed
  contract metadata derived from the Telegram runtime tool catalog so prompt
  assembly does not re-derive the same structures on every request.
- **FR-005**: The Telegram runtime tool and staged metadata caches MUST remain
  explicit, inspectable, and bounded in CPU-side control code.
- **FR-006**: The Telegram bridge MUST use reduced internal polling and
  reconnect delays for retained outbox and self-emit loops without changing the
  Telegram Bot API long-poll contract.
- **FR-007**: Automated tests MUST cover persistent DeepSeek client reuse,
  cache hit/miss behavior for Telegram runtime tool metadata, and tightened
  bridge polling timing constants.
- **FR-008**: The DeepSeek adapter MUST emit `temperature: 0.2` on every
  outbound DeepSeek request, including direct, staged, bridge-scoped, and
  background/internal turns.
- **FR-009**: Documentation MUST describe the new caching, transport reuse, and
  explicit temperature policy, along with any operator-visible polling policy
  change.

### Key Entities *(include if feature involves data)*

- **DeepSeek Client State**: The long-lived configured HTTP client and its
  bound provider URL parts, reused for repeated provider requests.
- **Telegram Runtime Tool Cache Entry**: The authoritative runtime tool payload,
  its cache key, and the already-appended Telegram relay tool surface.
- **Staged Tool Metadata Cache Entry**: The staged family/method/contract view
  derived from one runtime tool catalog and reused for prompt construction.
- **Bridge Polling Policy**: The explicit internal sleep and reconnect timings
  for outbox polling, self-emit reconnect, and watchdog checks.
- **Provider Sampling Policy**: The fixed `temperature: 0.2` value applied to
  every outbound DeepSeek request.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Provider-mode tests confirm repeated DeepSeek requests reuse one
  configured client path with no behavioral regression in request mapping.
- **SC-002**: Telegram bridge/server tests confirm repeated bridge-scoped turns
  avoid redundant catalog subprocess work and redundant staged metadata rebuilds
  when the tool payload is unchanged.
- **SC-003**: Internal bridge polling intervals are reduced by at least `50%`
  for retained outbox and self-emit reconnect loops.
- **SC-004**: Provider-mode tests confirm `100%` of outbound DeepSeek requests
  include `temperature: 0.2`.
- **SC-005**: Existing provider and bridge automated suites remain green after
  the latency changes.
