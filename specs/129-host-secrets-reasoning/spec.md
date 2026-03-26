# Feature Specification: Host Secrets And Verbatim Stage Traces

**Feature Branch**: `codex/129-host-secrets-reasoning`  
**Created**: 2026-03-26  
**Status**: Draft  
**Input**: User description: "i thought we set the API keys for all the media server tools, tavily, and supermemory, earlier? these need to be set on the host in a way that survives rebuilds. also, I'm not seeing the current VAD passed into this prompt, even though it should be. we need to log the explicit reasoning trace from each stage such that when i ask you for the reasoning traces, you can give them to me verbatim; we already capture this text for recomputing emotive moment, this should not be hard."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Host secrets survive rebuilds (Priority: P1)

As an operator, I want media-tool, Tavily, and memory credentials stored in a
host-level location outside the checkout so rebuilds and branch swaps do not
silently remove tool availability.

**Why this priority**: The live Radarr failure was caused by the runtime
executing a real media tool with no API key configured on the host.

**Independent Test**: A host-oriented unit/integration path can confirm the
runtime and bridge scripts load a persistent env file, sync tool secrets into a
stable OpenClaw secrets path, and keep the runtime catalog available after a
checkout change.

**Acceptance Scenarios**:

1. **Given** a host with `/etc/vicuna/vicuna.env`, **When** the runtime or
   bridge starts from any checkout, **Then** it sources that env file and sees
   the same durable secret values.
2. **Given** durable env values for Radarr, Sonarr, Chaptarr, Tavily, and
   Supermemory, **When** the startup sync runs, **Then** the OpenClaw secrets
   file and runtime catalog are written to stable non-checkout paths.

---

### User Story 2 - Request traces retain verbatim stage reasoning (Priority: P1)

As an operator, I want every staged provider turn to retain the exact returned
`reasoning_content` and visible `content` in the request-trace registry so I
can inspect failed or slow selector turns verbatim after the fact.

**Why this priority**: Current traces expose only character counts, which makes
it impossible to diagnose why a stage consumed its entire output budget without
emitting JSON.

**Independent Test**: Provider tests can inspect `/v1/debug/request-traces`
after staged turns and assert that provider-finished events retain the exact
reasoning and content strings returned by the mock provider.

**Acceptance Scenarios**:

1. **Given** a staged family, method, or payload turn, **When** the provider
   returns reasoning text, **Then** the correlated request-trace event stores
   that reasoning text verbatim.
2. **Given** a staged turn that also emits visible content, **When** the
   request trace is read, **Then** the exact visible content is also present.

---

### User Story 3 - VAD and guidance injection are inspectable (Priority: P1)

As an operator, I want request traces to state whether additive VAD and
heuristic guidance were injected, skipped, and why, so I can confirm the model
actually received the current emotive guidance on continuation turns.

**Why this priority**: The live debugging path currently cannot prove whether a
missing VAD sentence is caused by prompt assembly, continuation-span detection,
or later provider behavior.

**Independent Test**: A provider test can assert that request traces include an
explicit runtime-guidance event with the exact injected VAD sentence, or a skip
reason when no VAD guidance is added.

**Acceptance Scenarios**:

1. **Given** an active tool continuation with assistant reasoning state,
   **When** the runtime injects VAD guidance, **Then** the request trace records
   the exact VAD guidance text and insertion point.
2. **Given** a request where VAD guidance is not injected, **When** the trace is
   inspected, **Then** it records the skip reason explicitly.

## Edge Cases

- What happens when staged reasoning text is very large and multiple stage
  attempts occur in one request?
- What happens when a bridge-scoped turn has no active tool continuation span
  and therefore no VAD sentence should be injected?
- What happens when the host lacks `/etc/vicuna/vicuna.env` but a checkout
  `.envrc` still exists?
- What happens when Supermemory remains env-backed while media/Tavily tools are
  persisted in OpenClaw tool-secrets JSON?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime and bridge startup scripts MUST load a durable host
  env file from `VICUNA_SYSTEM_ENV_FILE` or `/etc/vicuna/vicuna.env` when it
  exists.
- **FR-002**: The runtime startup path MUST support syncing media-tool, Tavily,
  and memory settings from env into stable OpenClaw secrets/catalog files
  outside the active checkout.
- **FR-003**: The durable OpenClaw secrets path and runtime catalog path MUST
  be configurable and default to stable non-checkout locations suitable for
  rebuild-safe host deployment.
- **FR-004**: Provider request-trace events for completed provider calls MUST
  retain the exact `reasoning_content` string returned by the provider.
- **FR-005**: Provider request-trace events for completed provider calls MUST
  retain the exact visible `content` string returned by the provider.
- **FR-006**: Request traces MUST record explicit runtime-guidance events for
  additive VAD and heuristic guidance, including exact injected text when
  present.
- **FR-007**: Request traces MUST record explicit skip reasons when additive VAD
  guidance is not injected.
- **FR-008**: Documentation and tests MUST describe the durable host secret
  path, the startup sync behavior, and the new trace observability fields.

### Key Entities *(include if feature involves data)*

- **Durable Host Env File**: The rebuild-safe host environment file, typically
  `/etc/vicuna/vicuna.env`, that carries provider, bridge, and tool-secret
  inputs.
- **Durable OpenClaw Tool Secrets**: The JSON file that stores persisted media,
  Tavily, and related tool credentials outside the active checkout.
- **Runtime Guidance Trace Event**: A structured request-trace event recording
  additive VAD/heuristic guidance or a skip reason.
- **Provider Finished Trace Event**: A structured request-trace event recording
  exact provider outputs for a completed stage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Local tests can read request traces and recover verbatim staged
  reasoning strings from provider-finished events.
- **SC-002**: Local tests can confirm an explicit runtime-guidance trace event
  appears with exact VAD text or a concrete skip reason.
- **SC-003**: Startup scripts and docs clearly point to one durable host env
  file and one durable OpenClaw secrets path that survive checkout rebuilds.
- **SC-004**: A host deployment can rebuild or switch checkouts without losing
  Radarr, Sonarr, Chaptarr, Tavily, or Supermemory configuration.
