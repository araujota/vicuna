# Feature Specification: Strict Selector Tools And Waterfall VAD

**Feature Branch**: `codex/130-strict-selector-tools`  
**Created**: 2026-03-26  
**Status**: Draft  
**Input**: User description: "convert our current json output system to strict tool calls with enums, and modify all relevant prompts/the provider call itself to use the beta strict tool calls. preserve the waterfall process of selecting family, method, and passing contract schema, then awaiting response. we should also keep selector contexts prefix-stable, although there are several of them and they are populated dynamically, but we should do that if we can. also, ensure new VADs are added in every following step in the waterfall, it's a big failure that that wasn't happening."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Selector turns terminate with structured outputs (Priority: P1)

As an operator, I want family, method, and payload selector turns to use strict
provider-enforced schemas so the staged loop emits small structured decisions
without relying on freeform JSON parsing.

**Why this priority**: The live Telegram failures came from reasoning mode
consuming the full shared output budget and returning no visible JSON content.

**Independent Test**: Provider tests can assert that staged family and method
turns are sent as DeepSeek beta strict tool calls with enum-constrained
arguments and return validated tool-call arguments rather than freeform
assistant content.

**Acceptance Scenarios**:

1. **Given** a staged family-selection turn, **When** the provider request is
   sent, **Then** it uses a strict tool schema with one enum field for the
   available family names and no `response_format=json_object`.
2. **Given** a staged method-selection turn, **When** the provider request is
   sent, **Then** it uses a strict tool schema with one enum field for the
   available methods plus the `back` and `complete` sentinels.
3. **Given** a staged payload-construction turn, **When** the provider request
   is sent, **Then** it uses a strict tool schema that mirrors the typed method
   contract and returns tool-call arguments that can be validated directly.

---

### User Story 2 - Waterfall control remains explicit and inspectable (Priority: P1)

As an operator, I want the existing family -> method -> payload waterfall to
remain explicit in CPU-side code, with prefix-stable selector contexts and
traceable provider/system traffic at each step.

**Why this priority**: Replacing freeform JSON with strict tools must not
collapse the inspectable staged controller into hidden provider-owned policy.

**Independent Test**: Provider tests can inspect request traces and outbound
provider payloads to confirm that the server still runs three explicit stages,
retains stage-specific prompts, and records strict selector tool metadata.

**Acceptance Scenarios**:

1. **Given** a request with staged tools, **When** the staged loop runs,
   **Then** request traces still record separate family, method, and payload
   attempts with their selected outputs.
2. **Given** repeated selector turns with the same available options, **When**
   the prompt bundle is built, **Then** the stable prefix of the selector
   prompt remains unchanged aside from the dynamic options/validation suffix.

---

### User Story 3 - Renewed VAD guidance appears on each following stage (Priority: P1)

As an operator, I want the renewed VAD guidance sentence to be available on
every staged turn after new reasoning is produced so each subsequent selector
and payload step can reflect the latest emotive state.

**Why this priority**: The current staged loop recomputes emotive state from
reasoning text but skips VAD injection because the selector messages do not
look like active tool-continuation spans.

**Independent Test**: Provider tests can assert that family-to-method,
method-to-payload, and retry turns include one additive VAD `system` message
and that traces record the injected sentence rather than a skip reason.

**Acceptance Scenarios**:

1. **Given** a family-selection turn that returns reasoning text, **When** the
   method-selection turn is assembled, **Then** the latest VAD guidance
   sentence is injected additively and logged.
2. **Given** a method-selection turn that returns reasoning text, **When** the
   payload-selection turn or retry turn is assembled, **Then** the latest VAD
   guidance sentence is injected additively and logged.

## Edge Cases

- What happens when a selector turn still returns no tool call despite strict
  mode being enabled?
- What happens when the available family or method set is empty after
  normalization?
- What happens when a payload schema uses nested objects, arrays, enums, or
  `anyOf` and must be translated into the DeepSeek strict subset?
- What happens when the staged loop retries after a validation error and the
  VAD has changed from the prior attempt?
- What happens when bridge-scoped Telegram turns and non-bridge auto-tool turns
  share the same staged strict-tool controller?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The staged family-selection turn MUST use a DeepSeek beta strict
  tool call instead of `response_format={"type":"json_object"}`.
- **FR-002**: The staged method-selection turn MUST use a DeepSeek beta strict
  tool call with an enum-constrained method field and explicit `back` and
  `complete` sentinel support.
- **FR-003**: The staged payload-construction turn MUST use a DeepSeek beta
  strict tool call whose parameters mirror the selected method contract.
- **FR-004**: The provider adapter MUST route staged strict-tool selector turns
  to DeepSeek beta endpoints required for strict tool mode.
- **FR-005**: The staged controller MUST parse selector outputs from tool-call
  arguments instead of freeform assistant `content`.
- **FR-006**: The server MUST keep family, method, and payload selection as
  separate explicit controller stages and MUST continue to support `back` and
  `complete` transitions.
- **FR-007**: Selector prompt assembly MUST keep a stable shared prefix and
  avoid unnecessary prompt churn across repeated stage attempts.
- **FR-008**: Additive VAD guidance MUST be eligible on every staged turn after
  a prior staged result produced reasoning text, even when there is no classic
  tool-continuation span in the message history.
- **FR-009**: Request traces MUST record whether staged strict-tool selector
  turns injected renewed VAD guidance, plus the exact guidance string when
  present.
- **FR-010**: Provider tests and docs MUST be updated to describe DeepSeek beta
  strict selector tools, the preserved waterfall behavior, and the per-stage
  VAD propagation rule.

### Key Entities *(include if feature involves data)*

- **Staged Selector Tool**: A server-owned strict tool schema used only for one
  staged decision turn, such as family selection, method selection, or payload
  submission.
- **Selector Prompt Bundle**: The cached prompt assembly artifact containing a
  stable selector prefix plus stage-specific dynamic option/contract segments.
- **Staged Guidance Eligibility**: The runtime rule that determines whether the
  latest VAD sentence should be injected on a following staged turn.
- **Strict Selector Trace Event**: A request-trace event that records the
  schema-constrained selector request/response state for one stage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Local provider tests confirm family and method turns are sent as
  strict selector tools with enum-constrained arguments and no JSON-output
  response format.
- **SC-002**: A staged Telegram greeting path can reach family selection,
  method selection, and payload submission without relying on freeform JSON
  extraction from assistant content.
- **SC-003**: Request traces show renewed VAD guidance injected on each staged
  step after prior staged reasoning is produced.
- **SC-004**: Selector prompt caching/assembly keeps a stable shared prefix and
  reduces avoidable prompt churn across retries and repeated staged requests.
