# Feature Specification: Authoritative ReAct Action Contract Guarantee

**Feature Branch**: `[103-react-action-contract-guarantee]`  
**Created**: 2026-03-24  
**Status**: Draft  
**Input**: User descriptions:
- "we have had this issue many times now. you must absolutely guarantee the presence of the Action text, exactly as expected, always. whether that's achieved through prompting or whatever, it must be as close to absolute as possible."
- "also, when a request doesn't contain the Action label, rather that surfacing this as a message to the user, it should simply try again"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Staged tool turns always resolve to the exact action contract (Priority: P1)

As the operator, I need every staged authoritative ReAct tool phase to resolve to the exact required action token so the runtime never depends on the model to perfectly retype fixed control words like `select_tool`, `select_method`, `act`, `decide`, `answer`, or `ask`.

**Why this priority**: This is the direct reliability requirement. Prompting alone has already proven insufficient in production.

**Independent Test**: Feed each staged phase valid payloads with and without the `action` field and verify the runtime resolves them to the exact phase action while still rejecting conflicting action values.

**Acceptance Scenarios**:

1. **Given** a `select_tool_family` continuation, **When** the runtime parses the staged payload, **Then** the resulting control artifact MUST carry the exact `select_tool` action even if the model omitted that fixed field.
2. **Given** a `select_method`, `emit_arguments`, `decide_after_tool`, or `emit_response` continuation, **When** the runtime parses the stage, **Then** the resulting control artifact MUST carry the exact required action for that phase and MUST reject conflicting action values.
3. **Given** a staged phase payload reaches the parser, **When** validation runs, **Then** the parser MUST validate against the exact phase contract and reject any duplicate, conflicting, or malformed action surface.

---

### User Story 2 - Missing or malformed action labels stay inside retry (Priority: P1)

As a Telegram user, I need malformed staged control payloads to stay inside the runtime retry loop so I never receive raw control JSON, internal action-label errors, or other controller artifacts.

**Why this priority**: The current production failure leaks control-surface mistakes into Telegram, which is worse than a retry.

**Independent Test**: Reproduce a staged turn that emits a malformed control payload with a missing action field and verify the runtime retries the same turn or rewinds the same stage without publishing the malformed payload or internal parse detail to Telegram.

**Acceptance Scenarios**:

1. **Given** a staged turn emits a payload missing the required action field, **When** the parser evaluates it, **Then** the runtime MUST classify it as an internal parse failure and retry instead of surfacing the payload to the user.
2. **Given** repeated staged parse failures for the same phase, **When** the stage retry budget is exceeded, **Then** the runtime MUST rewind or re-prepare the stage according to explicit policy rather than publishing raw controller text.
3. **Given** the runtime ultimately cannot recover a staged control step, **When** the turn terminates, **Then** any user-visible failure MUST avoid exposing internal action-label or control-contract details.

---

### User Story 3 - Non-staged fallback cannot mistake control JSON for a user reply (Priority: P2)

As the operator, I need the non-staged fallback path to reject control-shaped JSON fragments as terminal visible prose so bare artifacts like `{"tool_family_id":"web_search"}` cannot be mistaken for a final assistant reply.

**Why this priority**: The observed Telegram leak happened because malformed control JSON was accepted as visible text after fallback parsing.

**Independent Test**: Feed a non-staged authoritative turn with visible JSON that resembles a staged control artifact but lacks a valid action contract, and verify the runtime retries instead of treating it as terminal assistant text.

**Acceptance Scenarios**:

1. **Given** a non-staged authoritative turn with visible JSON containing `tool_family_id`, `method_name`, `decision`, or other control-shape keys but no valid controlling action contract, **When** fallback parsing runs, **Then** the runtime MUST reject it as malformed control rather than infer a terminal visible-text action.
2. **Given** a non-staged authoritative turn with ordinary visible prose and no control-shape payload, **When** fallback parsing runs, **Then** the existing terminal fallback behavior MUST remain available.
3. **Given** targeted provenance or logs for a rejected control-shaped payload, **When** operators inspect them, **Then** they MUST show that the payload stayed inside retry and was not emitted as a user-visible response.

### Edge Cases

- The model emits the fixed action token twice because the stage prefill already contains it and the generated tail starts with another `action` field.
- The model emits only a partial staged payload after the host-pinned action prefix, such as an unterminated JSON object or truncated string.
- A first-step active turn that does not strictly require a tool still emits control-shaped JSON in visible output; fallback must not treat it as prose.
- A staged retry loops across the same malformed payload multiple times; the runtime must rewind the phase explicitly instead of surfacing the repeated failure.
- An `emit_response` stage already has a host-pinned `answer` or `ask` action and the model attempts to switch actions in the generated tail.
- Telegram deferred delivery is active; malformed staged control must not create a queued follow-up message before validation succeeds.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST own the exact staged controller action contract for `select_tool_family`, `select_method`, `emit_arguments`, `decide_after_tool`, and `emit_response`.
- **FR-002**: If a staged JSON payload omits the required action field but otherwise matches the current phase contract, the runtime MUST normalize the exact required action internally for that phase.
- **FR-003**: Staged parser validation MUST continue to enforce the exact phase contract and MUST reject duplicate, conflicting, or malformed action surfaces.
- **FR-004**: A staged payload that omits or corrupts the expected action contract MUST remain an internal parse failure and MUST trigger retry or stage rewind instead of becoming visible assistant text.
- **FR-005**: The runtime MUST NOT publish raw staged control JSON, internal action-label parse errors, or malformed control payloads to Telegram or other user-visible reply surfaces.
- **FR-006**: If staged retries are exhausted, any terminal user-visible error MUST be generic and MUST NOT expose internal controller-contract details.
- **FR-007**: The non-staged fallback path MUST reject visible JSON that matches staged-control shapes without a valid action contract, including payloads containing keys such as `tool_family_id`, `method_name`, `decision`, or control-only `action` fields.
- **FR-008**: Existing non-staged fallback inference for ordinary visible prose MUST remain available when the visible payload is not control-shaped.
- **FR-009**: Runtime logs and provenance MUST distinguish runtime-owned staged action contracts from model-generated payload content and MUST record retries caused by malformed staged control.
- **FR-010**: Targeted automated tests MUST cover host-owned staged action prefixes, malformed staged payload retry behavior, stage rewind after repeated failures, and rejection of control-shaped JSON in non-staged fallback.
- **FR-011**: Developer documentation MUST describe that staged action tokens are host-owned and that malformed staged control remains inside retry rather than being surfaced to users.

### Key Entities *(include if feature involves data)*

- **RuntimeOwnedStageAction**: The runtime-owned exact action value for one staged phase, either emitted directly by the model or normalized by the runtime when omitted.
- **StageVariablePayload**: The model-generated portion of a staged JSON artifact other than the fixed phase action.
- **MalformedControlRetry**: The explicit retry or rewind path used when a staged control payload is missing the required action contract or otherwise conflicts with the current phase.
- **ControlShapedVisiblePayload**: A visible JSON payload that resembles staged controller output and must not be accepted as terminal user prose without a valid contract.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Targeted tests prove that each staged phase resolves to the exact required action value even when the model omits the fixed action field.
- **SC-002**: A host-traced reproduction of the Telegram book-download request no longer publishes raw control JSON such as `{"tool_family_id":"web_search"}` to Telegram.
- **SC-003**: Malformed staged control payloads trigger retry or stage rewind and do not surface internal action-label text to the user.
- **SC-004**: Non-staged fallback continues to accept ordinary visible prose while rejecting control-shaped JSON fragments as terminal visible replies.
- **SC-005**: Developer documentation and provenance clearly expose the host-owned staged action contract and the retry-only handling of malformed control payloads.
