# Feature Specification: Ubiquitous Staged Tool Family/Method/Payload Orchestration

**Feature Branch**: `[119-staged-tool-loop]`  
**Created**: 2026-03-25  
**Status**: Draft  
**Input**: User description: "one final change, and this one is specifically designed to get the provider to touch our system more often:
- construct a concise but effective \"core system prompt\" for Vicuna as a helpful personal concierge with access to a wide variety of tools
- at the \"start\" of the process, expose only the names of the high-level tool families, and a brief description of what they facilitate for the system(and what they allow the system to facilitate for the user). the system's response at this stage MUST be exactly one of the tool families in json, even if it just wants to respond immediately(the telegram tool is the tool for this)
- when a tool family is sent to the system, we must parse it correctly, and fetch the methods for that tool family by name and by similar brief descriptions(what they do for the system, what they allow the system to do for the user). the prompt should construct a segment, something like \"you are choosing a method of the <toolfamilyname> tool...\" with instructions that it must output exactly one method name in json
- when a method name is sent to the system, we must parse it correctly, and fetch a typed contract for that method with brief descriptions of each field. construct a separate prompt segment at this stage to the effect of, \"you are constructing a payload for the <methodname> method of the <toolfamily> tool, with typed contract...\", with instructions that it must output exactly one payload in json, which we must parse and execute as the system
- at each step except the first one, options to \"go back\" to the previous selection(from method selection to tool selection, payload generation to method selection) must be present, and must also be parsed by the system and function as expected.
- build an extensible engine for fetching tool family names/descs, tool method names/descs, and method contracts based on the responses we receive from the provider. add in your guide for future development that all future tools/methods must provide these layers. engineer resusable prompt assembly engines as well, for each step.
- after a system has received the results of an actual tool call, the loops should start again from the family selection process.
- the system should also be able to end the loop and consider it complete if it has met the conditions, at which point the active part of the process is over, and the system transitions to cognitive episode replay
- this uber-staggered system must be ubuquitous across all modes of operation. the goal is to give us as many opportunities to inject VADs/heuristics as possible, which is only doable when the provider touches our system in some way

research and architect a way to achieve this extensibly, while preserving all the benefits or traditional ReAct. your implementation must include a reconciliation of this system with deepseek's exising support of tool use; possibly each selectable option at each stage is considered a \"tool\" to be selected? research and find the best option, architect the implementation, and build it out to completion."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stage Every Tool Decision Through Family, Method, and Payload Selection (Priority: P1)

As the runtime, I can force the provider to choose a tool family, then a method, then a typed payload in separate JSON-constrained turns, so every meaningful action path passes through explicit checkpoints before execution.

**Why this priority**: This is the core behavioral change. Without the staged controller, there is no new provider-touch cadence and no ubiquitous checkpoint surface for VAD and heuristic injection.

**Independent Test**: Can be fully tested by issuing one request that requires a tool call, asserting the runtime sends three staged prompts, parses each JSON response, executes the selected method, and restarts the loop from family selection after the tool result.

**Acceptance Scenarios**:

1. **Given** an active request with available tools, **When** the runtime enters the staged loop, **Then** the provider first receives only tool-family names plus short descriptions and must return exactly one family choice in JSON.
2. **Given** a chosen tool family, **When** the runtime continues the loop, **Then** the provider receives only methods for that family plus short descriptions and must return exactly one method choice in JSON.
3. **Given** a chosen method, **When** the runtime continues the loop, **Then** the provider receives the typed contract for that method and must return exactly one payload object in JSON that the runtime validates and executes.
4. **Given** a completed tool call, **When** the runtime receives the tool observation, **Then** the next active loop begins again at family selection rather than skipping directly to final output.

---

### User Story 2 - Support Back Navigation and Explicit Completion Without Losing ReAct Behavior (Priority: P2)

As the runtime, I can let the provider move back one stage or mark the active loop complete without abusing native tool calls, so the system stays inspectable and retains the think-act-observe discipline of ReAct while gaining more checkpoints.

**Why this priority**: A staged loop is only workable if the model can recover from bad branch choices and stop cleanly once the user-facing task is satisfied.

**Independent Test**: Can be tested by forcing the provider to choose `back` from payload generation and method selection, and by choosing the explicit completion path after a successful tool result, asserting the runtime transitions correctly each time.

**Acceptance Scenarios**:

1. **Given** the runtime is at method selection, **When** the provider returns the `back` sentinel, **Then** the runtime returns to family selection without executing a tool.
2. **Given** the runtime is at payload generation, **When** the provider returns the `back` sentinel, **Then** the runtime returns to method selection for the same family.
3. **Given** the runtime is at method selection after task conditions are satisfied, **When** the provider chooses the explicit completion sentinel, **Then** the active loop ends and the system becomes eligible for cognitive replay/background stages.

---

### User Story 3 - Make the Staged Controller Extensible and Ubiquitous Across Active and Background Modes (Priority: P3)

As a maintainer, I can add new tools by supplying family, method, and contract metadata, and I can rely on the same staged controller in normal, ongoing-task, and cognitive replay-adjacent flows, so tool growth does not fork orchestration logic.

**Why this priority**: The user requested that the staged system become ubiquitous and future-proof, not a one-off wrapper around one foreground provider path.

**Independent Test**: Can be tested by verifying that the runtime builds family/method/contract prompts from the shared catalog metadata, that background/idle execution paths use the same controller, and that docs specify the metadata requirements for future tools.

**Acceptance Scenarios**:

1. **Given** a tool capability appears in the normalized catalog with family, method, and schema descriptions, **When** the staged controller renders prompts, **Then** that capability is available without hand-written prompt wiring.
2. **Given** the runtime executes an ongoing-task or other autonomous action, **When** the staged controller is used there, **Then** VAD and heuristic injection continue to function while replay admission suppression rules remain intact.
3. **Given** maintainers add a future tool or method, **When** they consult the server docs, **Then** the required family/method/contract metadata layers are explicitly documented.

### Edge Cases

- The provider returns malformed JSON or a selection outside the offered family, method, or payload contract.
- A tool family contains one method, but the model still chooses `back` or `complete`.
- A method schema has optional nested objects or arrays that require short but complete field descriptions.
- A tool result arrives after the user interrupts the request and the staged loop must abort cleanly.
- A background mode such as ongoing-task execution or cognitive replay-adjacent processing must use the staged loop but must not admit new replay episodes.
- Some tools expose native DeepSeek tool-call semantics, but the staged controller still needs explicit CPU-side validation and prompt assembly.
- The family selected for an immediate user reply is the Telegram family, and the runtime must still treat it as a normal family→method→payload path instead of a bypass.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The server MUST define one concise reusable core system prompt that presents Vicuña as a helpful personal concierge with access to multiple tool families.
- **FR-002**: The staged controller MUST begin active execution by exposing only normalized tool-family names plus short descriptions and MUST require a single JSON family choice.
- **FR-003**: The staged controller MUST support a method-selection stage that exposes only methods for the chosen family plus short descriptions and MUST require a single JSON method choice.
- **FR-004**: The staged controller MUST support a payload-construction stage that exposes the chosen method’s typed input contract, including short descriptions for each field, and MUST require a single JSON payload choice.
- **FR-005**: The runtime MUST parse and validate each staged JSON response against the offered options before advancing or executing a method.
- **FR-006**: The method-selection and payload-construction stages MUST expose and honor explicit `back` navigation sentinels.
- **FR-007**: The staged controller MUST expose and honor an explicit completion path that ends the active loop and transitions the runtime toward the existing background replay/idle flow.
- **FR-008**: After every executed tool method and returned tool observation, the runtime MUST restart the active loop at family selection.
- **FR-009**: The staged controller MUST be driven from a reusable catalog abstraction that can fetch normalized family summaries, method summaries, and method contracts from shared tool metadata.
- **FR-010**: Future tools and methods MUST be documented as requiring the three metadata layers: family metadata, method metadata, and field-described method contracts.
- **FR-011**: The staged controller MUST use reusable prompt-assembly helpers for family selection, method selection, and payload construction rather than embedding ad hoc prompt text in each caller.
- **FR-012**: Existing VAD injection, reasoning replay, and heuristic guidance MUST remain additive and MUST continue to run at the staged checkpoints without mutating preserved `reasoning_content`.
- **FR-013**: The staged controller MUST be used in all runtime modes that currently invoke provider reasoning for tool-backed work, including foreground and pre-idle/background execution paths, unless a mode is explicitly documented as exempt.
- **FR-014**: Background paths that already suppress replay admission, including ongoing-task execution and replay-related flows, MUST continue suppressing replay admission while using the staged controller.
- **FR-015**: DeepSeek integration MUST reconcile staged selection with native tool support by keeping family/method/payload selection as CPU-side JSON-guided turns and reserving actual tool execution semantics for the final validated method call.
- **FR-016**: The implementation MUST update tests for staged family selection, method selection, payload validation, back navigation, completion, tool-result restart, and additive coexistence with VAD and heuristic guidance.
- **FR-017**: The implementation MUST update developer/operator docs and architecture guidance for the staged controller, metadata requirements, and validation commands.

### Key Entities *(include if feature involves data)*

- **Tool Family Summary**: The normalized high-level name and concise description shown during family selection.
- **Tool Method Summary**: The normalized method name and concise description shown during method selection for one chosen family.
- **Tool Method Contract**: The typed input schema plus short field descriptions shown during payload construction.
- **Staged Tool Loop State**: The explicit controller state that tracks current stage, selected family, selected method, navigation choices, completion, and last tool observation.
- **Staged Selection Response**: The parsed JSON selection object returned by the provider for one stage.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Automated tests prove that one tool-backed request executes through family selection, method selection, payload construction, method execution, and a restart to family selection after the tool result.
- **SC-002**: Automated tests prove that `back` navigation works from method selection and payload construction without executing invalid or partial tool calls.
- **SC-003**: Automated tests prove that the explicit completion path ends the active loop and hands control back to the existing replay/idle pipeline.
- **SC-004**: Automated tests prove that staged payload assembly preserves additive VAD and heuristic guidance behavior and preserves replayed `reasoning_content`.
- **SC-005**: Updated docs and contracts make the required family/method/contract metadata layers explicit for future tool authors.
