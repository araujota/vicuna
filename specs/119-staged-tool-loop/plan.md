# Implementation Plan: Ubiquitous Staged Tool Family/Method/Payload Orchestration

**Branch**: `[119-staged-tool-loop]` | **Date**: 2026-03-25 | **Spec**: [/Users/tyleraraujo/vicuna/specs/119-staged-tool-loop/spec.md](/Users/tyleraraujo/vicuna/specs/119-staged-tool-loop/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/119-staged-tool-loop/spec.md`

## Summary

Replace direct provider-side tool-choice behavior with an explicit staged controller in `tools/server`: the provider chooses a tool family, then a method, then a typed payload via strict JSON turns; the server validates each stage, executes the final method, re-injects the observation, and restarts from family selection until the provider chooses completion.

## Technical Context

**Language/Version**: C++17 native server/runtime code plus Python provider integration tests  
**Primary Dependencies**: existing `nlohmann::json`, current provider request assembly, current emotive runtime, existing tool catalog metadata in `tools/openclaw-harness`  
**Storage**: existing in-memory request/controller state plus existing hard-memory/replay persistence  
**Testing**: focused Python provider-mode tests, native server build validation  
**Target Platform**: macOS/Linux native `llama-server` provider-first runtime  
**Project Type**: Native HTTP server/runtime feature  
**Performance Goals**: bounded prompt assembly and stage parsing, no additional network hops beyond the deliberate staged provider turns, no regression in replay/idle worker stability  
**Constraints**: preserve `reasoning_content`, keep orchestration explicit in CPU-side code, avoid duplicating tool metadata, preserve additive VAD/heuristic guidance, keep background replay suppression rules intact  
**Scale/Scope**: `tools/server/server.cpp`, `tools/server/server-deepseek.cpp`, possible shared metadata import helpers, provider tests, docs, and feature artifacts

## Constitution Check

*GATE: Must pass before implementation. Re-check after design.*

- **Runtime Policy**: Pass. Stage transitions, completion, navigation, prompt assembly, and execution remain explicit in server-side control code.
- **Typed State**: Pass. The design introduces typed staged-loop state and normalized catalog entities rather than free-form prompt branching.
- **Bounded Behavior**: Pass. The runtime exposes only one stage at a time and validates exact JSON choices before progression.
- **Validation**: Required. Add tests for normal staged execution, back navigation, completion, restart-after-tool-result, and additive VAD/heuristic coexistence.
- **Documentation & Scope**: Pass. Update server docs and architecture notes with the required family/method/contract metadata policy.

## Research Summary

- DeepSeek JSON mode works best when the runtime sets `response_format={"type":"json_object"}` and explicitly instructs the model to emit JSON with an example shape. That fits the staged-selection turns.
- DeepSeek thinking/tool-use requires preserving `reasoning_content` across same-turn continuations. The staged controller should therefore add new selection prompts and guidance additively without mutating replayed reasoning text.
- ReAct’s benefit comes from explicit think→act→observe iteration, not from a single monolithic tool-selection turn. The staged controller preserves that by inserting finer-grained action checkpoints while still grounding on real tool observations.
- Semantic Kernel and OpenAI Agents both model tools as grouped metadata surfaces: plugin/family, function/method, and typed parameter schema. That matches the existing OpenClaw catalog structure and argues against inventing a second metadata system.
- Native provider tool calling should remain the execution substrate for the final validated method call only. Using native tool calls for family/method meta-selection would hide policy inside the provider and reduce the number of runtime checkpoints the user explicitly wants.

## Project Structure

### Documentation (this feature)

```text
specs/119-staged-tool-loop/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/
│   └── requirements.md
├── contracts/
│   └── staged-tool-loop.md
└── tasks.md
```

### Source Code (repository root)

```text
tools/server/
├── README-dev.md
├── README.md
├── server.cpp
├── server-deepseek.cpp
└── tests/unit/test_deepseek_provider.py

tools/openclaw-harness/
├── README.md
└── src/
    ├── catalog.ts
    └── contracts.ts
```

**Structure Decision**: Normalize staged prompt metadata from the existing capability catalog instead of hand-maintaining a second server-local registry. Keep the controller loop in `server.cpp`, where request assembly, VAD injection, heuristic injection, ongoing-task execution, and replay workers already converge.

## Design

### Catalog Normalization

- Build a server-side normalized view over the existing capability catalog:
  - family id, family name, family description
  - method name, method description, capability id, dispatch target
  - input schema with required field descriptions
- Reject or omit catalog entries that do not satisfy the required metadata shape.

### Staged Controller

- Introduce an explicit controller with stages:
  - `family_select`
  - `method_select`
  - `payload_build`
  - `execute`
  - `complete`
- Add synthetic navigation sentinels:
  - `back` at method/payload stages
  - `complete` at method stage
- Each stage produces:
  - one reusable prompt segment
  - one strict JSON response contract
  - one typed parser/validator

### Prompt Assembly

- Core system prompt: concise concierge identity with tool access.
- Family prompt: only family options plus concise descriptions.
- Method prompt: only methods for the chosen family plus concise descriptions, plus `back` and `complete`.
- Payload prompt: method contract with short field descriptions, plus `back`.
- Continue adding VAD and heuristic guidance as separate additive `system` messages when applicable.

### Execution Semantics

- After payload validation, execute the selected capability through the current runtime path.
- Convert the tool observation back into the active message history.
- Restart the loop at family selection after every real tool result.
- When the provider chooses `complete`, terminate the active loop and hand control back to the existing background replay/idle pipeline.

### Ubiquity Across Modes

- Foreground provider requests use the staged controller.
- Ongoing-task execution and other autonomous active loops use the same staged controller.
- Cognitive replay prompt/compression flows remain special-purpose prompts, but any tool-backed work they trigger should use the same staged controller if they execute tools.
- Replay-admission suppression flags remain unchanged in modes that already require them.

### DeepSeek Reconciliation

- Keep staged selections as ordinary JSON chat turns, not native `tool_calls`.
- Preserve existing DeepSeek native tool-call compatibility for external/provider-facing message translation where needed, but route internal autonomous orchestration through staged JSON turns.
- Continue replaying `reasoning_content` exactly on tool-continuation requests and never rewrite it when adding staged prompts or guidance messages.

## Phases

### Phase 0: Research and Contracts

- Finalize catalog normalization, stage contracts, prompt policy, and DeepSeek compatibility rules.

### Phase 1: Catalog and Controller Foundations

- Add normalized family/method/contract extraction.
- Add typed staged controller state and reusable prompt builders.

### Phase 2: Foreground Execution

- Wire the staged controller into the main provider/tool loop.
- Preserve additive VAD and heuristic guidance.

### Phase 3: Background/Ubiquitous Execution

- Route ongoing-task and related autonomous active flows through the same staged controller.
- Preserve replay suppression semantics.

### Phase 4: Validation and Docs

- Add tests for staged execution, navigation, completion, and restart.
- Update docs and architecture guidance.

## Validation Strategy

- Run focused provider tests for:
  - family selection
  - method selection
  - payload validation
  - back navigation
  - completion
  - restart after tool observation
  - coexistence with VAD and heuristic guidance
- Rebuild `llama-server`.
- Run the full provider test file after targeted tests pass.

## Complexity Tracking

No constitution exception is expected.
