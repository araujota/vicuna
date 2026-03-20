# Implementation Plan: Self-Model-Translated DMN ReAct Loop

**Branch**: `065-self-model-dmn-react-loop (spec-only on current 060-service-user-migration worktree)` | **Date**: 2026-03-20 | **Spec**: [/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/spec.md](/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/spec.md)
**Input**: Feature specification from `/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/spec.md`

## Summary

Replace the current pressure-seeded DMN winner-action architecture with a
prompt-revision architecture. A typed self-model revision compiler will map the
hidden mathematical self-model into reportable concept frames and then into a
bounded natural-language `DmnPromptRevision`. The DMN will consume that prompt
through the same planner-first ReAct runner used by active engagement. When the
self-model changes materially, the translator will emit a new prompt revision
and supersede the current DMN episode. User-facing outreach will move from
special background emit behavior to a first-class Telegram relay tool, and the
old endogenous-seed plus four-way DMN action stack will be removed or narrowed
to explicit translation-trigger policy.

## Technical Context

**Language/Version**: C++17 for runtime/core, existing Node.js bridge surface for Telegram  
**Primary Dependencies**: Existing cognitive loop runtime, self-state surfaces, planner/tool runner, OpenClaw tool fabric, Telegram bridge middleware  
**Storage**: In-memory typed runtime state plus existing server export/import and bridge state files where applicable  
**Testing**: Native C++ test binaries such as `tests/test-cognitive-loop.cpp` and `tests/test-self-state.cpp`, plus targeted server/bridge integration validation  
**Target Platform**: CPU-managed runtime and server paths on Linux/macOS development hosts  
**Project Type**: Native inference/runtime library plus local bridge/service integration  
**Performance Goals**: Keep prompt regeneration bounded by explicit materiality thresholds, avoid idle-loop churn, preserve bounded DMN memory and tool concurrency  
**Constraints**: Runtime policy must stay explicit in CPU-side control code; hidden mathematical self-state must remain non-directly inspectable by the DMN prompt consumer; active and DMN accounting must stay distinct  
**Scale/Scope**: One shared planner/tool substrate for both active and DMN loops, plus a new translation layer and DMN-native Telegram tool contract

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Runtime Policy**: Pass. Translation, materiality gating, prompt revision
  lineage, DMN supersession, and Telegram relay tool policy will live in
  inspectable CPU-side control code rather than hidden prompts or opaque helper
  text.
- **Typed State**: Pass. The design introduces typed revisions, translation
  traces, prompt artifacts, DMN episode lineage, and relay request/result
  structs.
- **Bounded Memory**: Pass. Translation input is bounded, prompt revisions are
  lineage-tracked and replaceable, and the DMN uses existing bounded planner
  and tool state rather than introducing unbounded text history.
- **Validation**: Pass. The plan requires targeted native tests for revision
  gating, translation determinism, DMN episode supersession, and Telegram tool
  integration, plus operator-facing runtime validation.
- **Documentation & Scope**: Pass. `ARCHITECTURE.md`, runtime headers, server
  docs, Telegram bridge docs, and the full Spec Kit artifact set are all in
  scope. No new third-party dependency is required by the design itself.

## Project Structure

### Documentation (this feature)

```text
specs/065-self-model-dmn-react-loop/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── dmn-telegram-relay-tool.md
│   └── self-model-translation-runtime.md
└── tasks.md
```

### Source Code (repository root)

```text
include/
└── llama.h

src/
├── llama-cognitive-loop.cpp
├── llama-cognitive-loop.h
├── llama-context.cpp
├── llama-context.h
├── llama-self-state.cpp
└── llama-self-state.h

tests/
├── test-cognitive-loop.cpp
└── test-self-state.cpp

tools/
├── server/
│   ├── README-dev.md
│   └── server-context.cpp
└── telegram-bridge/
    ├── README.md
    └── lib.mjs
```

**Structure Decision**: The feature stays inside the existing runtime, server,
and bridge layout because the change is a control-policy convergence inside the
cognitive loop rather than a new standalone subsystem.

## Phase 0: Research Conclusions

- Reuse the active planner/tool runner as the DMN execution substrate instead
  of extending the current winner-action stack.
- Introduce an explicit translation compiler from hidden self-state into
  reportable concept frames and then natural-language prompt revisions.
- Treat pressure, reactivation, contradiction, tool state, and social state as
  translation inputs and regeneration triggers rather than as direct DMN action
  scores.
- Make prompt regeneration revision-driven with explicit materiality thresholds
  and supersession rules.
- Add Telegram relay as a first-class tool so DMN-origin user contact remains a
  tool observation rather than a special background emit path.
- Remove or repurpose obsolete DMN heuristic surfaces that would conflict with
  the new architecture.

## Phase 1: Design & Contracts

### Data Model

See [/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/data-model.md](/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/data-model.md).

### Contract Surface

See [/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/contracts/self-model-translation-runtime.md](/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/contracts/self-model-translation-runtime.md) and [/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/contracts/dmn-telegram-relay-tool.md](/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/contracts/dmn-telegram-relay-tool.md).

### Quickstart / Validation

See [/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/quickstart.md](/Users/tyleraraujo/vicuna/specs/065-self-model-dmn-react-loop/quickstart.md).

## Implementation Design

### 1. Self-model revision and translation compiler

- Add a typed `SelfModelRevision` surface that increments when material
  self-state, discovered-state, extension, or social/tool summaries change.
- Build a bounded `SelfModelTranslationInput` from explicit typed sources:
  contradiction, uncertainty, favorable divergence, belief summary, extension
  summary, reactivation priorities, tool/social state, continuation, open goals
  and recent significant deltas.
- Compile that input through two explicit stages:
  - reportable concept selection and ranking
  - bounded natural-language realization
- Persist a `DmnPromptRevision` with source revision lineage and translation
  trace.

### 2. DMN episode convergence onto the planner/tool runner

- Replace DMN-specific winner-action selection as the primary policy path.
- Start DMN episodes from `DmnPromptRevision` artifacts using the same plan
  draft, plan revise, tool invoke, and tool observe semantics already used by
  the active loop.
- Keep DMN origin distinct through typed origin metadata, budget rules, and
  output policy, but not through a separate action architecture.
- Add explicit episode supersession when a newer prompt revision supersedes the
  currently executing DMN episode.

### 3. Telegram relay becomes a first-class DMN tool

- Add an explicit tool kind or named capability for DMN-origin Telegram relay.
- Define request metadata such as intent kind, urgency, text payload, routing
  mode, dedupe key, and whether the relay is advisory/question/conclusion.
- Integrate tool result handling so delivery success or failure becomes a normal
  tool observation available to later DMN reasoning.
- Preserve existing bridge anti-spam and registration semantics, but separate
  DMN relay accounting from active engagement response accounting.

### 4. Cleanup inventory for obsolete code

The following current surfaces should be removed, narrowed, or repurposed:

- `select_reactivation_targets(...)` as a direct action-selection precursor
- `assemble_seed(...)` / `assemble_endogenous_seed(...)` as the DMN prompt
  source
- `fill_dmn_candidate(...)` and the dedicated DMN candidate array
- `LLAMA_DMN_ACTION_SILENT`
- `LLAMA_DMN_ACTION_INTERNAL_WRITE`
- `LLAMA_DMN_ACTION_INVOKE_TOOL`
- `LLAMA_DMN_ACTION_EMIT`
- special-case `select_dmn_spec(...)` that exists only because DMN has its own
  action stack
- background emit-specific plan-step semantics where Telegram relay should now
  be represented as a tool

Pressure, reactivation, and social/tool surfaces are still valuable, but they
should survive as translator inputs, materiality heuristics, or planner/tool
features rather than as a parallel DMN behavior-selection language.

### 5. Observability and lineage

- Add typed traces for:
  - self-model revision detection
  - translation input selection
  - concept ranking
  - prompt realization
  - prompt revision supersession
  - DMN episode lineage
  - Telegram relay tool requests and outcomes
- Update server inspection surfaces so operators can inspect hidden-to-reportable
  transitions without exposing raw hidden state directly to the DMN prompt
  consumer.

## Validation Strategy

- `tests/test-self-state.cpp`
  - verify material revision detection and bounded translation input selection
  - verify extension whitelisting and non-whitelisted exclusion behavior
- `tests/test-cognitive-loop.cpp`
  - verify DMN prompt revision creation from self-model changes
  - verify DMN episode supersession on newer prompt revisions
  - verify DMN uses shared plan/tool runner semantics instead of winner-action
    scoring
- server/runtime validation
  - verify typed traces for prompt revisions and DMN episode lineage are exposed
  - verify DMN-origin Telegram relay is recorded as a tool observation
- bridge/tool validation
  - verify DMN-origin relay does not count as active engagement completion
  - verify relay failures remain visible to later DMN reasoning
- manual validation from `quickstart.md`
  - rebuild, inspect health, inspect journal traces, and verify a DMN-origin
    Telegram question does not stop DMN evolution

## Post-Design Constitution Check

- **Runtime Policy**: Still passes. Translation and supersession rules remain
  explicit and typed.
- **Typed State**: Still passes. New revision, concept, prompt, episode, and
  relay surfaces are all modeled as explicit structs and traces.
- **Bounded Memory**: Still passes. Prompt revisions are revision-driven rather
  than append-only and reuse the existing bounded plan/tool substrate.
- **Validation**: Still passes. The planned tests exercise both hidden-to
  reportable translation and shared-runner convergence.
- **Documentation & Scope**: Still passes. The work explicitly includes
  architecture cleanup and operator docs, not just code.
