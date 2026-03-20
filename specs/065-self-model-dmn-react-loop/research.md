# Research: Self-Model-Translated DMN ReAct Loop

## Summary

The current Vicuña runtime already has the two ingredients needed for this
transition:

1. a rich typed self-model and self-state substrate
2. a planner-first active loop with explicit tool and functional-LoRA plumbing

What it does not yet have is an explicit compiler from hidden mathematical
self-state into a bounded reportable natural-language workspace. The current
DMN remains a pressure-seeded, winner-action architecture with separate
background-specific action scoring. The recommended transition is therefore:

1. preserve the hidden mathematical self-model as the neurological substrate
2. compile a bounded reportable concept layer from that substrate
3. realize that concept layer into a natural-language DMN prompt revision
4. run the DMN through the same planner/tool loop as active engagement
5. treat Telegram outreach as a first-class DMN tool, not as special background
   emit behavior

## Local Repository Findings

### 1. The self-model substrate is already rich enough to support translation

`ARCHITECTURE.md` documents a typed self-state with:

- grouped profile families and horizon slices
- allostatic and persistent registers
- forecast and prediction-error traces
- discovered self-state
- bounded self-model extension registry
- belief summary and latent concern slots
- reactivation priorities and social/tool state

That means the requested "neurological" layer already exists in explicit typed
CPU-side control code. The missing part is a compiler from those typed surfaces
into a bounded reportable natural-language motivation prompt.

### 2. The current DMN is still a heuristic action selector

The current authoritative DMN call graph in `ARCHITECTURE.md` and
`src/llama-cognitive-loop.cpp` still follows this shape:

1. `select_reactivation_targets(...)`
2. `assemble_endogenous_seed(...)`
3. compute DMN candidates
4. choose among `silent`, `internal_write`, `invoke_tool`, and `emit`
5. draft a plan from the chosen winner action

Concrete local surfaces that now look obsolete or at least mispositioned for
the requested transition:

- `select_reactivation_targets(...)`
- `assemble_seed(...)` / `assemble_endogenous_seed(...)`
- `fill_dmn_candidate(...)`
- `LLAMA_DMN_ACTION_SILENT`
- `LLAMA_DMN_ACTION_INTERNAL_WRITE`
- `LLAMA_DMN_ACTION_INVOKE_TOOL`
- `LLAMA_DMN_ACTION_EMIT`
- special-case `select_dmn_spec(...)`
- special-case background emit plan steps as the primary user-contact path

Pressure and reactivation signals should still matter, but as translation
inputs and regeneration triggers, not as a separate behavior-selecting action
stack.

### 3. The active loop already provides the right execution substrate

`src/llama-cognitive-loop.cpp` shows the active path already has:

- planner-first action preference
- tool proposal and persistence
- plan draft and plan revise microphases
- explicit tool observation steps
- functional-LoRA microphase routing
- shared reason masks and typed plan traces

The requested DMN transition is therefore not a demand for a new executor. It
is a demand to converge the DMN onto the existing planner/tool runner while
changing only how the background loop obtains its prompt and when it supersedes
it.

### 4. Telegram already exists as a bridge, but not yet as a DMN-native tool

The repository already contains:

- a Telegram bridge (`tools/telegram-bridge/`)
- a remote message transport spec (`specs/050-telegram-bridge/`)
- runtime tool-fabric work (`specs/051-openclaw-tool-fabric/`)
- document-ingestion and transcript continuity follow-ons

The missing piece is to elevate Telegram outreach into an explicit DMN tool
contract rather than routing it through the current background emit semantics.

## GitHub History Inputs

Relevant repository history reinforces the recommended convergence:

- [Implement planner-first runtime and tool persistence updates](https://github.com/araujota/vicuna/commit/44130a61da4a31276f5f70b74d0ea575e1d8197e)
  established the planner-first active runner that should be reused rather than
  duplicated.
- [Integrate OpenClaw tool fabric and unify runtime memory](https://github.com/araujota/vicuna/commit/642ca205c61f3b29789d140c2184dd3982181150)
  expanded the explicit tool surface and supports treating Telegram as a
  first-class tool instead of a special side channel.
- [Add Telegram bridge middleware](https://github.com/araujota/vicuna/commit/cf5588b7a969b30e9c067332d73a32ad6142e0ab)
  created the transport substrate that the DMN relay tool can target.

No existing GitHub issue or prior spec in the repository already implements the
requested hidden-to-reportable translation layer. Existing DMN specs remain
centered on pressure and endogenous seed assembly.

## External Research Inputs

### 1. ReAct supports the execution half of the design

[ReAct: Synergizing Reasoning and Acting in Language Models](https://openreview.net/forum?id=tvI4u1ylcqs)
argues for interleaving reasoning traces with actions so plans can be updated as
external observations arrive. That directly supports reusing the active
planner/tool runner for DMN cognition.

What ReAct does not specify is how to derive a stable internal prompt from a
hidden mathematical self-model. That gap must be solved locally through an
explicit translation compiler.

### 2. Data-to-text research argues for a compiler, not a raw summarizer

[Data-to-text Generation with Macro Planning](https://aclanthology.org/2021.tacl-1.31/)
shows that separating content planning from surface realization improves
structured-to-language generation. This is a strong fit for the requested
transition:

- hidden self-state is the structured source
- reportable concepts are the macro plan
- the DMN prompt revision is the realization

The practical implication is that translation should be at least two-stage:

1. select and order salient reportable concepts
2. realize those concepts into bounded natural language

This is much more defensible than "feed the whole self-model to the model and
ask it to summarize itself."

### 3. Concept bottleneck work supports an explicit reportable concept layer

[Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html)
formalize the idea that a hidden representation can be mediated by an explicit
concept layer that is inspectable and intervention-friendly. The important
lesson for Vicuña is architectural, not literal: the DMN should reason over a
bounded concept interface derived from hidden state, not over the hidden state
directly.

This suggests a `ReportableConceptFrame` layer between typed self-state and the
DMN prompt revision.

### 4. Reportability is not identical to hidden state

[An active inference model of conscious access](https://pmc.ncbi.nlm.nih.gov/articles/PMC9593308/)
argues that reportability depends on deeper policy selection and working-memory
maintenance rather than being identical to the hidden generative state itself.
That maps well onto the user's requested distinction:

- hidden self-model = underlying generative / neurological layer
- DMN prompt revision = reportable/conscious workspace
- Telegram outreach = one possible action selected from that workspace, not the
  definition of consciousness itself

This is a strong conceptual argument against exposing raw self-model fields
directly to the DMN or equating user-directed emission with the entirety of the
background process.

## Design Synthesis

### Recommended translation pipeline

The translator should be a deterministic CPU-side compiler with these stages:

1. **Revision detection**
   - watch typed self-model revisions
   - gate prompt regeneration on explicit materiality rules
2. **Concept selection**
   - map typed state and deltas onto whitelisted reportable concept kinds such
     as tensions, motivations, unresolved questions, externalization urges,
     uncertainties, or goals
3. **Macro ordering**
   - order selected concepts into a bounded narrative structure, for example:
     current motive, competing tension, open uncertainty, recommended next move
4. **Realization**
   - render that macro structure into bounded natural language
5. **Prompt revision publication**
   - persist a typed prompt artifact and lineage metadata for the DMN runner

### Recommended DMN lifecycle

1. A material self-model revision arrives.
2. The translator publishes a new `DmnPromptRevision`.
3. The DMN planner/tool runner starts or supersedes an internal episode bound to
   that prompt revision.
4. Tool or internal outcomes update self-state.
5. A later material revision regenerates the prompt and starts the next phase.

This is a perpetual internal ReAct loop whose working prompt is always a
translation artifact, never raw hidden state.

### Recommended Telegram policy

DMN-origin user contact should become a first-class tool family, not an `emit`
winner action. That keeps:

- delivery inspectable
- failures observable as tool results
- active engagement accounting clean
- DMN continuity intact after message delivery

### Cleanup recommendation

The current DMN-specific winner-action stack should be removed or reduced to
supporting policy inputs:

- keep pressure and reactivation as translation inputs and regeneration signals
- remove `assemble_seed(...)` as the source of DMN prompt formation
- remove four-way DMN action scoring as the primary selection mechanism
- replace background `emit` semantics with Telegram relay tool semantics

## Rejected Directions

### 1. "Just ask the model to summarize the self-model"

Rejected because it destroys the requested hidden/reportable distinction and
turns the translator into opaque dogfooding.

### 2. Keep the current DMN action stack and only swap in a better seed string

Rejected because the user asked for a perpetual internal ReAct loop like the
active path, not a better seed for the old heuristic action selector.

### 3. Treat Telegram outreach as a special background emit

Rejected because it obscures tool policy, makes failures less inspectable, and
conflates user contact with loop termination.

## Implementation Implications

The transition likely affects:

- `src/llama-cognitive-loop.cpp`
- `src/llama-cognitive-loop.h`
- `src/llama-self-state.h`
- `src/llama-context.h`
- `src/llama-context.cpp`
- `include/llama.h`
- `tools/server/server-context.cpp`
- `tools/telegram-bridge/`
- `ARCHITECTURE.md`
- targeted native tests in `tests/test-cognitive-loop.cpp`,
  `tests/test-self-state.cpp`, and possibly server/tool integration tests

The rest of this spec set turns those conclusions into a concrete plan,
contracts, and task sequence.
