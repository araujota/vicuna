# Contract: Self-Model Translation Runtime

## Purpose

Define the explicit runtime surfaces that compile hidden mathematical
self-model state into a bounded reportable DMN prompt revision.

## Translation Lifecycle

1. A typed self-state update produces or contributes to a `SelfModelRevision`.
2. Explicit materiality policy decides whether prompt regeneration is required.
3. The runtime builds a bounded `SelfModelTranslationInput`.
4. The runtime maps that input into ordered `ReportableConceptFrame` entries.
5. The runtime realizes those concept frames into a `DmnPromptRevision`.
6. The DMN planner/tool runner starts or supersedes an episode bound to that
   prompt revision.

## Required Runtime Surfaces

- `SelfModelRevision`
  - monotonic revision identity
  - changed-domain mask
  - materiality outcome
- `SelfModelTranslationInput`
  - bounded typed snapshot used for translation
- `ReportableConceptFrame`
  - explicit intermediate concept representation
- `DmnPromptRevision`
  - final reportable prompt artifact with lineage
- `DmnPromptSupersessionTrace`
  - explains why one prompt revision replaced another

## Policy Rules

- Hidden mathematical self-state MUST remain non-directly inspectable by the
  DMN prompt consumer.
- Translation MUST be driven by explicit CPU-side concept selection and
  realization policy.
- Materiality thresholds MUST be explicit and testable.
- The translator MUST accept new self-model extension fields only through
  explicit mapping rules.
- Prompt revisions MUST be lineage-tracked and bounded.

## Non-Goals

- This feature does not flatten the hidden self-model into raw text.
- This feature does not require exposing full self-state internals to the model.
- This feature does not require a separate DMN executor distinct from the
  existing planner/tool runner.
