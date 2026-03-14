# Implementation Approach: Hard Memory Primitives

## Core Strategy

1. Extend the hard-memory public type system.
2. Refactor archival from “one event string” into “one bounded primitive batch”.
3. Add explicit emitters for:
   - self-state event fragments
   - loop trajectories
   - loop outcomes
   - tool observations
   - user-model fragments
   - self-model fragments
4. Parse typed metadata back out of query hits.
5. Summarize retrieval into a fixed-width structure and feed it into:
   - self-model promotion
   - functional-gating observation

## Why This Is The Elegant Cut

- It uses the runtime artifacts Vicuña already computes.
- It keeps storage and retrieval policy explicit in CPU-side code.
- It avoids turning hard memory into a second hidden prompt system.
- It stays compatible with the existing self-model extension and gain-gradient
  design by summarizing retrieval instead of exposing variable-length memory
  lists directly to the controller.

## Archive Policy

### Event path

- Always candidate for `EVENT_FRAGMENT`
- Optional `USER_MODEL` or `SELF_MODEL_FRAGMENT` when the event materially
  changes those surfaces

### Active loop

- Archive only after episode settlement or tool-result integration
- Build one trajectory primitive and one outcome primitive
- Add tool observation when bash result exists

### DMN loop

- Archive only when pressure admission led to a real cycle
- Build trajectory/outcome primitives
- Add self-model fragment summarizing counterfactual/governance/remediation or
  temporal self-improvement when signal is present

## Retrieval Cooperation

- Query parsing reconstructs primitive kinds/domains/tags from metadata
- Retrieval summary aggregates similarity-weighted counts and domain support
- Self-model promotion uses:
  - primitive kind
  - primitive domain
  - similarity
  - gain bias
  - allostatic relevance
- Functional gating gets the retrieval summary as extra fixed-width input

## Bounds

- explicit primitive cap per batch
- explicit tag cap per primitive
- bounded string copies everywhere
- no new unbounded in-memory caches
