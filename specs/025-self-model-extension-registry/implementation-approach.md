# Implementation Approach: Self-Model Extension Registry

## Architectural Shape

The system should be:

```text
authored self-model core
+ bounded extension registry
+ fixed extension summary
-> self-model update
-> favorable / gain control
```

This keeps the authored core as the genetic prior while allowing bounded
runtime-discovered or tool-authored additions.

## Extension Application

1. Validate and clamp the extension write.
2. Upsert it into a bounded slot by key.
3. During self-model recomputation:
   - compute contextual activation for each extension
   - apply a small explicit domain-specific adjustment to relevant horizon
     fields
   - accumulate extension summary statistics
4. Feed the extension summary into:
   - self-model inspection APIs
   - recovery/allostatic summary logic when allowed
   - the functional gating MLP input tail

## Hard-Memory Promotion

1. Execute a normal hard-memory query.
2. Convert each top hit into a candidate `MEMORY_CONTEXT` extension.
3. Run a bounded shadow application of that candidate against current self-model
   horizon state.
4. Score whether the candidate would reduce epistemic/goal/efficiency pressure
   enough to justify insertion.
5. Promote the best candidates above threshold into the extension registry and
   record a typed trace.

## Tool Extension Contract

Tool integrations should not write free-form formulas. They should write typed
extensions through one API:

- choose a domain
- choose `MEMORY_CONTEXT` or `SCALAR_PARAM`
- set flags for gain/allostasis
- optionally set a desired state

This keeps tool extensibility inspectable and safe.

## Backward Compatibility

- The authored core remains intact.
- Existing favorable-state dimensions remain intact.
- The gating MLP sees a fixed additional summary tail, not a variable-length
  extension list.
- Hard-memory context additions are kept out of allostasis unless explicitly
  authored otherwise through tool scalar parameters.
