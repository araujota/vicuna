# Quickstart: Extending The Self-Model Safely

## Default Runtime Behavior

- The authored self-model core is always present.
- Hard-memory query results may be promoted into self-model as bounded
  `MEMORY_CONTEXT` extensions after a counterfactual scoring step.
- These default hard-memory additions affect gain control context but do not
  become allostatic objectives.

## Tool Integration Path

To let a tool add self-model state:

1. Build a `llama_self_model_extension_update`.
2. Choose a stable `key`.
3. Pick the right `domain`.
4. Set `AFFECT_GAIN` if the extension should bias future control.
5. Set `AFFECT_ALLOSTASIS` only if the extension is a true internal objective
   with a meaningful desired state.
6. Upsert it through the public API instead of encoding it only in tool text.

## Accuracy Rules

- Use stable keys so repeated tool writes update the same extension instead of
  creating duplicates.
- Keep values normalized to `[0, 1]`.
- Set `confidence` and `salience` conservatively.
- Do not mark retrieved memories as allostatic objectives by default.
- Only assign desired states to tool-authored scalar parameters that represent
  real internal goals or constraints.
