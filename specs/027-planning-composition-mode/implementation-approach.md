# Implementation Approach

1. Extend the public cognitive API with bounded plan types.
2. Add a planning/composition functional LoRA family and microphases.
3. Build a shared planner/composer in `llama-cognitive-loop.cpp`.
4. Convert active and DMN winners into plan drafts rather than immediate
   terminal branches.
5. Execute the plan’s next step into existing command queue and tool proposal
   surfaces.
6. Reuse tool-result resumption to revise the plan and settle the planning
   family update.

The old loop phases remain because they still encode runtime lifecycle
(`ASSEMBLE`, `PREPARE_TOOL`, `WAIT_TOOL`, `OBSERVE`, `FINISH`). What changes is
that they are no longer the semantic decision model.
