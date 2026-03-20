# Data Model: Self-Model-Translated DMN ReAct Loop

## Entity: Self Model Revision

Represents one material change to the hidden mathematical self-model.

### Fields

- `revision_id`
- `created_at_us`
- `changed_domain_mask`
- `source_event_id`
- `source_loop_origin`
- `materiality_score`
- `requires_prompt_regen`
- `supersedes_revision_id`

### Invariants

- `revision_id` MUST increase monotonically.
- `requires_prompt_regen` MUST be derived from explicit materiality policy, not
  from free-form prompt text.
- A revision MAY exist without forcing prompt regeneration when the change is
  below the configured materiality threshold.

## Entity: Self Model Translation Input

The bounded typed snapshot consumed by the translation compiler.

### Fields

- `revision_id`
- `contradiction_summary`
- `uncertainty_summary`
- `favorable_divergence_summary`
- `belief_summary`
- `extension_summary`
- `reactivation_targets`
- `social_state`
- `tool_state`
- `goal_summary`
- `continuation_summary`
- `recent_significant_deltas`
- `bounded_working_handles`

### Invariants

- The input MUST be constructed only from explicit typed self-state surfaces.
- The input MUST be bounded in cardinality and tokenizable summary size.
- Newly discovered extension fields MUST NOT reach translation unless mapped
  through an explicit whitelist or translation rule.

## Entity: Reportable Concept Frame

An intermediate concept selected from the hidden self-model for reportable DMN
use.

### Fields

- `concept_id`
- `revision_id`
- `concept_kind`
- `salience`
- `confidence`
- `signed_pressure`
- `evidence_mask`
- `user_contact_affordance`
- `tool_affordance`
- `realization_priority`
- `rendered_hint`

### Relationships

- Many `Reportable Concept Frame` entries may derive from one
  `Self Model Translation Input`.
- One `DmnPromptRevision` is realized from an ordered set of concept frames.

### Invariants

- Concept frames MUST be derived from explicit mapping policy, not from raw
  prompt generation over the full hidden self-model.
- `rendered_hint` MUST stay bounded and must not become a hidden-state dump.

## Entity: DMN Prompt Revision

The natural-language prompt artifact consumed by the DMN ReAct runner.

### Fields

- `prompt_revision_id`
- `source_revision_id`
- `created_at_us`
- `prompt_hash`
- `macro_outline`
- `rendered_prompt`
- `supersedes_prompt_revision_id`
- `concept_count`
- `status`

### Invariants

- A prompt revision MUST be traceable back to exactly one `Self Model Revision`.
- `rendered_prompt` MUST be produced from reportable concept frames, not by
  directly exposing raw self-state slots.
- Superseded prompt revisions MUST remain inspectable through lineage metadata
  even when they are no longer current.

## Entity: DMN Internal Episode

One planner/tool episode of background cognition bound to a prompt revision.

### Fields

- `episode_id`
- `prompt_revision_id`
- `origin`
- `status`
- `started_at_us`
- `last_updated_us`
- `active_plan_id`
- `current_step_kind`
- `superseded_by_prompt_revision_id`
- `waiting_on_tool`
- `last_tool_result_kind`

### Invariants

- The episode MUST use the shared planner/tool runner semantics used by the
  active loop.
- The episode MUST remain distinct from active engagement accounting.
- A superseded episode MUST transition cleanly into a terminal or superseded
  status before a successor episode becomes current.

## Entity: DMN Telegram Relay Request

Represents a user-directed DMN tool action routed through the Telegram bridge.

### Fields

- `request_id`
- `episode_id`
- `prompt_revision_id`
- `intent_kind`
- `urgency`
- `routing_mode`
- `text`
- `dedupe_key`
- `policy_gate_mask`

### Invariants

- The request MUST be treated as a tool action, not as active engagement
  completion.
- `intent_kind` MUST distinguish at least question, comment, and conclusion
  modes.
- `text` MUST stay within explicit bounded delivery rules.

## Entity: DMN Telegram Relay Result

Represents the observed result of a DMN Telegram relay attempt.

### Fields

- `request_id`
- `delivered`
- `delivery_at_us`
- `bridge_message_id`
- `failure_kind`
- `failure_detail`
- `retry_allowed`

### Invariants

- The result MUST be integrated through the same tool-observation path used for
  other tools.
- Failure results MUST remain available to later DMN prompt revisions and
  episodes.
