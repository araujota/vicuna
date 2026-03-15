# Contract: Self-State API

## Context Ownership

- Each `llama_context` owns exactly one self-state container.
- The self-state container exists for the lifetime of the context.
- Self-state policy remains in CPU-side control code.

## Register Inspection

- The runtime MUST expose a fixed predefined register set for the initial implementation slice.
- Register lookup by id MUST fail explicitly for unknown ids.
- Register inspection MUST return:
  - register id
  - register family
  - scalar and/or categorical value fields
  - confidence
  - last-update wall-clock and monotonic timestamps
  - source mask
  - updater version
  - dirty flag

## Time Contract

- The runtime MUST accept explicit caller-supplied time points.
- Explicit time points MUST update the datetime surface deterministically.
- The runtime MUST reject non-monotonic time updates.
- The runtime MUST provide a convenience API to refresh from the local system clock.

## Event-Anchor Contract

- The runtime MUST track anchor times for:
  - last user event
  - last tool event
  - last emit event
- Elapsed delta fields MUST derive from monotonic time.

## Feature Builder Contract

- The runtime MUST expose explicit prewrite and postwrite feature-builder APIs.
- Feature builders MUST accept a typed token event object and output a typed feature vector.
- Feature builders MUST stay functional when decoder-stat metadata is absent.

## Learned Head Contract

- Contradiction and uncertainty heads MUST be independently configurable.
- Learned heads MUST be optional in the initial slice.
- If a learned head is disabled or lacks a callback, the runtime MUST fall back to the analytic path.
- Learned head outputs MUST be clamped to a bounded range before use.

## Initial Analytic Register Rules

- `r_time_phase` derives from local wall-clock phase.
- `r_tool_salience` derives from elapsed time since the last tool event and remains bounded.
- `r_channel_state` mirrors the explicit categorical channel state.

## Failure Contract

The API MUST fail explicitly for:
- null context
- null output pointers
- invalid register ids
- invalid channel states
- invalid or non-monotonic time inputs
