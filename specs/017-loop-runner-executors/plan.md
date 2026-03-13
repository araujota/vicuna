# Implementation Plan: Planner-Executor Runners For Active And DMN

## Goal

Turn the existing loop scaffold into real bounded planner-executor runners with
host-visible pending commands and resumable state.

## Architecture Decision

Use an event-driven controller pattern:

1. score and select using the existing explicit policy
2. project that into runner state
3. enqueue host-visible commands
4. resume the runner when host observations arrive
5. allow DMN to take additional bounded internal steps before yielding

## Workstreams

### 1. Public API

- add runner command enums and structs
- add runner status structs
- add pending-command polling and acknowledgment APIs

### 2. Runtime State

- extend `llama_cognitive_loop` with active-runner state, DMN-runner state, and
  a bounded pending-command queue
- preserve current traces as snapshots of runner decisions

### 3. Active Runner

- create commands for answer, ask, and tool invocation
- persist waiting state across tool observations
- prevent unbounded continuation

### 4. DMN Runner

- execute internal local actions inline
- re-plan within a bounded step budget
- enqueue tool or emit commands when appropriate

### 5. Host Integration

- expose queue polling in the public C API
- update `llama-server` to log or acknowledge commands using the new surface

### 6. Verification

- extend cognitive-loop tests for command queue and resume semantics
- keep adjacent Active/Past LoRA parity tests green
