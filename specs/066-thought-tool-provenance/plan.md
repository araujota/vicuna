# Implementation Plan: Full Thought and Tool Provenance Logging

**Branch**: `066-thought-tool-provenance (on current 060-service-user-migration worktree)` | **Date**: 2026-03-20 | **Spec**: [/Users/tyleraraujo/vicuna/specs/066-thought-tool-provenance/spec.md](/Users/tyleraraujo/vicuna/specs/066-thought-tool-provenance/spec.md)

## Summary

Expand the unified provenance repository so it records the complete active and
DMN narration surfaces already present in the runtime, plus exact structured
tool request payloads at dispatch time. Keep journal output concise while making
the append-only JSONL repository the canonical inspection path.

## Technical Context

- Language: C++17 for runtime/server, Python for server tests
- Primary files:
  - `tools/server/server-context.cpp`
  - `tests/test-cognitive-loop.cpp`
  - `tools/server/tests/unit/test_basic.py`
  - `tools/server/README-dev.md`
- Constraints:
  - preserve existing `tool_result` provenance
  - avoid duplicating logging subsystems
  - keep runtime policy explicit and inspectable

## Design

1. Extend provenance JSON helpers:
   - serialize plan traces with step arrays
   - serialize active and DMN candidate arrays
   - attach active planner narration, visible output, and raw tool XML when available
2. Add structured request serializers for:
   - bash
   - hard memory
   - Codex
   - Telegram relay
3. Append a new `tool_call` provenance event at server dispatch time, before
   execution begins.
4. Update documentation and tests to validate the richer provenance stream.

## Validation

- Native cognitive-loop tests for planner narration note persistence
- Server unit tests for provenance file contents
- Targeted build/test run for `llama-server`, `test-cognitive-loop`, and server
  unit coverage
