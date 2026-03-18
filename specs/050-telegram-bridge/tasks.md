# Tasks: Telegram Bridge Middleware

**Input**: Design documents from `/specs/050-telegram-bridge/`

**Tests**: Required. This task must verify bridge parser behavior and live
startup against the local Vicuña server.

## Phase 1: Bridge Contract

- [ ] T001 Confirm the Vicuña endpoints for direct replies and proactive
  self-emission streaming
- [ ] T002 Define bridge state, polling, and relay behavior

## Phase 2: Implementation

- [ ] T003 Implement the Telegram long-polling bridge service
- [ ] T004 Implement proactive self-emission SSE subscription and Telegram relay
- [ ] T005 Add runtime env defaults and operator documentation

## Phase 3: Validation

- [ ] T006 Add helper-level tests for stream parsing and response text
  extraction
- [ ] T007 Run bridge tests locally
- [ ] T008 Start the bridge against the live GPU-enabled Vicuña server
