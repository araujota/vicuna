# Feature Specification: OpenClaw Tool Descriptor Unification

**Feature Branch**: `[070-truth-runtime-refactor]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "all tools should expose parameter descriptions, and should be part of the same openclaw fabric. confirm/ensure this. we cannot have competing tool systems"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Every OpenClaw Tool Is Self-Describing (Priority: P1)

As a maintainer, I need every tool exposed to the authoritative ReAct loop to carry parameter descriptions, so the model sees a complete semantic contract instead of bare types.

**Why this priority**: Tool choice quality depends on parameter semantics, not just tool names and top-level descriptions.

**Independent Test**: Validate that builtin and external OpenClaw capabilities are rejected if any exposed parameter lacks a description.

**Acceptance Scenarios**:

1. **Given** an OpenClaw capability with an object schema, **When** the fabric validates it, **Then** each exposed parameter MUST carry a non-empty `description`.
2. **Given** nested object or array item properties, **When** they are exposed as part of a tool schema, **Then** their parameter surfaces MUST also carry non-empty descriptions.
3. **Given** the active or DMN ReAct loop receives tool metadata, **When** a tool is present, **Then** its parameter descriptions MUST already be part of the same exposed schema.

### User Story 2 - ReAct Sees One Tool Fabric Only (Priority: P1)

As a maintainer, I need all live tools for active and DMN ReAct to flow through the same OpenClaw fabric, so there are no competing descriptor systems or hidden alternate tool registries.

**Why this priority**: Duplicate tool-definition paths invite drift and conflicting selection behavior.

**Independent Test**: Inspect the runtime wiring and verify that chat tools and cognitive specs are sourced from `server_openclaw_fabric` only.

**Acceptance Scenarios**:

1. **Given** any tool available to active or DMN ReAct, **When** it is surfaced to the model, **Then** it MUST come from `server_openclaw_fabric`.
2. **Given** builtin and external tools, **When** the fabric builds the ReAct tool surface, **Then** they MUST share the same capability contract type and validation path.
3. **Given** a capability definition that bypasses the OpenClaw validation path, **When** it would otherwise reach the model, **Then** the implementation MUST be corrected so the fabric remains the only authoritative tool surface.

### User Story 3 - Docs and Tests Stay Aligned (Priority: P2)

As a maintainer, I need the descriptor rules documented and tested, so future tool additions cannot silently regress back into underspecified or split tool surfaces.

**Why this priority**: This repo depends on explicit, inspectable policy and typed contracts.

**Independent Test**: Run targeted native and harness tests that fail on missing parameter descriptions and verify the docs describe the single-fabric rule.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `openclaw_tool_capability_descriptor_validate` MUST reject tool schemas whose exposed parameters lack non-empty descriptions.
- **FR-002**: The TypeScript OpenClaw catalog contract MUST reject the same class of underspecified schemas.
- **FR-003**: All builtin tools surfaced by `server_openclaw_fabric` MUST include parameter descriptions for all exposed parameters.
- **FR-004**: All external tools emitted through the OpenClaw harness runtime catalog MUST include parameter descriptions for all exposed parameters.
- **FR-005**: Active and DMN ReAct tool surfaces MUST continue to be built from `server_openclaw_fabric` only.
- **FR-006**: Tests and documentation MUST describe and enforce the single-fabric, fully-described-schema rule.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Capability validation fails when any OpenClaw tool schema omits parameter descriptions.
- **SC-002**: All shipped builtin and external OpenClaw tools pass the strengthened validation.
- **SC-003**: The documented ReAct tool path shows one authoritative fabric source for active and DMN tools.
