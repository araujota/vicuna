# Feature Specification: Radarr and Sonarr OpenClaw Tools

**Feature Branch**: `[070-truth-runtime-refactor]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "research the radarr and sonarr API shapes and create tools for Radarr and Sonarr in the openclaw surface that are visible to the agent. ensure the same descriptors for each tool and parameter of said tool's calls. these tools should point at the radarr/sonarr containers on the LAN-connected devices. treat this as a new spec."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The Agent Can Inspect Radarr and Sonarr (Priority: P1)

As the operator, I need the authoritative ReAct loop to see Radarr and Sonarr as first-class OpenClaw tools, so it can inspect media-server state instead of guessing.

**Why this priority**: Visibility into the media stack is the minimum useful outcome and is prerequisite to any write action.

**Independent Test**: Rebuild the host runtime and verify the active OpenClaw capability set includes `radarr` and `sonarr`, each with complete parameter descriptions and XML guidance.

**Acceptance Scenarios**:

1. **Given** the OpenClaw fabric is initialized, **When** the runtime builds the active or DMN tool surface, **Then** `radarr` and `sonarr` MUST be part of the same authoritative capability registry as the existing tools.
2. **Given** the ReAct loop receives tool metadata, **When** it inspects `radarr` or `sonarr`, **Then** the tool and each exposed parameter MUST have a non-empty description.
3. **Given** the tools are invoked for read actions, **When** the wrapper calls the LAN media services, **Then** the returned payload MUST preserve typed upstream results rather than flattening them into free text.

### User Story 2 - The Agent Can Search and Add Media Through the Same Tool Surface (Priority: P1)

As the operator, I need the agent to be able to look up and add movies or series through Radarr and Sonarr, so it can manage the media stack without leaving the ReAct tool path.

**Why this priority**: The user explicitly wants Radarr and Sonarr to be operational tools, not just status readers.

**Independent Test**: Invoke `radarr` and `sonarr` through the harness for lookup and add-style payloads, confirm request validation, upstream HTTP shaping, and clear typed error handling when auth is missing or upstream rejects the request.

**Acceptance Scenarios**:

1. **Given** a `lookup_movie` or `lookup_series` action with a search term, **When** the wrapper calls the official `/api/v3/.../lookup` endpoint, **Then** it MUST return the ranked upstream candidates.
2. **Given** an `add_movie` or `add_series` action with required folder/profile inputs, **When** the wrapper posts to the official `/api/v3/movie` or `/api/v3/series` endpoint, **Then** the wrapper MUST send a validated upstream-compatible resource body.
3. **Given** auth is missing or invalid, **When** the tool is invoked, **Then** the result MUST fail explicitly with a typed authorization/configuration error rather than silently disappearing from the tool surface.

### User Story 3 - Host Deployment Keeps One Tool System Only (Priority: P2)

As the operator, I need Radarr and Sonarr to deploy through the same OpenClaw runtime catalog and host rebuild flow as the other external tools, so the repo does not drift into multiple competing tool systems.

**Why this priority**: The runtime already has one authoritative tool fabric and this change must not undermine that.

**Independent Test**: Run the harness runtime-catalog sync, rebuild the host, and confirm the live startup logs advertise the new capabilities in the same OpenClaw surface.

**Acceptance Scenarios**:

1. **Given** the runtime catalog is written, **When** external capabilities are loaded, **Then** `radarr` and `sonarr` MUST enter through the same `openclaw_tool_capability_catalog` path as `web_search`.
2. **Given** the host rebuild runs, **When** the runtime restarts, **Then** the live capability count and capability log MUST include the new tools.
3. **Given** future tool additions are validated, **When** descriptor validation runs, **Then** the new tools MUST satisfy the same parameter-description contract as all other OpenClaw capabilities.

### Edge Cases

- What happens when the NAS base URL is reachable but the API key is missing or invalid?
- What happens when the lookup action returns zero candidates or multiple ambiguous candidates?
- What happens when an add action omits the required folder/profile inputs?
- What happens when the LAN media service is down, times out, or returns malformed JSON?
- What happens when the host is not on the same LAN and the default NAS base URLs are unreachable?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST expose a `radarr` tool and a `sonarr` tool through the same OpenClaw fabric used by the existing ReAct tool surface.
- **FR-002**: Both tools MUST be described through `openclaw_tool_capability_descriptor` contracts with non-empty top-level descriptions and non-empty descriptions for every exposed parameter.
- **FR-003**: Both tools MUST be emitted by the external OpenClaw runtime catalog rather than a second server-local registry.
- **FR-004**: The Radarr tool MUST support at least these actions: `system_status`, `queue`, `calendar`, `root_folders`, `quality_profiles`, `list_movies`, `lookup_movie`, and `add_movie`.
- **FR-005**: The Sonarr tool MUST support at least these actions: `system_status`, `queue`, `calendar`, `root_folders`, `quality_profiles`, `list_series`, `lookup_series`, and `add_series`.
- **FR-006**: The wrappers MUST target the LAN-connected Radarr and Sonarr instances by default at `http://10.0.0.218:7878` and `http://10.0.0.218:8989`, while allowing explicit override through OpenClaw tool secrets.
- **FR-007**: The wrappers MUST authenticate using the official `X-Api-Key` mechanism documented by the upstream APIs.
- **FR-008**: Missing or invalid auth MUST produce explicit typed tool failures rather than removing the tools from visibility once they are configured to be part of the runtime catalog.
- **FR-009**: `add_movie` and `add_series` MUST validate required action inputs before making upstream write requests and MUST construct upstream-compatible request bodies from lookup results plus explicit user/runtime inputs.
- **FR-010**: The server dispatch path MUST preserve the same authoritative ReAct tool-call flow, including XML guidance, parsing, bash-backed wrapper execution, and typed tool observation admission.
- **FR-011**: Tests and developer/operator docs MUST be updated for the new tools, their parameters, their secrets model, and the host rebuild/verification flow.

## Key Entities *(include if feature involves data)*

- **OpenClaw Servarr Service Config**: Runtime tool secret/config payload containing service base URL and API key for a LAN media service.
- **Radarr Action Request**: The structured Radarr tool invocation, including an `action` selector and action-specific inputs for lookup, listing, queue/calendar reads, or add-movie writes.
- **Sonarr Action Request**: The structured Sonarr tool invocation, including an `action` selector and action-specific inputs for lookup, listing, queue/calendar reads, or add-series writes.
- **Servarr Tool Observation**: The typed wrapper result containing service name, action, upstream request metadata, and sanitized JSON payload or typed error details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After rebuild, the live host startup log advertises both `radarr` and `sonarr` in the OpenClaw capability set.
- **SC-002**: The OpenClaw descriptor validators pass for both tools and fail if any exposed parameter description is removed.
- **SC-003**: The harness can successfully shape upstream read requests for both tools against the official `/api/v3` endpoints.
- **SC-004**: The wrappers return explicit typed configuration/auth errors when API keys are absent or rejected instead of silently hiding the tool surface.
