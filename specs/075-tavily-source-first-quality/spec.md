# Feature Specification: Tavily Source-First Quality

**Feature Branch**: `[070-truth-runtime-refactor]`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "check the logs of the last request where i asked for the present working directory; it should've used the exec tool. update our tavily tool with the best recommended architecture for good results"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Web Search Returns Evidence, Not a Synthetic Oracle (Priority: P1)

As a user, I need the `web_search` tool to return strong source evidence rather than a fragile generated answer, so downstream ReAct steps synthesize from citations and excerpts instead of trusting a provider-side hallucinated summary.

**Why this priority**: The current integration always requests Tavily's generated `answer`, which can amplify bad top hits and fabricate entities.

**Independent Test**: Invoke the Tavily wrapper with default settings and verify the request disables synthetic answer mode while returning bounded multi-source evidence with URLs, scores, and excerpts.

**Acceptance Scenarios**:

1. **Given** a normal factual search, **When** the Tavily wrapper builds the API request, **Then** it MUST disable provider-generated answer synthesis by default.
2. **Given** a normal factual search, **When** the Tavily wrapper returns results, **Then** it MUST return ranked sources with bounded excerpts and metadata suitable for model-side synthesis.
3. **Given** a query where one weak result would dominate, **When** the wrapper normalizes the request, **Then** it MUST avoid thin single-source retrieval by default.

---

### User Story 2 - The Tool Surface Exposes Better Search Controls (Priority: P1)

As a ReAct planner, I need the `web_search` tool to expose time, domain, and topical controls, so it can retrieve better evidence for live facts and specialized topics.

**Why this priority**: Tavily quality depends heavily on parameters such as `search_depth`, `time_range`, `include_domains`, and `topic`.

**Independent Test**: Inspect the runtime catalog and unit tests to verify the tool schema now exposes the relevant parameters and accepted enums.

**Acceptance Scenarios**:

1. **Given** the external runtime catalog, **When** `web_search` is advertised, **Then** it MUST expose `time_range`, `include_domains`, `exclude_domains`, and `country` in addition to `query`, `topic`, `search_depth`, and `max_results`.
2. **Given** a finance query, **When** the planner emits `topic=finance`, **Then** the wrapper MUST pass that topic through correctly.
3. **Given** advanced search mode, **When** the wrapper calls Tavily, **Then** it MUST request richer source content chunks instead of only the shallow default snippet.

---

### User Story 3 - Quality Defaults Are Explicit and Inspectable (Priority: P2)

As a maintainer, I need Tavily quality defaults to be explicit in code and docs, so result quality is governed by inspectable policy rather than hidden provider behavior.

**Why this priority**: This tool is part of the authoritative ReAct loop and directly affects factual reliability.

**Independent Test**: Inspect docs and tests to verify the source-first policy, default result count floor, and richer content retrieval settings.

**Acceptance Scenarios**:

1. **Given** the wrapper source, **When** operators inspect it, **Then** they MUST be able to see the explicit defaults for `max_results`, `search_depth`, `include_raw_content`, and chunk sizing.
2. **Given** docs and tests, **When** this feature ships, **Then** they MUST describe why provider-generated answer fields are not treated as authoritative.

### Edge Cases

- A query asks for live financial data and needs `topic=finance`.
- A query asks for recent/current information and needs `time_range`.
- The model requests `max_results=1`; the wrapper should still maintain a quality floor.
- Domain-restricted searches should still work with explicit `include_domains`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The Tavily wrapper MUST disable provider-generated answer synthesis by default.
- **FR-002**: The Tavily wrapper MUST request richer source evidence via advanced/raw-content-compatible parameters by default.
- **FR-003**: The Tavily wrapper MUST normalize `max_results` to a quality-oriented floor instead of honoring pathological single-source requests.
- **FR-004**: The runtime catalog schema for `web_search` MUST expose `topic`, `search_depth`, `max_results`, `time_range`, `include_domains`, `exclude_domains`, and `country`.
- **FR-005**: The runtime dispatch layer MUST pass the new supported search parameters through to the Tavily wrapper.
- **FR-006**: The wrapper result shape MUST remain bounded and inspectable while preserving URLs, titles, scores, published dates, and content excerpts.
- **FR-007**: Tests MUST cover request normalization and exposed schema changes.
- **FR-008**: Documentation MUST explain the source-first Tavily policy and the reason for avoiding provider-generated answer text as authority.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The Tavily request builder no longer sends `include_answer: true` by default.
- **SC-002**: The runtime catalog advertises the expanded search-control schema.
- **SC-003**: Regression tests prove the wrapper uses a multi-source quality floor and richer content retrieval settings.
- **SC-004**: Operator documentation explicitly states that Tavily `answer` is not treated as authoritative evidence.
