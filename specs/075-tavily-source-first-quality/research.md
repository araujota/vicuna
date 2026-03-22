# Research: Tavily Source-First Quality

## Host Trace Context

- The recent Telegram request that should have used `exec` to report the present working directory did not use any tool at all.
- Host runtime logs showed repeated `authoritative react parsed: action=1 ... tool_calls=0`, followed by continuation rejection for procedural non-answer text.
- This is a separate tool-selection/grounding issue, but it confirms that when tools do run, their outputs need to be high quality because the loop depends on them.

## Local Code Audit

- `tools/openclaw-harness/src/tavily-search.ts` currently always sends `include_answer: true`.
- The wrapper returns the provider-generated `answer` verbatim plus a bounded result list.
- The result truncation helper currently slices to 5 items regardless of the requested `max_results`.
- The runtime dispatch path in `tools/server/server-context.cpp` only passes `query`, `topic`, `search_depth`, and `max_results`.
- The runtime catalog in `tools/openclaw-harness/src/catalog.ts` only exposes those same limited fields.

## GitHub Research

### LangChain JS Tavily integration

Sources:
- [libs/community/langchain-tavily/src/tavily-search.ts](https://github.com/langchain-ai/langchainjs/blob/443253bd5336597d3619f2e21a8d5a814202997e/libs/community/langchain-tavily/src/tavily-search.ts)
- [libs/community/langchain-tavily/src/utils.ts](https://github.com/langchain-ai/langchainjs/blob/443253bd5336597d3619f2e21a8d5a814202997e/libs/community/langchain-tavily/src/utils.ts)

Relevant findings:
- `includeAnswer` defaults to `false`.
- The integration exposes `includeRawContent`, `includeDomains`, `excludeDomains`, `timeRange`, `topic`, `country`, `chunksPerSource`, and `autoParameters`.
- The docs/comments describe `answer` explicitly as an LLM-generated answer, not primary evidence.

## Web Research

### Tavily parameter guidance

Source: [Optimizing Your Request Parameters](https://help.tavily.com/articles/7879881576-optimizing-your-query-parameters)

Relevant guidance:
- `search_depth=advanced` is recommended for higher relevance and specific information needs.
- `chunks_per_source` and `include_raw_content` improve retrieval quality.
- `time_range` should be used for recent information.
- `include_domains` improves focus for domain-specific searches.

## Design Conclusions

- The current integration is too answer-centric. It should be source-centric.
- The strongest immediate fix is to stop requesting Tavily's generated `answer` by default and instead retrieve richer evidence from multiple sources.
- The tool surface should expose the filters Tavily itself recommends for better quality.
- The wrapper should maintain an inspectable quality floor on `max_results` rather than allowing thin single-source requests to dominate.
