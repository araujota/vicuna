# Research: Authoritative ReAct Action Contract Guarantee

## Local Runtime Findings

- `tools/server/server-context.cpp:4491` currently prefills hidden reasoning with a staged `Action:` line, but it does not own the visible staged JSON action bytes. The model still generates the full visible object, including fixed fields such as `"action":"select_tool"`.
- `tools/server/server-context.cpp:4274` builds a grammar that requires the `action` field for staged JSON selectors, but the observed production trace still emitted `{"tool_family_id":"web_search"}`. That means prompt wording plus schema alone are not sufficient as an operational guarantee.
- `tools/server/server-context.cpp:5460` allows missing hidden action labels to fall back through visible-surface inference. For ordinary prose this is intentional, but it also creates a leak path when the visible payload is control-shaped JSON instead of user prose.
- `tools/server/server-context.cpp:11244` and `tools/server/server-context.cpp:11385` already retry parse failures internally before surfacing an error, so the safest hardening path is to keep malformed staged artifacts inside this retry loop instead of letting them be misclassified as terminal assistant text.
- `tools/server/server-openclaw-fabric.cpp:2128` already rejects staged selector JSON that omits `action="select_tool"`. The missing piece is making that action token effectively unavoidable at generation time and preventing non-staged fallback from accepting control-shaped JSON as prose.

## GitHub History Findings

- GitHub commit [`f09532551693e098e8e76170d1ff35694f7d8fc2`](https://github.com/araujota/vicuna/commit/f09532551693e098e8e76170d1ff35694f7d8fc2) introduced earlier authoritative ReAct emission hardening, showing this problem has already required CPU-side policy rather than prompt-only nudges.
- GitHub commit [`8fd0e256b1b1cb2e8956684953ce23cc1941e00f`](https://github.com/araujota/vicuna/commit/8fd0e256b1b1cb2e8956684953ce23cc1941e00f) added malformed tool-call recovery. That history supports keeping malformed control artifacts inside explicit recovery and retry policy instead of surfacing them.
- GitHub commit [`0f545c3fecdec2c37014ed5c89861e61f77984f5`](https://github.com/araujota/vicuna/commit/0f545c3fecdec2c37014ed5c89861e61f77984f5) added visible-tail action inference. That explains the current failure mode: once a malformed staged payload escapes the strict staged parser, the generic visible-tail fallback can wrongly interpret it as a terminal reply.
- GitHub commit [`e74a01864e9a18d8a7f95177a81c0f68df578148`](https://github.com/araujota/vicuna/commit/e74a01864e9a18d8a7f95177a81c0f68df578148) continued authoritative ReAct until grounded, reinforcing that retries should stay in the same turn rather than degrade into leaked intermediate artifacts.

## Design Implications

- The strongest low-risk guarantee in this architecture is to make the exact staged action contract runtime-owned: if the model omits the fixed `action` field but the rest of the staged payload matches the current phase, runtime parsing can normalize the exact required action instead of trusting the model to retype it.
- The staged parser should continue to reject malformed control strictly, but repeated failures should remain in the existing retry-and-rewind machinery.
- Non-staged visible-tail fallback must explicitly reject control-shaped JSON so malformed controller artifacts cannot be emitted as user-visible prose.
