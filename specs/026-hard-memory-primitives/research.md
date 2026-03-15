# Research: Hard Memory Primitives

## Local Codebase Findings

### Current hard-memory surface

- `src/llama-hard-memory.cpp` currently implements:
  - `query()` against Supermemory `/v4/profile`
  - `archive_event()` against Supermemory `/v4/memories`
- `archive_event()` emits one generic text record built from:
  - event role/channel/flags
  - total and max self-state delta
  - changed register list
  - raw message text
- `src/llama-self-state.cpp` archives only from `self_state_apply_postwrite()`
  when a self-state delta crosses `archival_delta_threshold`.
- `src/llama-self-state.cpp` already promotes hard-memory query hits into
  bounded self-model extensions, but query hits currently expose only:
  - id
  - title
  - content
  - similarity

### Runtime artifacts already available and worth archiving

- Self-state deltas:
  - register deltas
  - forecast and prediction error
  - self-model horizon summaries
  - self-model extension summary
- Active-loop settlement artifacts from `src/llama-cognitive-loop.cpp`:
  - chosen action
  - tool proposal and observation
  - functional activation decision
  - pre/post favorable snapshots and deltas
- DMN settlement artifacts:
  - counterfactual trace
  - remediation plan
  - governance trace
  - temporal self-improvement trace
- Tool-state artifacts:
  - bash request intent/command
  - bash result stdout/stderr/exit behavior
- User-model and social artifacts:
  - trust, reciprocity, dissatisfaction, recent user valence
  - user-outcome profile fields like misunderstanding risk and autonomy
    tolerance

### Architectural conclusion from local code

The codebase is already generating the right semantic artifacts, but the
hard-memory layer collapses them into one undifferentiated archival string. The
best improvement is to create a typed primitive layer and archive bounded
multi-record batches from the existing runtime traces.

## GitHub Research

### Letta

- `letta-ai/letta/letta/schemas/block.py`
- `letta-ai/letta/letta/schemas/memory.py`
- `letta-ai/letta/letta/schemas/memory_repo.py`

Relevant takeaways:

- Memory should be typed and bounded, not just appended text.
- There is clear separation between core memory, archival memory, and git-backed
  change history.
- Rich metadata and explicit labels matter for later retrieval and migration.

Implication for Vicuña:

- Hard memory should store primitive kind, domain, and provenance explicitly.
- Some Vicuña memory should look more like durable “blocks” or “commits” than
  anonymous blobs.

### LangGraph

- `langchain-ai/langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`

Relevant takeaways:

- Tool state and persistent store should be injected by the runtime, not left as
  free-form model text.
- Tool contracts should distinguish model-controlled args from system-injected
  state/store/runtime.

Implication for Vicuña:

- Future tools should archive memory primitives through explicit host/runtime
  APIs, not by stuffing structured meaning into stdout alone.

### Voyager

- `MineDojo/Voyager/voyager/voyager.py`

Relevant takeaways:

- Task trajectories, critiques, and learned skills are durable artifacts.
- Memory becomes more useful when it stores outcome-bearing trajectories, not
  just observations.

Implication for Vicuña:

- Vicuña should archive loop trajectories and outcomes explicitly, since that is
  the closest local analogue to Voyager’s task and skill residue.

## Web / Paper Research

### Supermemory

- Supermemory docs expose memory creation and profile/search APIs under their
  `memories` and `profile` endpoints.
- Their public docs emphasize metadata and profile retrieval, which fits
  Vicuña’s need to store typed memory records without replacing Supermemory.

Implication:

- Vicuña should lean on metadata/tags for typed primitive storage and query
  reconstruction rather than inventing a separate storage backend.

### MemGPT

- MemGPT frames long-term memory as explicit archival memory that can be fetched
  into the active context when needed.

Implication:

- Vicuña’s hard-memory layer should not be a dead archive. It should feed back
  into current control through retrieval summaries and self-model promotion.

### Generative Agents

- The paper’s memory pipeline distinguishes raw observations, higher-level
  reflections, and planning-relevant retrieval.

Implication:

- Vicuña’s primitive vocabulary should distinguish:
  - event fragments / observations
  - trajectories / reflections
  - outcomes / validated residue

### Mem0 / current agent-memory practice

- Recent agent-memory work treats memory as a curated layer with selective
  extraction of facts, preferences, and durable context.

Implication:

- Vicuña should explicitly archive user-model fragments like preference
  uncertainty, trust, dissatisfaction, and autonomy tolerance.

### Biologically inspired thesis

- The closest biologically inspired decomposition is not “one memory type,” but
  a separation between episodic, semantic, and procedural residue.

Implication:

- A strong first Vicuña primitive set is:
  - episodic: `EVENT_FRAGMENT`, `TRAJECTORY`
  - semantic: `USER_MODEL`, `SELF_MODEL_FRAGMENT`
  - procedural/outcome-bearing: `OUTCOME`, `TOOL_OBSERVATION`

## Recommended Architecture

1. Add a typed hard-memory primitive vocabulary and bounded archive batch type.
2. Upgrade event archival to emit multiple primitives when signal is present.
3. Add cognitive-loop archival for settled active/DMN trajectories and outcomes.
4. Parse typed metadata back from query hits.
5. Derive a fixed-width retrieval summary and feed it into self-model promotion
   and functional-gating observation.
6. Keep all policy CPU-side, bounded, and inspectable.
