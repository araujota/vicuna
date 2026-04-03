# Vicuña Architecture

## Retained Product Surface

Vicuña is now the host-side control plane for two inference paths:

- `standard`
  - direct DeepSeek API requests
  - large output budget
  - no request-level policy learning or runtime-control logic in the standard
    path
- `experimental`
  - host-owned request assembly, tool surface, Telegram delivery, capture, and
    rendering
  - relay to the RunPod pod through a stable host connector
  - host-side storage of transitions, decode traces, emotive traces, and video
    renders for offline learning

## Retained Components

- `tools/server`
  - HTTP API
  - mode router
  - DeepSeek client
  - RunPod connector
  - experimental capture persistence
  - request-level and decode-level observation schemas for RL data
- `tools/telegram-bridge`
  - Telegram polling and outbox delivery
  - WebGL render submission and delayed video delivery
- `tools/openclaw-harness`
  - host-owned tool access for hard memory, skills, host shell, and Telegram
    relay support
- `tools/policy-learning`
  - dataset export
  - PPO / GRU / cvec training infrastructure
  - registry and rollout metadata tooling
- `tools/ops`
  - systemd install and launch scripts
  - RunPod pod management and relay support
  - experimental capture sync to the host
  - host-side MP4 rendering from synced emotive traces

## Explicitly Removed From This Repository

- local `llama.cpp` runtime code
- upstream `llama.cpp` source and support trees
- media-management tooling and related wrappers
- speculative future runtime-fork assets that are not needed by the host plane

## Experimental Data Flow

1. The host receives a request.
2. The host routes to `standard` or `experimental`.
3. In `experimental`, the host forwards a plain chat/tool request to the pod.
4. The pod executes inference and returns content plus telemetry artifacts.
5. The host persists:
   - `transitions.jsonl`
   - `decode_traces.jsonl`
   - `emotive_traces.jsonl`
6. The host renderer turns synced emotive traces into MP4 video artifacts.
7. Offline RL jobs train against the host-owned datasets, not inside the pod.

## Boundary Rules

- Standard-mode DeepSeek requests remain detached from experimental learning.
- The pod owns experimental inference behavior.
- The host owns routing, delivery, durable capture, tool access, and training
  orchestration.
