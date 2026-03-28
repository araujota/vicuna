#!/usr/bin/env python3

from __future__ import annotations

import json
import sys


def _choose_first(values: list[str], fallback: str) -> str:
    return values[0] if values else fallback


def main() -> int:
    payload = json.load(sys.stdin)
    observation = payload.get("observation", {})
    action_mask = payload.get("action_mask", {})
    allowed_modes = action_mask.get("allowed_modes", [])
    allowed_reasoning_depths = action_mask.get("allowed_reasoning_depths", [])
    allowed_thinking_modes = action_mask.get("allowed_thinking_modes", ["enabled", "disabled"])
    allowed_prefix_profiles = action_mask.get("allowed_prefix_profiles", ["none"])
    allowed_stop_profiles = action_mask.get("allowed_stop_profiles", ["none"])
    allowed_sampling_profiles = action_mask.get("allowed_sampling_profiles", ["provider_default"])
    allowed_repetition_profiles = action_mask.get("allowed_repetition_profiles", ["none"])
    allowed_tool_choice_profiles = action_mask.get("allowed_tool_choice_profiles", ["caller_default"])
    max_parallelism = int(action_mask.get("max_tool_parallelism_cap", 0))

    preferred_mode = "tool_light" if int(observation.get("available_tool_count", 0)) > 0 else "direct"
    if preferred_mode not in allowed_modes:
        preferred_mode = _choose_first(allowed_modes, "direct")

    preferred_reasoning = "medium" if observation.get("heuristic_matched") else "short"
    if preferred_reasoning not in allowed_reasoning_depths:
        preferred_reasoning = _choose_first(allowed_reasoning_depths, "short")

    token_budget_bucket = {
        "none": 256,
        "short": 512,
        "medium": 1024,
        "deep": 2048,
    }.get(preferred_reasoning, 512)

    preferred_thinking = "disabled" if preferred_reasoning in {"none", "short"} else "enabled"
    if preferred_thinking not in allowed_thinking_modes:
        preferred_thinking = _choose_first(allowed_thinking_modes, "enabled")

    preferred_prefix = "bounded_answer" if preferred_thinking == "disabled" else "none"
    if preferred_prefix not in allowed_prefix_profiles:
        preferred_prefix = _choose_first(allowed_prefix_profiles, "none")

    preferred_stop = "concise_answer" if preferred_prefix == "bounded_answer" else "none"
    if preferred_stop not in allowed_stop_profiles:
        preferred_stop = _choose_first(allowed_stop_profiles, "none")

    preferred_sampling = "deterministic" if preferred_thinking == "disabled" else "provider_default"
    if preferred_sampling not in allowed_sampling_profiles:
        preferred_sampling = _choose_first(allowed_sampling_profiles, "provider_default")

    preferred_repetition = "anti_stall_soft" if preferred_prefix == "bounded_answer" else "none"
    if preferred_repetition not in allowed_repetition_profiles:
        preferred_repetition = _choose_first(allowed_repetition_profiles, "none")

    preferred_tool_choice = "auto" if preferred_mode in {"tool_light", "tool_heavy"} else "caller_default"
    if preferred_tool_choice not in allowed_tool_choice_profiles:
        preferred_tool_choice = _choose_first(allowed_tool_choice_profiles, "caller_default")

    result = {
        "policy_version": "example_candidate_v1",
        "action": {
            "selected_mode": preferred_mode,
            "reasoning_depth": preferred_reasoning,
            "thinking_mode": preferred_thinking,
            "prefix_profile": preferred_prefix,
            "stop_profile": preferred_stop,
            "sampling_profile": preferred_sampling,
            "repetition_profile": preferred_repetition,
            "tool_choice_profile": preferred_tool_choice,
            "token_budget_bucket": token_budget_bucket,
            "tool_parallelism_cap": 1 if max_parallelism >= 1 else 0,
            "interrupt_allowed": bool(action_mask.get("allow_interrupt", True)),
            "replan_required": False,
            "early_stop_ok": bool(action_mask.get("allow_early_stop", True)),
            "force_synthesis": False,
        },
    }
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
