#!/usr/bin/env python3

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from policy_dataset import load_dataset_rows, load_manifest
from policy_runtime_client import PolicyRuntimeClient
from policy_training_contract import (
    BOOLEAN_HEADS,
    normalize_action_mask,
    normalize_action_targets,
)


OFFLINE_EVAL_SCHEMA_VERSION = "vicuna.offline_policy_evaluation.v1"


def _evaluate_with_command(
    command: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    result = subprocess.run(
        shlex.split(command),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"candidate command exited {result.returncode}")
    return json.loads(result.stdout)


def _evaluate_with_http(
    candidate_url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    client = PolicyRuntimeClient(base_url=candidate_url, timeout_s=timeout_s)
    return client.propose_candidate(candidate_url, payload)


def _compare_targets(candidate_targets: dict[str, Any], executed_targets: dict[str, Any]) -> dict[str, bool]:
    comparison = {
        "selected_mode": candidate_targets["selected_mode"] == executed_targets["selected_mode"],
        "reasoning_depth": candidate_targets["reasoning_depth"]
        == executed_targets["reasoning_depth"],
        "thinking_mode": candidate_targets["thinking_mode"]
        == executed_targets["thinking_mode"],
        "prefix_profile": candidate_targets["prefix_profile"]
        == executed_targets["prefix_profile"],
        "stop_profile": candidate_targets["stop_profile"]
        == executed_targets["stop_profile"],
        "sampling_profile": candidate_targets["sampling_profile"]
        == executed_targets["sampling_profile"],
        "repetition_profile": candidate_targets["repetition_profile"]
        == executed_targets["repetition_profile"],
        "tool_choice_profile": candidate_targets["tool_choice_profile"]
        == executed_targets["tool_choice_profile"],
        "response_budget_bucket": candidate_targets["response_budget_bucket"]
        == executed_targets["response_budget_bucket"],
        "reasoning_budget_bucket": candidate_targets["reasoning_budget_bucket"]
        == executed_targets["reasoning_budget_bucket"],
        "token_budget_bucket": candidate_targets["token_budget_bucket"]
        == executed_targets["token_budget_bucket"],
        "tool_parallelism_cap": candidate_targets["tool_parallelism_cap"]
        == executed_targets["tool_parallelism_cap"],
    }
    for head in BOOLEAN_HEADS:
        comparison[head] = candidate_targets[head] == executed_targets[head]
    return comparison


def evaluate_dataset(
    dataset_dir: Path,
    *,
    candidate_command: str | None = None,
    candidate_url: str | None = None,
    limit: int | None = None,
    timeout_s: float = 5.0,
    report_path: Path | None = None,
) -> dict[str, Any]:
    if not candidate_command and not candidate_url:
        raise ValueError("candidate_command or candidate_url is required")
    if candidate_command and candidate_url:
        raise ValueError("candidate_command and candidate_url are mutually exclusive")

    manifest = load_manifest(dataset_dir)
    if manifest is None:
        raise FileNotFoundError(f"dataset manifest not found in {dataset_dir}")
    rows = load_dataset_rows(dataset_dir)
    if limit is not None:
        rows = rows[:limit]

    evaluated_records = 0
    successful_predictions = 0
    candidate_failure_count = 0
    invalid_action_count = 0
    exact_match_count = 0
    head_match_counts = {
        "selected_mode": 0,
        "reasoning_depth": 0,
        "thinking_mode": 0,
        "prefix_profile": 0,
        "stop_profile": 0,
        "sampling_profile": 0,
        "repetition_profile": 0,
        "tool_choice_profile": 0,
        "response_budget_bucket": 0,
        "reasoning_budget_bucket": 0,
        "token_budget_bucket": 0,
        "tool_parallelism_cap": 0,
        **{head: 0 for head in BOOLEAN_HEADS},
    }
    reward_total_sum = 0.0
    reward_total_match_sum = 0.0
    reward_total_mismatch_sum = 0.0
    reward_total_match_count = 0
    reward_total_mismatch_count = 0
    candidate_policy_versions: set[str] = set()
    per_head_disagreement = {
        "selected_mode": 0,
        "reasoning_depth": 0,
        "thinking_mode": 0,
        "prefix_profile": 0,
        "stop_profile": 0,
        "sampling_profile": 0,
        "repetition_profile": 0,
        "tool_choice_profile": 0,
        "response_budget_bucket": 0,
        "reasoning_budget_bucket": 0,
        "token_budget_bucket": 0,
        "tool_parallelism_cap": 0,
        **{head: 0 for head in BOOLEAN_HEADS},
    }

    for row in rows:
        transition = row["transition"]
        normalized_mask = normalize_action_mask(transition["action_mask"])
        executed_targets = normalize_action_targets(
            transition["executed_action"], normalized_mask
        )
        reward_total = float(transition.get("reward_total", 0.0))
        reward_total_sum += reward_total
        evaluated_records += 1

        request_payload = {
            "policy_mode": "offline_eval",
            "dataset_id": manifest["dataset_id"],
            "observation": transition["observation"],
            "action_mask": transition["action_mask"],
            "executed_action": transition["executed_action"],
        }
        try:
            candidate_payload = (
                _evaluate_with_command(candidate_command, request_payload, timeout_s)
                if candidate_command
                else _evaluate_with_http(candidate_url, request_payload, timeout_s)
            )
            candidate_policy_version = candidate_payload.get("policy_version")
            if candidate_policy_version:
                candidate_policy_versions.add(str(candidate_policy_version))
        except Exception:
            candidate_failure_count += 1
            continue

        try:
            candidate_targets = normalize_action_targets(
                candidate_payload["action"], normalized_mask
            )
        except Exception:
            invalid_action_count += 1
            continue

        successful_predictions += 1
        comparison = _compare_targets(candidate_targets, executed_targets)

        exact_match = all(comparison.values())
        if exact_match:
            exact_match_count += 1
            reward_total_match_sum += reward_total
            reward_total_match_count += 1
        else:
            reward_total_mismatch_sum += reward_total
            reward_total_mismatch_count += 1

        for head, matched in comparison.items():
            if matched:
                head_match_counts[head] += 1
            else:
                per_head_disagreement[head] += 1

    def safe_rate(count: int) -> float:
        return 0.0 if evaluated_records == 0 else count / evaluated_records

    def safe_mean(total: float, count: int) -> float | None:
        if count == 0:
            return None
        return total / count

    report = {
        "schema_version": OFFLINE_EVAL_SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "reward_model_version": manifest.get("source_reward_model_version"),
        "candidate_adapter": "command" if candidate_command else "http",
        "candidate_label": candidate_command or candidate_url,
        "candidate_policy_version": sorted(candidate_policy_versions)[0]
        if len(candidate_policy_versions) == 1
        else None,
        "evaluated_records": evaluated_records,
        "successful_predictions": successful_predictions,
        "candidate_failure_count": candidate_failure_count,
        "invalid_action_count": invalid_action_count,
        "invalid_action_rate": safe_rate(invalid_action_count),
        "exact_match_rate": safe_rate(exact_match_count),
        "mode_match_rate": safe_rate(head_match_counts["selected_mode"]),
        "reasoning_depth_match_rate": safe_rate(head_match_counts["reasoning_depth"]),
        "thinking_mode_match_rate": safe_rate(head_match_counts["thinking_mode"]),
        "prefix_profile_match_rate": safe_rate(head_match_counts["prefix_profile"]),
        "stop_profile_match_rate": safe_rate(head_match_counts["stop_profile"]),
        "sampling_profile_match_rate": safe_rate(head_match_counts["sampling_profile"]),
        "repetition_profile_match_rate": safe_rate(head_match_counts["repetition_profile"]),
        "tool_choice_profile_match_rate": safe_rate(head_match_counts["tool_choice_profile"]),
        "response_budget_bucket_match_rate": safe_rate(head_match_counts["response_budget_bucket"]),
        "reasoning_budget_bucket_match_rate": safe_rate(head_match_counts["reasoning_budget_bucket"]),
        "reward_total_mean": safe_mean(reward_total_sum, evaluated_records),
        "reward_total_mean_on_match": safe_mean(
            reward_total_match_sum, reward_total_match_count
        ),
        "reward_total_mean_on_mismatch": safe_mean(
            reward_total_mismatch_sum, reward_total_mismatch_count
        ),
        "per_head_disagreement": per_head_disagreement,
        "generated_at_ms": int(time.time() * 1000),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["report_path"] = str(report_path)
    return report
