#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from policy_dataset import append_decode_traces, append_transitions
from policy_evaluator import evaluate_dataset
from artifact_live_rollout import ArtifactLiveRolloutConfig, advance_artifact_rollout
from policy_live_rollout import LiveRolloutConfig, advance_rollout
from policy_registry import (
    BATCH_RUN_SCHEMA_VERSION,
    get_alias_version,
    promote_alias,
    register_artifact,
    registry_status,
    resolve_artifact_path,
    write_batch_run,
)
from policy_registry_server import run_server as run_registry_server
from policy_runtime_client import PolicyRuntimeClient
from policy_training_contract import build_training_corpus


def emit_log(service: str, event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "schema_version": "vicuna.service_event.v1",
                "timestamp_ms": int(time.time() * 1000),
                "service": service,
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def export_dataset(args: argparse.Namespace) -> dict:
    client = PolicyRuntimeClient(args.server, timeout_s=args.timeout_ms / 1000.0)
    summary = {}
    export_mode = "follow" if args.follow else "snapshot"
    polls = 0
    while True:
        status = client.get_status()
        transitions_payload = client.get_transitions(limit=args.limit)
        transition_summary = append_transitions(
            dataset_dir=Path(args.dataset_dir),
            dataset_id=args.dataset_id,
            source_base_url=args.server,
            export_mode=export_mode,
            status_payload=status,
            transitions=transitions_payload.get("items", []),
        )
        decode_payload = client.get_decode_traces(limit=getattr(args, "decode_limit", 128))
        decode_summary = append_decode_traces(
            dataset_dir=Path(args.dataset_dir),
            dataset_id=args.dataset_id,
            source_base_url=args.server,
            export_mode=export_mode,
            status_payload=status,
            decode_traces=decode_payload.get("items", []),
        )
        summary = {
            **transition_summary,
            "decode_appended_count": decode_summary["appended_count"],
            "stored_decode_trace_count": decode_summary["stored_decode_trace_count"],
            "decode_traces_path": decode_summary["decode_traces_path"],
        }
        polls += 1
        if not args.follow:
            break
        if args.max_polls is not None and polls >= args.max_polls:
            break
        time.sleep(args.poll_interval_s)
    summary["polls"] = polls
    summary["mode"] = export_mode
    return summary


def evaluate_candidate(args: argparse.Namespace) -> dict:
    return evaluate_dataset(
        dataset_dir=Path(args.dataset_dir),
        candidate_command=args.candidate_command,
        candidate_url=args.candidate_url,
        limit=args.limit,
        timeout_s=args.timeout_ms / 1000.0,
        report_path=Path(args.report_path) if getattr(args, "report_path", None) else None,
    )


def evaluate_cvec_candidate(args: argparse.Namespace) -> dict:
    from cvec_evaluator import evaluate_cvec_generator

    return evaluate_cvec_generator(
        dataset_dir=Path(args.dataset_dir),
        artifact_path=Path(args.artifact_path),
        report_path=Path(args.report_path) if getattr(args, "report_path", None) else None,
    )


def train_candidate(args: argparse.Namespace) -> dict:
    from ppo_trainer import train_ppo_policy

    return train_ppo_policy(
        dataset_dir=Path(args.dataset_dir),
        model_name=args.model_name,
        hidden_dims=getattr(args, "hidden_dims", None),
        warmstart_epochs=getattr(args, "warmstart_epochs", 250),
        ppo_epochs=getattr(args, "ppo_epochs", 200),
        learning_rate=getattr(args, "learning_rate", 0.01),
        clip_coef=getattr(args, "clip_coef", 0.2),
        ent_coef=getattr(args, "ent_coef", 0.01),
        vf_coef=getattr(args, "vf_coef", 0.5),
        run_root=Path(args.run_root) if getattr(args, "run_root", None) else None,
    )


def build_cvec_training_set(args: argparse.Namespace) -> dict:
    from cvec_generator_contract import build_cvec_training_corpus

    return build_cvec_training_corpus(
        dataset_dir=Path(args.dataset_dir),
        target_embedding_dim=args.target_embedding_dim,
        vector_library_path=Path(args.vector_library_path) if getattr(args, "vector_library_path", None) else None,
    )


def build_ppo_training_set(args: argparse.Namespace) -> dict:
    from ppo_training_contract import build_ppo_training_corpus

    return build_ppo_training_corpus(Path(args.dataset_dir))


def build_decode_training_set(args: argparse.Namespace) -> dict:
    from decode_gru_contract import build_decode_gru_training_corpus

    return build_decode_gru_training_corpus(
        Path(args.dataset_dir),
        sequence_length=args.sequence_length,
        stride=args.stride,
    )


def train_cvec_candidate(args: argparse.Namespace) -> dict:
    from cvec_trainer import train_cvec_generator

    return train_cvec_generator(
        dataset_dir=Path(args.dataset_dir),
        model_name=args.model_name,
        target_embedding_dim=args.target_embedding_dim,
        target_layer_start=args.target_layer_start,
        target_layer_end=args.target_layer_end,
        vector_library_path=Path(args.vector_library_path) if getattr(args, "vector_library_path", None) else None,
        hidden_dims=args.hidden_dims,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        reward_weight_power=args.reward_weight_power,
        output_norm_cap=args.output_norm_cap,
        output_mode=args.output_mode,
        activation=args.activation,
        run_root=Path(args.run_root) if getattr(args, "run_root", None) else None,
    )


def evaluate_ppo_candidate(args: argparse.Namespace) -> dict:
    from ppo_evaluator import evaluate_ppo_policy

    return evaluate_ppo_policy(
        dataset_dir=Path(args.dataset_dir),
        artifact_path=Path(args.artifact_path),
        report_path=Path(args.report_path) if getattr(args, "report_path", None) else None,
    )


def train_decode_candidate(args: argparse.Namespace) -> dict:
    from decode_gru_trainer import train_decode_gru_controller

    return train_decode_gru_controller(
        dataset_dir=Path(args.dataset_dir),
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        teacher_forcing_weight=args.teacher_forcing_weight,
        run_root=Path(args.run_root) if getattr(args, "run_root", None) else None,
    )


def evaluate_decode_candidate(args: argparse.Namespace) -> dict:
    from decode_gru_evaluator import evaluate_decode_gru_controller

    return evaluate_decode_gru_controller(
        dataset_dir=Path(args.dataset_dir),
        artifact_path=Path(args.artifact_path),
        report_path=Path(args.report_path) if getattr(args, "report_path", None) else None,
    )


def register_candidate(args: argparse.Namespace) -> dict:
    tags = {}
    for tag in args.tag:
        key, value = tag.split("=", 1)
        tags[key] = value
    return register_artifact(
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        artifact_path=Path(args.artifact_path),
        training_run_manifest_path=Path(args.training_run_manifest_path),
        evaluation_report_path=Path(args.evaluation_report_path),
        tags=tags or None,
    )


def promote_candidate(args: argparse.Namespace) -> dict:
    return promote_alias(
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        alias=args.alias,
        version=args.version,
        reason=args.reason,
        artifact_kind=getattr(args, "artifact_kind", None),
    )


def write_runtime_artifact_env(args: argparse.Namespace) -> dict:
    from policy_live_rollout import update_env_assignment

    registry_dir = Path(args.registry_dir)
    env_path = Path(args.runtime_env_file)
    changes: dict[str, str | None] = {}

    if getattr(args, "decode_alias", None) or getattr(args, "decode_version", None) is not None:
        decode_path = resolve_artifact_path(
            registry_dir=registry_dir,
            model_name=args.model_name,
            alias=getattr(args, "decode_alias", None),
            version=getattr(args, "decode_version", None),
            artifact_kind="decode_controller",
        )
        update_env_assignment(env_path, "VICUNA_LOCAL_DECODE_CONTROLLER_ARTIFACT", str(decode_path))
        changes["VICUNA_LOCAL_DECODE_CONTROLLER_ARTIFACT"] = str(decode_path)

    if getattr(args, "cvec_alias", None) or getattr(args, "cvec_version", None) is not None:
        cvec_path = resolve_artifact_path(
            registry_dir=registry_dir,
            model_name=args.model_name,
            alias=getattr(args, "cvec_alias", None),
            version=getattr(args, "cvec_version", None),
            artifact_kind="cvec_generator",
        )
        update_env_assignment(env_path, "VICUNA_LOCAL_CVEC_GENERATOR_ARTIFACT", str(cvec_path))
        changes["VICUNA_LOCAL_CVEC_GENERATOR_ARTIFACT"] = str(cvec_path)

    if getattr(args, "clear_decode", False):
        update_env_assignment(env_path, "VICUNA_LOCAL_DECODE_CONTROLLER_ARTIFACT", "")
        changes["VICUNA_LOCAL_DECODE_CONTROLLER_ARTIFACT"] = ""

    if getattr(args, "clear_cvec", False):
        update_env_assignment(env_path, "VICUNA_LOCAL_CVEC_GENERATOR_ARTIFACT", "")
        changes["VICUNA_LOCAL_CVEC_GENERATOR_ARTIFACT"] = ""

    return {
        "ok": True,
        "runtime_env_file": str(env_path),
        "changes": changes,
    }


def serve_registry(args: argparse.Namespace) -> dict:
    run_registry_server(
        host=args.host,
        port=args.port,
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        default_alias=args.default_alias,
        fallback_alias=args.fallback_alias,
    )
    return {
        "ok": True,
        "host": args.host,
        "port": args.port,
        "registry_dir": args.registry_dir,
        "model_name": args.model_name,
        "default_alias": args.default_alias,
        "fallback_alias": args.fallback_alias,
    }


def advance_live_rollout(args: argparse.Namespace) -> dict:
    emit_log(
        "policy-rollout",
        "advance_started",
        model_name=args.model_name,
        server=args.server,
        state_path=args.state_path,
    )
    result = advance_rollout(
        LiveRolloutConfig(
            server=args.server,
            registry_dir=Path(args.registry_dir),
            model_name=args.model_name,
            runtime_env_file=Path(args.runtime_env_file),
            runtime_service=args.runtime_service,
            state_path=Path(args.state_path),
            journal_dir=Path(args.journal_dir),
            shadow_min_requests=args.shadow_min_requests,
            shadow_max_disagreement_rate=args.shadow_max_disagreement_rate,
            shadow_max_candidate_failure_rate=args.shadow_max_candidate_failure_rate,
        )
    )
    emit_log(
        "policy-rollout",
        "advance_finished",
        action=result.get("action"),
        reason=result.get("reason"),
        candidate_version=result.get("candidate_version"),
        champion_version=result.get("champion_version"),
    )
    return result


def deploy_runtime_artifact(args: argparse.Namespace) -> dict:
    client = PolicyRuntimeClient(args.server, timeout_s=args.timeout_ms / 1000.0)
    payload: dict[str, object] = {
        "artifact_kind": args.artifact_kind,
        "slot": args.slot,
    }
    if getattr(args, "mode", None):
        payload["mode"] = args.mode
    if getattr(args, "current_rollout_step_index", None) is not None:
        payload["current_rollout_step_index"] = args.current_rollout_step_index
    if getattr(args, "reset_metrics", False):
        payload["reset_metrics"] = True
    if getattr(args, "clear", False):
        payload["clear"] = True
        return client.apply_runtime_artifact(payload)

    if args.artifact_path:
        artifact_payload = json.loads(Path(args.artifact_path).read_text(encoding="utf-8"))
    else:
        if not args.registry_dir or not args.model_name:
            raise ValueError("registry_dir and model_name are required when artifact_path is omitted")
        resolved = resolve_artifact_path(
            registry_dir=Path(args.registry_dir),
            model_name=args.model_name,
            alias=getattr(args, "artifact_alias", None),
            version=getattr(args, "registry_version", None),
            artifact_kind=args.artifact_kind,
        )
        artifact_payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    payload["artifact"] = artifact_payload
    if getattr(args, "artifact_alias", None):
        payload["artifact_alias"] = args.artifact_alias
    if getattr(args, "artifact_version", None):
        payload["artifact_version"] = args.artifact_version
    elif args.artifact_kind == "decode_controller":
        payload["artifact_version"] = artifact_payload.get("controller_version")
    elif args.artifact_kind == "cvec_generator":
        payload["artifact_version"] = artifact_payload.get("generator_version")
    return client.apply_runtime_artifact(payload)


def advance_non_request_rollout(args: argparse.Namespace) -> dict:
    config = ArtifactLiveRolloutConfig(
        server=args.server,
        registry_dir=Path(args.registry_dir),
        model_name=args.model_name,
        artifact_kind=args.artifact_kind,
        state_path=Path(args.state_path),
        shadow_min_comparisons=args.shadow_min_comparisons,
        shadow_max_disagreement_rate=args.shadow_max_disagreement_rate,
        shadow_min_mean_cosine_similarity=args.shadow_min_mean_cosine_similarity,
        shadow_max_mean_norm_delta=args.shadow_max_mean_norm_delta,
        canary_min_samples=args.canary_min_samples,
        canary_max_disagreement_rate=args.canary_max_disagreement_rate,
        canary_min_mean_cosine_similarity=args.canary_min_mean_cosine_similarity,
        canary_max_mean_norm_delta=args.canary_max_mean_norm_delta,
    )
    return advance_artifact_rollout(config)


def advance_all_rollouts(args: argparse.Namespace) -> dict:
    results = {
        "ppo_policy": advance_rollout(
            LiveRolloutConfig(
                server=args.server,
                registry_dir=Path(args.registry_dir),
                model_name=args.model_name,
                runtime_env_file=Path(args.runtime_env_file),
                runtime_service=args.runtime_service,
                state_path=Path(args.policy_state_path),
                journal_dir=Path(args.journal_dir),
                shadow_min_requests=args.shadow_min_requests,
                shadow_max_disagreement_rate=args.shadow_max_disagreement_rate,
                shadow_max_candidate_failure_rate=args.shadow_max_candidate_failure_rate,
            )
        ),
        "decode_controller": advance_artifact_rollout(
            ArtifactLiveRolloutConfig(
                server=args.server,
                registry_dir=Path(args.registry_dir),
                model_name=args.model_name,
                artifact_kind="decode_controller",
                state_path=Path(args.decode_state_path),
                shadow_min_comparisons=args.artifact_shadow_min_comparisons,
                shadow_max_disagreement_rate=args.artifact_shadow_max_disagreement_rate,
                shadow_min_mean_cosine_similarity=args.artifact_shadow_min_mean_cosine_similarity,
                shadow_max_mean_norm_delta=args.artifact_shadow_max_mean_norm_delta,
                canary_min_samples=args.artifact_canary_min_samples,
                canary_max_disagreement_rate=args.artifact_canary_max_disagreement_rate,
                canary_min_mean_cosine_similarity=args.artifact_canary_min_mean_cosine_similarity,
                canary_max_mean_norm_delta=args.artifact_canary_max_mean_norm_delta,
            )
        ),
        "cvec_generator": advance_artifact_rollout(
            ArtifactLiveRolloutConfig(
                server=args.server,
                registry_dir=Path(args.registry_dir),
                model_name=args.model_name,
                artifact_kind="cvec_generator",
                state_path=Path(args.cvec_state_path),
                shadow_min_comparisons=args.artifact_shadow_min_comparisons,
                shadow_max_disagreement_rate=args.artifact_shadow_max_disagreement_rate,
                shadow_min_mean_cosine_similarity=args.artifact_shadow_min_mean_cosine_similarity,
                shadow_max_mean_norm_delta=args.artifact_shadow_max_mean_norm_delta,
                canary_min_samples=args.artifact_canary_min_samples,
                canary_max_disagreement_rate=args.artifact_canary_max_disagreement_rate,
                canary_min_mean_cosine_similarity=args.artifact_canary_min_mean_cosine_similarity,
                canary_max_mean_norm_delta=args.artifact_canary_max_mean_norm_delta,
            )
        ),
    }
    return {"ok": True, "results": results}


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "min_record_count": int(args.min_record_count),
        "min_exact_match_rate": float(args.min_exact_match_rate),
        "max_invalid_action_rate": float(args.max_invalid_action_rate),
        "min_reward_delta": float(args.min_reward_delta),
    }


def _artifact_command(artifact_path: Path) -> str:
    adapter_path = Path(__file__).resolve().parent / "registry_policy_adapter.py"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(adapter_path))} --artifact {shlex.quote(str(artifact_path))}"


def nightly_batch(args: argparse.Namespace) -> dict:
    registry_dir = Path(args.registry_dir)
    thresholds = _thresholds_from_args(args)
    batch_run_id = f"nightly-{int(time.time() * 1000)}"
    started_at_ms = int(time.time() * 1000)
    steps: dict[str, dict] = {}
    promotion_decision = {"decision": "not_started"}
    candidate_version = None
    baseline_version = None
    status = "passed"
    emit_log(
        "policy-nightly",
        "nightly_batch_started",
        batch_run_id=batch_run_id,
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
    )

    try:
        export_args = argparse.Namespace(
            server=args.server,
            dataset_dir=args.dataset_dir,
            dataset_id=args.dataset_id,
            limit=args.limit,
            timeout_ms=args.timeout_ms,
            follow=False,
            max_polls=None,
            poll_interval_s=1.0,
        )
        steps["export"] = export_dataset(export_args)
        emit_log("policy-nightly", "nightly_step_completed", batch_run_id=batch_run_id, step="export")
        steps["build_training_set"] = build_ppo_training_set(argparse.Namespace(dataset_dir=args.dataset_dir))
        emit_log("policy-nightly", "nightly_step_completed", batch_run_id=batch_run_id, step="build_training_set")
        if int(steps["build_training_set"]["record_count"]) < int(args.min_record_count):
            raise ValueError(
                f"training corpus below min_record_count={args.min_record_count}: "
                f"{steps['build_training_set']['record_count']}"
            )

        train_args = argparse.Namespace(
            dataset_dir=args.dataset_dir,
            model_name=args.model_name,
            run_root=args.run_root,
            registry_dir=args.registry_dir,
        )
        steps["train"] = train_candidate(train_args)
        emit_log(
            "policy-nightly",
            "nightly_step_completed",
            batch_run_id=batch_run_id,
            step="train",
            policy_version=steps["train"].get("policy_version"),
        )
        candidate_report_path = (
            Path(args.dataset_dir)
            / "reports"
            / f"offline_eval_{steps['train']['policy_version']}.json"
        )
        eval_args = argparse.Namespace(
            dataset_dir=args.dataset_dir,
            candidate_command=_artifact_command(Path(steps["train"]["artifact_path"])),
            candidate_url=None,
            limit=args.limit,
            timeout_ms=args.timeout_ms,
            report_path=str(candidate_report_path),
        )
        steps["evaluate_candidate"] = evaluate_candidate(eval_args)
        emit_log("policy-nightly", "nightly_step_completed", batch_run_id=batch_run_id, step="evaluate_candidate")

        baseline_alias = None
        for alias_name in ["candidate", "champion"]:
            alias_version = get_alias_version(registry_dir, args.model_name, alias_name)
            if alias_version is not None:
                baseline_alias = alias_name
                baseline_version = alias_version
                break
        if baseline_alias is not None and baseline_version is not None:
            baseline_artifact_path = resolve_artifact_path(
                registry_dir=registry_dir,
                model_name=args.model_name,
                alias=baseline_alias,
            )
            baseline_report_path = (
                Path(args.dataset_dir)
                / "reports"
                / f"offline_eval_baseline_{baseline_alias}_{baseline_version}.json"
            )
            baseline_eval_args = argparse.Namespace(
                dataset_dir=args.dataset_dir,
                candidate_command=_artifact_command(baseline_artifact_path),
                candidate_url=None,
                limit=args.limit,
                timeout_ms=args.timeout_ms,
                report_path=str(baseline_report_path),
            )
            steps["evaluate_baseline"] = evaluate_candidate(baseline_eval_args)
            emit_log("policy-nightly", "nightly_step_completed", batch_run_id=batch_run_id, step="evaluate_baseline")

        reward_delta = 0.0
        baseline_match_reward = None
        if "evaluate_baseline" in steps:
            baseline_match_reward = steps["evaluate_baseline"].get("reward_total_mean_on_match")
            candidate_match_reward = steps["evaluate_candidate"].get("reward_total_mean_on_match")
            if baseline_match_reward is not None and candidate_match_reward is not None:
                reward_delta = float(candidate_match_reward) - float(baseline_match_reward)

        validation_reasons = []
        if float(steps["evaluate_candidate"]["exact_match_rate"]) < float(args.min_exact_match_rate):
            validation_reasons.append("exact_match_rate below threshold")
        if float(steps["evaluate_candidate"]["invalid_action_rate"]) > float(args.max_invalid_action_rate):
            validation_reasons.append("invalid_action_rate above threshold")
        if reward_delta < float(args.min_reward_delta):
            validation_reasons.append("reward_delta below threshold")

        register_args = argparse.Namespace(
            registry_dir=args.registry_dir,
            model_name=args.model_name,
            artifact_path=steps["train"]["artifact_path"],
            training_run_manifest_path=steps["train"]["training_run_manifest_path"],
            evaluation_report_path=steps["evaluate_candidate"]["report_path"],
            tag=[
                f"batch_run_id={batch_run_id}",
                f"validation_status={'passed' if not validation_reasons else 'rejected'}",
            ],
        )
        steps["register"] = register_candidate(register_args)
        candidate_version = steps["register"]["version"]
        emit_log(
            "policy-nightly",
            "nightly_step_completed",
            batch_run_id=batch_run_id,
            step="register",
            candidate_version=candidate_version,
        )

        if validation_reasons:
            promotion_decision = promote_alias(
                registry_dir=registry_dir,
                model_name=args.model_name,
                alias="candidate",
                version=candidate_version,
                reason="; ".join(validation_reasons),
                thresholds={**thresholds, "reward_delta": reward_delta},
                decision="rejected",
            )
        else:
            promotion_decision = promote_alias(
                registry_dir=registry_dir,
                model_name=args.model_name,
                alias="candidate",
                version=candidate_version,
                reason="nightly thresholds passed",
                thresholds={**thresholds, "reward_delta": reward_delta},
                decision="promoted",
            )
        steps["promote"] = promotion_decision
        emit_log(
            "policy-nightly",
            "nightly_step_completed",
            batch_run_id=batch_run_id,
            step="promote",
            decision=promotion_decision.get("decision"),
            candidate_version=candidate_version,
        )
    except Exception as exc:
        status = "failed"
        steps["error"] = {
            "message": str(exc),
            "type": exc.__class__.__name__,
        }
        emit_log(
            "policy-nightly",
            "nightly_batch_failed",
            batch_run_id=batch_run_id,
            error=str(exc),
            error_type=exc.__class__.__name__,
        )

    finished_at_ms = int(time.time() * 1000)
    batch_manifest = {
        "schema_version": BATCH_RUN_SCHEMA_VERSION,
        "batch_run_id": batch_run_id,
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "dataset_dir": args.dataset_dir,
        "registry_dir": args.registry_dir,
        "model_name": args.model_name,
        "steps": steps,
        "thresholds": thresholds,
        "promotion_decision": promotion_decision,
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
        "status": status,
    }
    batch_run_path = write_batch_run(
        registry_dir=registry_dir,
        model_name=args.model_name,
        batch_run_id=batch_run_id,
        payload=batch_manifest,
    )
    batch_manifest["batch_run_manifest_path"] = str(batch_run_path)
    emit_log(
        "policy-nightly",
        "nightly_batch_finished",
        batch_run_id=batch_run_id,
        status=status,
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        batch_run_manifest_path=str(batch_run_path),
    )
    return batch_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vicuña policy-learning offline tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export runtime transitions")
    export_parser.add_argument("--server", required=True)
    export_parser.add_argument("--dataset-dir", required=True)
    export_parser.add_argument("--dataset-id", required=True)
    export_parser.add_argument("--limit", type=int, default=512)
    export_parser.add_argument("--decode-limit", type=int, default=128)
    export_parser.add_argument("--timeout-ms", type=int, default=5000)
    export_parser.add_argument("--follow", action="store_true")
    export_parser.add_argument("--poll-interval-s", type=float, default=1.0)
    export_parser.add_argument("--max-polls", type=int)

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a candidate policy")
    eval_parser.add_argument("--dataset-dir", required=True)
    eval_parser.add_argument("--candidate-command")
    eval_parser.add_argument("--candidate-url")
    eval_parser.add_argument("--limit", type=int)
    eval_parser.add_argument("--timeout-ms", type=int, default=5000)
    eval_parser.add_argument("--report-path")

    eval_cvec_parser = subparsers.add_parser("evaluate-cvec-generator", help="evaluate a cvec generator artifact")
    eval_cvec_parser.add_argument("--dataset-dir", required=True)
    eval_cvec_parser.add_argument("--artifact-path", required=True)
    eval_cvec_parser.add_argument("--report-path")

    eval_ppo_parser = subparsers.add_parser("evaluate-ppo", help="evaluate a PPO policy artifact")
    eval_ppo_parser.add_argument("--dataset-dir", required=True)
    eval_ppo_parser.add_argument("--artifact-path", required=True)
    eval_ppo_parser.add_argument("--report-path")

    build_parser_cmd = subparsers.add_parser(
        "build-training-set", help="materialize training-contract records"
    )
    build_parser_cmd.add_argument("--dataset-dir", required=True)

    build_cvec_parser = subparsers.add_parser(
        "build-cvec-training-set", help="materialize EM/VAD-to-cvec training records"
    )
    build_cvec_parser.add_argument("--dataset-dir", required=True)
    build_cvec_parser.add_argument("--target-embedding-dim", type=int, required=True)
    build_cvec_parser.add_argument("--vector-library-path")

    build_ppo_parser = subparsers.add_parser(
        "build-ppo-training-set", help="materialize PPO training records"
    )
    build_ppo_parser.add_argument("--dataset-dir", required=True)

    build_decode_parser = subparsers.add_parser(
        "build-decode-training-set", help="materialize decode-level GRU training records"
    )
    build_decode_parser.add_argument("--dataset-dir", required=True)
    build_decode_parser.add_argument("--sequence-length", type=int, default=32)
    build_decode_parser.add_argument("--stride", type=int, default=16)

    train_parser = subparsers.add_parser("train", help="train a policy artifact")
    train_parser.add_argument("--dataset-dir", required=True)
    train_parser.add_argument("--model-name", required=True)
    train_parser.add_argument("--run-root")
    train_parser.add_argument("--hidden-dims", default="64,64")
    train_parser.add_argument("--warmstart-epochs", type=int, default=250)
    train_parser.add_argument("--ppo-epochs", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--clip-coef", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--vf-coef", type=float, default=0.5)

    train_ppo_parser = subparsers.add_parser("train-ppo", help="train a PPO policy artifact")
    train_ppo_parser.add_argument("--dataset-dir", required=True)
    train_ppo_parser.add_argument("--model-name", required=True)
    train_ppo_parser.add_argument("--run-root")
    train_ppo_parser.add_argument("--hidden-dims", default="64,64")
    train_ppo_parser.add_argument("--warmstart-epochs", type=int, default=250)
    train_ppo_parser.add_argument("--ppo-epochs", type=int, default=200)
    train_ppo_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_ppo_parser.add_argument("--clip-coef", type=float, default=0.2)
    train_ppo_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_ppo_parser.add_argument("--vf-coef", type=float, default=0.5)

    train_cvec_parser = subparsers.add_parser("train-cvec-generator", help="train an EM/VAD control-vector generator")
    train_cvec_parser.add_argument("--dataset-dir", required=True)
    train_cvec_parser.add_argument("--model-name", required=True)
    train_cvec_parser.add_argument("--target-embedding-dim", type=int, required=True)
    train_cvec_parser.add_argument("--target-layer-start", type=int, default=0)
    train_cvec_parser.add_argument("--target-layer-end", type=int, default=-1)
    train_cvec_parser.add_argument("--vector-library-path")
    train_cvec_parser.add_argument("--hidden-dims", default="64,64")
    train_cvec_parser.add_argument("--epochs", type=int, default=600)
    train_cvec_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_cvec_parser.add_argument("--reward-weight-power", type=float, default=1.0)
    train_cvec_parser.add_argument("--output-norm-cap", type=float, default=8.0)
    train_cvec_parser.add_argument("--output-mode", default="none")
    train_cvec_parser.add_argument("--activation", default="tanh")
    train_cvec_parser.add_argument("--run-root")

    train_decode_parser = subparsers.add_parser(
        "train-decode-controller", help="train a decode-level GRU controller artifact"
    )
    train_decode_parser.add_argument("--dataset-dir", required=True)
    train_decode_parser.add_argument("--model-name", required=True)
    train_decode_parser.add_argument("--run-root")
    train_decode_parser.add_argument("--hidden-dim", type=int, default=96)
    train_decode_parser.add_argument("--epochs", type=int, default=25)
    train_decode_parser.add_argument("--batch-size", type=int, default=16)
    train_decode_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_decode_parser.add_argument("--teacher-forcing-weight", type=float, default=1.0)

    register_parser = subparsers.add_parser("register", help="register an artifact")
    register_parser.add_argument("--registry-dir", required=True)
    register_parser.add_argument("--model-name", required=True)
    register_parser.add_argument("--artifact-path", required=True)
    register_parser.add_argument("--training-run-manifest-path", required=True)
    register_parser.add_argument("--evaluation-report-path", required=True)
    register_parser.add_argument("--tag", action="append", default=[])

    promote_parser = subparsers.add_parser("promote", help="assign a registry alias")
    promote_parser.add_argument("--registry-dir", required=True)
    promote_parser.add_argument("--model-name", required=True)
    promote_parser.add_argument("--alias", required=True)
    promote_parser.add_argument("--version", required=True, type=int)
    promote_parser.add_argument("--reason", required=True)
    promote_parser.add_argument("--artifact-kind")

    eval_decode_parser = subparsers.add_parser(
        "evaluate-decode-controller", help="evaluate a decode-level GRU controller artifact"
    )
    eval_decode_parser.add_argument("--dataset-dir", required=True)
    eval_decode_parser.add_argument("--artifact-path", required=True)
    eval_decode_parser.add_argument("--report-path")

    serve_parser = subparsers.add_parser(
        "serve-registry", help="serve registry-backed policy proposals"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=18081, type=int)
    serve_parser.add_argument("--registry-dir", required=True)
    serve_parser.add_argument("--model-name", required=True)
    serve_parser.add_argument("--default-alias", default="candidate")
    serve_parser.add_argument("--fallback-alias")

    rollout_parser = subparsers.add_parser(
        "advance-rollout", help="advance live candidate rollout based on runtime evidence"
    )
    rollout_parser.add_argument("--server", required=True)
    rollout_parser.add_argument("--registry-dir", required=True)
    rollout_parser.add_argument("--model-name", required=True)
    rollout_parser.add_argument("--runtime-env-file", required=True)
    rollout_parser.add_argument("--runtime-service", required=True)
    rollout_parser.add_argument("--state-path", required=True)
    rollout_parser.add_argument("--journal-dir", required=True)
    rollout_parser.add_argument("--shadow-min-requests", type=int, default=25)
    rollout_parser.add_argument("--shadow-max-disagreement-rate", type=float, default=0.25)
    rollout_parser.add_argument("--shadow-max-candidate-failure-rate", type=float, default=0.10)

    artifact_rollout_parser = subparsers.add_parser(
        "advance-artifact-rollout", help="advance live rollout for decode-controller or cvec artifacts"
    )
    artifact_rollout_parser.add_argument("--server", required=True)
    artifact_rollout_parser.add_argument("--registry-dir", required=True)
    artifact_rollout_parser.add_argument("--model-name", required=True)
    artifact_rollout_parser.add_argument("--artifact-kind", required=True, choices=["decode_controller", "cvec_generator"])
    artifact_rollout_parser.add_argument("--state-path", required=True)
    artifact_rollout_parser.add_argument("--shadow-min-comparisons", type=int, default=64)
    artifact_rollout_parser.add_argument("--shadow-max-disagreement-rate", type=float, default=0.25)
    artifact_rollout_parser.add_argument("--shadow-min-mean-cosine-similarity", type=float, default=0.90)
    artifact_rollout_parser.add_argument("--shadow-max-mean-norm-delta", type=float, default=0.30)
    artifact_rollout_parser.add_argument("--canary-min-samples", type=int, default=8)
    artifact_rollout_parser.add_argument("--canary-max-disagreement-rate", type=float, default=0.20)
    artifact_rollout_parser.add_argument("--canary-min-mean-cosine-similarity", type=float, default=0.88)
    artifact_rollout_parser.add_argument("--canary-max-mean-norm-delta", type=float, default=0.35)

    all_rollouts_parser = subparsers.add_parser(
        "advance-all-rollouts", help="advance PPO, decode-controller, and cvec rollouts together"
    )
    all_rollouts_parser.add_argument("--server", required=True)
    all_rollouts_parser.add_argument("--registry-dir", required=True)
    all_rollouts_parser.add_argument("--model-name", required=True)
    all_rollouts_parser.add_argument("--runtime-env-file", required=True)
    all_rollouts_parser.add_argument("--runtime-service", required=True)
    all_rollouts_parser.add_argument("--journal-dir", required=True)
    all_rollouts_parser.add_argument("--policy-state-path", required=True)
    all_rollouts_parser.add_argument("--decode-state-path", required=True)
    all_rollouts_parser.add_argument("--cvec-state-path", required=True)
    all_rollouts_parser.add_argument("--shadow-min-requests", type=int, default=25)
    all_rollouts_parser.add_argument("--shadow-max-disagreement-rate", type=float, default=0.25)
    all_rollouts_parser.add_argument("--shadow-max-candidate-failure-rate", type=float, default=0.10)
    all_rollouts_parser.add_argument("--artifact-shadow-min-comparisons", type=int, default=64)
    all_rollouts_parser.add_argument("--artifact-shadow-max-disagreement-rate", type=float, default=0.25)
    all_rollouts_parser.add_argument("--artifact-shadow-min-mean-cosine-similarity", type=float, default=0.90)
    all_rollouts_parser.add_argument("--artifact-shadow-max-mean-norm-delta", type=float, default=0.30)
    all_rollouts_parser.add_argument("--artifact-canary-min-samples", type=int, default=8)
    all_rollouts_parser.add_argument("--artifact-canary-max-disagreement-rate", type=float, default=0.20)
    all_rollouts_parser.add_argument("--artifact-canary-min-mean-cosine-similarity", type=float, default=0.88)
    all_rollouts_parser.add_argument("--artifact-canary-max-mean-norm-delta", type=float, default=0.35)

    registry_parser = subparsers.add_parser(
        "registry-status", help="inspect the registry"
    )
    registry_parser.add_argument("--registry-dir", required=True)
    registry_parser.add_argument("--model-name", required=True)

    write_env_parser = subparsers.add_parser(
        "write-runtime-artifact-env", help="write promoted decode/cvec artifact paths into a runtime env file"
    )
    write_env_parser.add_argument("--registry-dir", required=True)
    write_env_parser.add_argument("--model-name", required=True)
    write_env_parser.add_argument("--runtime-env-file", required=True)
    write_env_parser.add_argument("--decode-alias")
    write_env_parser.add_argument("--decode-version", type=int)
    write_env_parser.add_argument("--cvec-alias")
    write_env_parser.add_argument("--cvec-version", type=int)
    write_env_parser.add_argument("--clear-decode", action="store_true")
    write_env_parser.add_argument("--clear-cvec", action="store_true")

    deploy_runtime_parser = subparsers.add_parser(
        "deploy-runtime-artifact", help="hot-load or clear a runtime artifact on a running server"
    )
    deploy_runtime_parser.add_argument("--server", required=True)
    deploy_runtime_parser.add_argument("--artifact-kind", required=True, choices=["decode_controller", "cvec_generator"])
    deploy_runtime_parser.add_argument("--slot", required=True, choices=["active", "candidate"])
    deploy_runtime_parser.add_argument("--artifact-path")
    deploy_runtime_parser.add_argument("--registry-dir")
    deploy_runtime_parser.add_argument("--model-name")
    deploy_runtime_parser.add_argument("--artifact-alias")
    deploy_runtime_parser.add_argument("--registry-version", type=int)
    deploy_runtime_parser.add_argument("--artifact-version")
    deploy_runtime_parser.add_argument("--mode")
    deploy_runtime_parser.add_argument("--current-rollout-step-index", type=int)
    deploy_runtime_parser.add_argument("--timeout-ms", type=int, default=5000)
    deploy_runtime_parser.add_argument("--reset-metrics", action="store_true")
    deploy_runtime_parser.add_argument("--clear", action="store_true")

    batch_parser = subparsers.add_parser(
        "nightly-batch", help="run export, train, evaluate, and register once"
    )
    batch_parser.add_argument("--server", required=True)
    batch_parser.add_argument("--dataset-dir", required=True)
    batch_parser.add_argument("--dataset-id", required=True)
    batch_parser.add_argument("--registry-dir", required=True)
    batch_parser.add_argument("--model-name", required=True)
    batch_parser.add_argument("--run-root")
    batch_parser.add_argument("--limit", type=int, default=512)
    batch_parser.add_argument("--timeout-ms", type=int, default=5000)
    batch_parser.add_argument("--min-record-count", type=int, default=25)
    batch_parser.add_argument("--min-exact-match-rate", type=float, default=0.55)
    batch_parser.add_argument("--max-invalid-action-rate", type=float, default=0.0)
    batch_parser.add_argument("--min-reward-delta", type=float, default=0.0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "export":
        payload = export_dataset(args)
    elif args.command == "evaluate":
        payload = evaluate_candidate(args)
    elif args.command == "evaluate-cvec-generator":
        payload = evaluate_cvec_candidate(args)
    elif args.command == "evaluate-ppo":
        payload = evaluate_ppo_candidate(args)
    elif args.command == "build-training-set":
        payload = build_training_corpus(Path(args.dataset_dir))
    elif args.command == "build-cvec-training-set":
        payload = build_cvec_training_set(args)
    elif args.command == "build-ppo-training-set":
        payload = build_ppo_training_set(args)
    elif args.command == "build-decode-training-set":
        payload = build_decode_training_set(args)
    elif args.command == "train":
        payload = train_candidate(args)
    elif args.command == "train-ppo":
        payload = train_candidate(args)
    elif args.command == "train-cvec-generator":
        payload = train_cvec_candidate(args)
    elif args.command == "train-decode-controller":
        payload = train_decode_candidate(args)
    elif args.command == "register":
        payload = register_candidate(args)
    elif args.command == "promote":
        payload = promote_candidate(args)
    elif args.command == "evaluate-decode-controller":
        payload = evaluate_decode_candidate(args)
    elif args.command == "serve-registry":
        payload = serve_registry(args)
    elif args.command == "advance-rollout":
        payload = advance_live_rollout(args)
    elif args.command == "advance-artifact-rollout":
        payload = advance_non_request_rollout(args)
    elif args.command == "advance-all-rollouts":
        payload = advance_all_rollouts(args)
    elif args.command == "registry-status":
        payload = registry_status(Path(args.registry_dir), args.model_name)
    elif args.command == "write-runtime-artifact-env":
        payload = write_runtime_artifact_env(args)
    elif args.command == "deploy-runtime-artifact":
        payload = deploy_runtime_artifact(args)
    elif args.command == "nightly-batch":
        payload = nightly_batch(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
