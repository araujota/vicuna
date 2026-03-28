#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from policy_dataset import append_transitions
from policy_evaluator import evaluate_dataset
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
from policy_trainer import train_policy
from policy_training_contract import build_training_corpus


def export_dataset(args: argparse.Namespace) -> dict:
    client = PolicyRuntimeClient(args.server, timeout_s=args.timeout_ms / 1000.0)
    summary = {}
    export_mode = "follow" if args.follow else "snapshot"
    polls = 0
    while True:
        status = client.get_status()
        transitions_payload = client.get_transitions(limit=args.limit)
        summary = append_transitions(
            dataset_dir=Path(args.dataset_dir),
            dataset_id=args.dataset_id,
            source_base_url=args.server,
            export_mode=export_mode,
            status_payload=status,
            transitions=transitions_payload.get("items", []),
        )
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


def train_candidate(args: argparse.Namespace) -> dict:
    return train_policy(
        dataset_dir=Path(args.dataset_dir),
        model_name=args.model_name,
        run_root=Path(args.run_root) if getattr(args, "run_root", None) else None,
        registry_dir=Path(args.registry_dir) if getattr(args, "registry_dir", None) else None,
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
    )


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
        steps["build_training_set"] = build_training_corpus(Path(args.dataset_dir))
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
    except Exception as exc:
        status = "failed"
        steps["error"] = {
            "message": str(exc),
            "type": exc.__class__.__name__,
        }

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
    return batch_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vicuña policy-learning offline tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export runtime transitions")
    export_parser.add_argument("--server", required=True)
    export_parser.add_argument("--dataset-dir", required=True)
    export_parser.add_argument("--dataset-id", required=True)
    export_parser.add_argument("--limit", type=int, default=512)
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

    build_parser_cmd = subparsers.add_parser(
        "build-training-set", help="materialize training-contract records"
    )
    build_parser_cmd.add_argument("--dataset-dir", required=True)

    train_parser = subparsers.add_parser("train", help="train a policy artifact")
    train_parser.add_argument("--dataset-dir", required=True)
    train_parser.add_argument("--model-name", required=True)
    train_parser.add_argument("--registry-dir")
    train_parser.add_argument("--run-root")

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

    serve_parser = subparsers.add_parser(
        "serve-registry", help="serve registry-backed policy proposals"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=18081, type=int)
    serve_parser.add_argument("--registry-dir", required=True)
    serve_parser.add_argument("--model-name", required=True)
    serve_parser.add_argument("--default-alias", default="candidate")
    serve_parser.add_argument("--fallback-alias")

    registry_parser = subparsers.add_parser(
        "registry-status", help="inspect the registry"
    )
    registry_parser.add_argument("--registry-dir", required=True)
    registry_parser.add_argument("--model-name", required=True)

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
    elif args.command == "build-training-set":
        payload = build_training_corpus(Path(args.dataset_dir))
    elif args.command == "train":
        payload = train_candidate(args)
    elif args.command == "register":
        payload = register_candidate(args)
    elif args.command == "promote":
        payload = promote_candidate(args)
    elif args.command == "serve-registry":
        payload = serve_registry(args)
    elif args.command == "registry-status":
        payload = registry_status(Path(args.registry_dir), args.model_name)
    elif args.command == "nightly-batch":
        payload = nightly_batch(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
