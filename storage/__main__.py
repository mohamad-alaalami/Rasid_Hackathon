"""Command line for the GoTrain store: ``python -m storage <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import STORE_ENV, get_store
from .datasets import list_stored_datasets, pull_dataset, push_dataset
from .logs import push_logs
from .model_versions import (
    download_version,
    find_version_for_experiment,
    list_versions,
    promote_version,
    register_experiment,
    run_final_test,
    summarize,
)


def _version_arg(value: str) -> int | str:
    return value if value in {"champion", "latest"} else int(value.lstrip("v"))


def _doctor(store) -> dict:
    report = {"configured": os.environ.get(STORE_ENV) or None}
    if store is None:
        report["status"] = "disabled"
        report["hint"] = f"Set {STORE_ENV}=s3://<bucket>/gotrain (or a directory) to enable."
        try:
            import boto3

            session = boto3.session.Session()
            account = boto3.client("sts").get_caller_identity()["Account"]
            report["suggested"] = f"s3://sagemaker-{session.region_name}-{account}/gotrain"
        except Exception as exc:
            report["aws"] = f"unavailable ({type(exc).__name__})"
        return report
    probe = ".doctor/probe.json"
    try:
        store.write_json(probe, {"probe": True})
        readable = store.read_json(probe) == {"probe": True}
        store.delete(probe)
    except Exception as exc:
        return {**report, "status": "error", "uri": store.uri, "error": f"{type(exc).__name__}: {exc}"}
    return {**report, "status": "ok" if readable else "error", "backend": store.kind, "uri": store.uri}


def _register(store, args) -> list[dict]:
    root = Path(args.experiments_dir)
    directories = (
        sorted(path.parent for path in root.glob("*/result.json"))
        if args.all else [root / experiment_id for experiment_id in args.experiment_ids]
    )
    outcomes = []
    for directory in directories:
        result_path = directory / "result.json"
        if not result_path.is_file():
            outcomes.append({"experiment": directory.name, "status": "missing"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            outcomes.append({"experiment": directory.name, "status": "skipped_failed_run"})
            continue
        existing = find_version_for_experiment(store, result["dataset_id"], result["experiment_id"])
        if existing is not None:
            outcomes.append({"experiment": directory.name, "status": "already_versioned", "name": existing["name"]})
            continue
        try:
            record = register_experiment(store, result, directory, source="backfill")
        except Exception as exc:
            outcomes.append({"experiment": directory.name, "status": "error",
                             "message": f"{type(exc).__name__}: {exc}"})
            continue
        outcomes.append({"experiment": directory.name, "status": "registered", "name": record["name"]})
    return outcomes


def _final_test(store, args) -> dict:
    if not args.yes:
        raise SystemExit(
            "final-test uses the test split exactly once for the dataset's champion and locks "
            "its final model. Re-run with --yes to authorize it."
        )
    from ml.test_evaluation import evaluate_checkpoint_on_test

    return run_final_test(
        store,
        args.dataset,
        lambda checkpoint: evaluate_checkpoint_on_test(
            checkpoint,
            benchmark_dir=args.benchmark_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        retry_unfinished=args.retry_unfinished,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m storage", description=__doc__)
    parser.add_argument("--store", help=f"Store URI; defaults to ${STORE_ENV}.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Check that the store is reachable and writable.")
    commands.add_parser("summary", help="Champion, best validation, and test score per dataset.")
    commands.add_parser("datasets", help="List datasets stored in the store.")

    for name, help_text in (("push-dataset", "Upload a local GEO-Bench dataset."),
                            ("pull-dataset", "Download a stored dataset into $GEO_BENCH_DIR.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("dataset")
        command.add_argument("--benchmark", help="Benchmark folder for unregistered datasets.")
        command.add_argument("--geo-bench-dir", help="Defaults to $GEO_BENCH_DIR.")
        if name == "push-dataset":
            command.add_argument("--overwrite", action="store_true")

    register = commands.add_parser("register", help="Version existing local experiments (backfill).")
    register.add_argument("experiment_ids", nargs="*")
    register.add_argument("--all", action="store_true", help="Every completed experiment.")
    register.add_argument("--experiments-dir", default="experiments")

    versions = commands.add_parser("versions", help="List a dataset's model versions.")
    versions.add_argument("dataset")

    promote = commands.add_parser("promote", help="Make a version the dataset's champion.")
    promote.add_argument("dataset")
    promote.add_argument("version", type=_version_arg)
    promote.add_argument("--reason", required=True)

    download = commands.add_parser("download", help="Download a model version's files.")
    download.add_argument("dataset")
    download.add_argument("version", type=_version_arg, nargs="?", default="champion")
    download.add_argument("--to", help="Destination directory.")

    final_test = commands.add_parser("final-test", help="One-time test evaluation of the champion.")
    final_test.add_argument("dataset")
    final_test.add_argument("--yes", action="store_true", help="Authorize the one-time test use.")
    final_test.add_argument("--retry-unfinished", action="store_true")
    final_test.add_argument("--benchmark-dir")
    final_test.add_argument("--batch-size", type=int, default=32)
    final_test.add_argument("--num-workers", type=int, default=2)

    logs = commands.add_parser("push-logs", help="Back up campaign logs and experiment records.")
    logs.add_argument("--artifacts-dir", default="artifacts")
    logs.add_argument("--experiments-dir", default="experiments")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = get_store(args.store)
    if args.command == "doctor":
        output = _doctor(store)
    elif store is None:
        raise SystemExit(f"Storage is disabled. Set {STORE_ENV} or pass --store.")
    elif args.command == "summary":
        output = summarize(store)
    elif args.command == "datasets":
        output = list_stored_datasets(store)
    elif args.command == "push-dataset":
        output = push_dataset(store, args.dataset, benchmark_name=args.benchmark,
                              geo_bench_dir=args.geo_bench_dir, overwrite=args.overwrite)
    elif args.command == "pull-dataset":
        output = pull_dataset(store, args.dataset, benchmark_name=args.benchmark,
                              geo_bench_dir=args.geo_bench_dir)
    elif args.command == "register":
        if not args.all and not args.experiment_ids:
            raise SystemExit("Pass experiment IDs or --all.")
        output = _register(store, args)
    elif args.command == "versions":
        output = list_versions(store, args.dataset)
    elif args.command == "promote":
        output = promote_version(store, args.dataset, args.version, args.reason, promoted_by="human")
    elif args.command == "download":
        output = download_version(store, args.dataset, args.version, args.to)
    elif args.command == "final-test":
        output = _final_test(store, args)
    else:
        output = push_logs(store, args.artifacts_dir, args.experiments_dir)
    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
