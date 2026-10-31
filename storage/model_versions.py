"""Versioned model storage for GoTrain.

Every completed training run becomes one immutable, numbered version per
dataset (``m-eurosat/v0003``) holding its checkpoint, configuration, metrics,
and provenance. One version per dataset can be promoted to champion, and only
the champion can receive the single final test-split evaluation, after which
the dataset's final model is locked.

Store layout::

    models/<dataset_id>/v0001/version.json   provenance record (written first and last)
    models/<dataset_id>/v0001/best.pt         checkpoint and run artifacts
    models/<dataset_id>/champion.json         current champion and promotion history
    models/<dataset_id>/test_evaluation.json  the one-time final test result
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .backends import cache_dir, get_store, sha256_file

MODELS_PREFIX = "models"
ARTIFACT_FILES = ("best.pt", "config.json", "metrics.json")
RECORD_FIELDS = (
    "experiment_id",
    "model",
    "task_type",
    "primary_metric",
    "best_val_score",
    "best_epoch",
    "requested_epochs",
    "completed_epochs",
    "stopped_early",
    "training_seconds",
    "device",
    "native_resolution",
    "effective_resolution",
    "preprocessing",
    "pretraining_source",
    "pretrained_checkpoint_id",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_prefix(dataset_id: str) -> str:
    if not dataset_id or "/" in dataset_id:
        raise ValueError(f"Invalid dataset_id: {dataset_id!r}")
    return f"{MODELS_PREFIX}/{dataset_id}"


def _version_prefix(dataset_id: str, version: int) -> str:
    return f"{_dataset_prefix(dataset_id)}/v{version:04d}"


def _record_key(dataset_id: str, version: int) -> str:
    return f"{_version_prefix(dataset_id, version)}/version.json"


def _champion_key(dataset_id: str) -> str:
    return f"{_dataset_prefix(dataset_id)}/champion.json"


def _test_key(dataset_id: str) -> str:
    return f"{_dataset_prefix(dataset_id)}/test_evaluation.json"


def _code_version() -> dict | None:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"git_commit": commit, "uncommitted_changes": bool(dirty)}


def list_versions(store, dataset_id: str) -> list[dict]:
    pattern = re.compile(rf"^{re.escape(_dataset_prefix(dataset_id))}/v(\d{{4,}})/version\.json$")
    numbers = sorted(
        int(match.group(1))
        for key in store.list_keys(_dataset_prefix(dataset_id))
        if (match := pattern.match(key))
    )
    return [record for n in numbers if (record := store.read_json(_record_key(dataset_id, n)))]


def list_model_datasets(store) -> list[str]:
    return sorted({
        key.split("/")[1]
        for key in store.list_keys(MODELS_PREFIX)
        if key.endswith("/version.json")
    })


def get_champion(store, dataset_id: str) -> dict | None:
    return store.read_json(_champion_key(dataset_id))


def get_test_evaluation(store, dataset_id: str) -> dict | None:
    return store.read_json(_test_key(dataset_id))


def get_version(store, dataset_id: str, version: int | str = "champion") -> dict:
    """Resolve an integer version, ``"champion"``, or ``"latest"`` to its record."""
    if version == "champion":
        champion = get_champion(store, dataset_id)
        if champion is None:
            raise ValueError(f"{dataset_id} has no champion; promote a version first.")
        version = champion["version"]
    elif version == "latest":
        versions = list_versions(store, dataset_id)
        if not versions:
            raise ValueError(f"{dataset_id} has no stored model versions.")
        version = versions[-1]["version"]
    record = store.read_json(_record_key(dataset_id, int(version)))
    if record is None:
        raise ValueError(f"{dataset_id} v{int(version):04d} does not exist in {store.uri}.")
    return record


def find_version_for_experiment(store, dataset_id: str, experiment_id: str) -> dict | None:
    return next(
        (record for record in list_versions(store, dataset_id)
         if record.get("experiment_id") == experiment_id),
        None,
    )


def _claim_next_version(store, dataset_id: str, record: dict) -> int:
    existing = list_versions(store, dataset_id)
    version = (existing[-1]["version"] + 1) if existing else 1
    for _ in range(1000):
        claim = {**record, "version": version, "name": f"{dataset_id}/v{version:04d}"}
        if store.create_json(_record_key(dataset_id, version), claim):
            return version
        version += 1
    raise RuntimeError(f"Could not allocate a model version for {dataset_id}.")


def register_experiment(store, result: dict, experiment_dir: str | Path, *, source: str = "training") -> dict:
    """Upload a completed experiment's artifacts as the dataset's next version."""
    if result.get("status") != "completed":
        raise ValueError("Only completed experiments can become model versions.")
    experiment_dir = Path(experiment_dir)
    missing = [name for name in ARTIFACT_FILES if not (experiment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{experiment_dir} is missing {missing}.")
    fields = {field: result.get(field) for field in RECORD_FIELDS}
    if fields["best_val_score"] is None and "best_val_accuracy" in result:
        # Runs from the original m-eurosat-only trainer used this field name.
        fields.update(best_val_score=result["best_val_accuracy"], primary_metric="accuracy")
    if not isinstance(fields["best_val_score"], (int, float)):
        raise ValueError(f"{experiment_dir.name} has no validation score.")
    dataset_id = result["dataset_id"]
    record = {
        "dataset_id": dataset_id,
        "status": "uploading",
        "created_at": _now(),
        "source": source,
        **fields,
        "config": json.loads((experiment_dir / "config.json").read_text(encoding="utf-8")),
        "code": _code_version(),
    }
    version = _claim_next_version(store, dataset_id, record)
    prefix = _version_prefix(dataset_id, version)
    record.update(version=version, name=f"{dataset_id}/v{version:04d}", uri=store.describe(prefix))
    try:
        artifacts = {}
        for name in ARTIFACT_FILES:
            path = experiment_dir / name
            store.upload_file(path, f"{prefix}/{name}")
            artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        store.write_json(f"{prefix}/training_result.json", result)
        record.update(status="ready", artifacts=artifacts, uploaded_at=_now())
    except Exception as exc:
        record.update(status="upload_failed", error=f"{type(exc).__name__}: {exc}")
        store.write_json(_record_key(dataset_id, version), record)
        raise
    store.write_json(_record_key(dataset_id, version), record)
    return record


def publish_experiment(result: dict, experiment_dir: str | Path) -> dict:
    """Trainer hook: version a completed run when $GOTRAIN_STORE is set.

    Returns ``{}`` when storage is disabled, so results are unchanged. Storage
    problems are reported in the result and never fail a finished training run.
    """
    try:
        store = get_store()
        if store is None:
            return {}
        record = register_experiment(store, result, experiment_dir)
        return {"model_version": {"name": record["name"], "version": record["version"], "uri": record["uri"]}}
    except Exception as exc:
        return {"model_version_error": f"{type(exc).__name__}: {exc}"}


def promote_version(store, dataset_id: str, version: int | str, reason: str, promoted_by: str = "agent") -> dict:
    """Make a ready version the dataset's champion (the model to ship and test)."""
    if not reason or not reason.strip():
        raise ValueError("A promotion reason is required.")
    if get_test_evaluation(store, dataset_id) is not None:
        raise ValueError(
            f"{dataset_id}'s final model is locked: the test split was already used once."
        )
    record = get_version(store, dataset_id, version)
    if record.get("status") != "ready":
        raise ValueError(f"{record['name']} is {record.get('status')!r}, not ready.")
    if not record.get("preprocessing"):
        raise ValueError(
            f"{record['name']} was trained by an older GoTrain that did not save its "
            "preprocessing, so it cannot be test-evaluated reliably. Promote a newer version."
        )
    ready = [item for item in list_versions(store, dataset_id) if item.get("status") == "ready"]
    best = max(ready, key=lambda item: item["best_val_score"])
    previous = get_champion(store, dataset_id)
    event = {
        "version": record["version"],
        "promoted_at": _now(),
        "promoted_by": promoted_by,
        "reason": reason.strip(),
        "best_val_score": record["best_val_score"],
    }
    champion = {
        "dataset_id": dataset_id,
        "version": record["version"],
        "name": record["name"],
        "model": record["model"],
        "primary_metric": record["primary_metric"],
        "best_val_score": record["best_val_score"],
        "is_best_validation_version": record["version"] == best["version"],
        "best_validation_version": best["version"],
        "promoted_at": event["promoted_at"],
        "promoted_by": promoted_by,
        "reason": event["reason"],
        "history": (previous or {}).get("history", []) + [event],
    }
    store.write_json(_champion_key(dataset_id), champion)
    return champion


def download_version(store, dataset_id: str, version: int | str = "champion", destination=None) -> dict:
    """Download and checksum-verify a version's artifacts into a local directory."""
    record = get_version(store, dataset_id, version)
    if record.get("status") != "ready":
        raise ValueError(f"{record['name']} is {record.get('status')!r}, not ready.")
    target = Path(destination) if destination else cache_dir() / "models" / dataset_id / f"v{record['version']:04d}"
    prefix = _version_prefix(dataset_id, record["version"])
    paths = {}
    for name, expected in record["artifacts"].items():
        path = target / name
        if not (path.is_file() and sha256_file(path) == expected["sha256"]):
            store.download_file(f"{prefix}/{name}", path)
            actual = sha256_file(path)
            if actual != expected["sha256"]:
                path.unlink()
                raise ValueError(f"Checksum mismatch for {record['name']}/{name}.")
        paths[name] = str(path)
    return {"name": record["name"], "version": record["version"], "directory": str(target), "files": paths}


def run_final_test(
    store,
    dataset_id: str,
    evaluate: Callable[[str], dict[str, Any]],
    *,
    retry_unfinished: bool = False,
) -> dict:
    """Evaluate the champion on the test split exactly once and record the result.

    ``evaluate`` receives the local checkpoint path and returns test metrics.
    A completed evaluation can never be repeated. An attempt that crashed
    before producing metrics may be retried only with ``retry_unfinished``.
    """
    champion = get_champion(store, dataset_id)
    if champion is None:
        raise ValueError(f"{dataset_id} has no champion; promote the final model first.")
    claim = {
        "dataset_id": dataset_id,
        "status": "in_progress",
        "split": "test",
        "version": champion["version"],
        "name": champion["name"],
        "started_at": _now(),
    }
    key = _test_key(dataset_id)
    if not store.create_json(key, claim):
        existing = store.read_json(key) or {}
        if existing.get("status") == "completed":
            raise ValueError(
                f"{dataset_id} was already evaluated on the test split "
                f"({existing['name']}, {existing['completed_at']}). The test split is used once."
            )
        if not retry_unfinished:
            raise ValueError(
                f"A previous test evaluation of {dataset_id} is {existing.get('status')!r} without "
                "results. Re-run with --retry-unfinished only if it crashed before reporting metrics."
            )
        store.write_json(key, claim)
    try:
        local = download_version(store, dataset_id, champion["version"])
        evaluation = evaluate(local["files"]["best.pt"])
    except Exception as exc:
        store.write_json(key, {**claim, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    record = {
        **claim,
        "status": "completed",
        "completed_at": _now(),
        "model": champion["model"],
        "primary_metric": champion["primary_metric"],
        "best_val_score": champion["best_val_score"],
        "evaluation": evaluation,
        "code": _code_version(),
    }
    store.write_json(key, record)
    return record


def summarize(store) -> list[dict]:
    rows = []
    for dataset_id in list_model_datasets(store):
        versions = list_versions(store, dataset_id)
        ready = [item for item in versions if item.get("status") == "ready"]
        best = max(ready, key=lambda item: item["best_val_score"]) if ready else None
        champion = get_champion(store, dataset_id)
        test = get_test_evaluation(store, dataset_id)
        rows.append({
            "dataset_id": dataset_id,
            "versions": len(versions),
            "best_validation_version": best and best["name"],
            "best_val_score": best and best["best_val_score"],
            "champion": champion and champion["name"],
            "champion_model": champion and champion["model"],
            "champion_val_score": champion and champion["best_val_score"],
            "primary_metric": (champion or best or {}).get("primary_metric"),
            "test_status": test and test["status"],
            "test_score": test and test.get("evaluation", {}).get("test_score"),
        })
    return rows
