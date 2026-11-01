"""Strands tools for GoTrain's dataset and model-version store.

The tools exist only when $GOTRAIN_STORE is set; otherwise campaigns get no
extra tools and an unchanged prompt. Model-version tools reveal results from
earlier runs, so only feedback modes that already see history (E2, E3) get
them; E0/E1 keep their no-history contract. No tool here can reach the test
split: final test evaluation stays a human-run command.
"""

import json
from pathlib import Path

from strands import tool

from storage import get_store
from storage.datasets import list_stored_datasets, sync_dataset as sync_store_dataset
from storage.model_versions import (
    download_version,
    find_version_for_experiment,
    get_champion,
    get_test_evaluation,
    get_version,
    list_versions,
    promote_version,
    register_experiment,
)

HISTORY_MODES = {"E2", "E3"}
EXPERIMENTS_DIR = Path("experiments")
CONFIG_SUMMARY_FIELDS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "pretrained",
    "image_size",
    "early_stopping",
)


def _call(action) -> dict:
    try:
        return {"status": "success", **action()}
    except Exception as exc:
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


def _version_row(record: dict) -> dict:
    config = record.get("config") or {}
    return {
        "name": record.get("name"),
        "version": record.get("version"),
        "status": record.get("status"),
        "model": record.get("model"),
        "primary_metric": record.get("primary_metric"),
        "best_val_score": record.get("best_val_score"),
        "configuration": {field: config.get(field) for field in CONFIG_SUMMARY_FIELDS},
        "experiment_id": record.get("experiment_id"),
        "created_at": record.get("created_at"),
    }


def store_campaign_instructions(feedback_version: str, store=None) -> str:
    store = store or get_store()
    if store is None:
        return ""
    text = f"""
Model store: enabled at {store.uri}. Call sync_dataset for the chosen dataset
before inspecting it, so it is downloaded if missing and backed up if not stored.
Every completed experiment is saved automatically as a numbered model version.
"""
    if feedback_version in HISTORY_MODES:
        text += """Use list_model_versions to compare stored versions by validation score.
Before your final report, promote the version with the best validation evidence
using promote_model_version and state the reason. Never request test evaluation.
"""
    return text


def make_store_tools(feedback_version: str, store=None) -> list:
    """Return store tools for a campaign, or [] when storage is disabled."""
    store = store or get_store()
    if store is None:
        return []

    @tool
    def model_store_status() -> dict:
        """Show where GoTrain stores datasets, model versions, and logs, and which datasets are stored."""
        return _call(lambda: {
            "backend": store.kind,
            "uri": store.uri,
            "stored_datasets": [
                {key: item[key] for key in ("dataset_id", "benchmark_name", "archive_bytes", "uploaded_at")}
                for item in list_stored_datasets(store)
            ],
        })

    @tool
    def sync_dataset(dataset_id: str) -> dict:
        """Make a dataset available on this instance and backed up in the store.

        Downloads it from the store when it is missing locally, and uploads it when
        the store does not have it yet. Call this before inspecting a dataset.
        """
        return _call(lambda: sync_store_dataset(store, dataset_id))

    @tool
    def upload_experiment(experiment_id: str) -> dict:
        """Save a completed local experiment as a model version if its automatic upload failed."""
        def action():
            if not experiment_id or "/" in experiment_id or ".." in experiment_id:
                raise ValueError("Invalid experiment_id.")
            directory = EXPERIMENTS_DIR / experiment_id
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            existing = find_version_for_experiment(store, result["dataset_id"], experiment_id)
            if existing is not None:
                return {"model_version": existing["name"], "uploaded": False}
            record = register_experiment(store, result, directory, source="agent_upload")
            return {"model_version": record["name"], "uploaded": True}
        return _call(action)

    tools = [model_store_status, sync_dataset, upload_experiment]
    if feedback_version not in HISTORY_MODES:
        return tools

    @tool
    def list_model_versions(dataset_id: str, limit: int = 20) -> dict:
        """List a dataset's stored model versions (oldest first) with validation scores and the champion."""
        def action():
            if not 1 <= limit <= 100:
                raise ValueError("limit must be 1..100")
            champion = get_champion(store, dataset_id)
            return {
                "dataset_id": dataset_id,
                "champion": champion and champion["name"],
                "final_model_locked": get_test_evaluation(store, dataset_id) is not None,
                "versions": [_version_row(record) for record in list_versions(store, dataset_id)[-limit:]],
            }
        return _call(action)

    @tool
    def get_model_version(dataset_id: str, version: int) -> dict:
        """Return the full provenance record of one stored model version."""
        return _call(lambda: {"record": get_version(store, dataset_id, version)})

    @tool
    def promote_model_version(dataset_id: str, version: int, reason: str) -> dict:
        """Promote a stored version to champion, the dataset's final model.

        Choose it from validation evidence only and explain the reason. Promotion is
        refused once the human-run final test evaluation has locked the dataset.
        """
        return _call(lambda: {"champion": promote_version(store, dataset_id, version, reason)})

    @tool
    def download_model_version(dataset_id: str, version: int) -> dict:
        """Download a stored version's checkpoint and files and verify their checksums."""
        return _call(lambda: download_version(store, dataset_id, version))

    return tools + [list_model_versions, get_model_version, promote_model_version, download_model_version]
