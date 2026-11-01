"""Versioned, immutable solution specifications and their execution records.

A SolutionSpec is the complete, reproducible definition of one candidate ML
solution: dataset contract, input channels and normalization policy, model and
pretrained-weight provenance, pipeline components, training settings, and
seed. Its ID is the SHA-256 of its canonical JSON, so the same specification
always has the same ID and a stored spec can never change.

Results arrive separately as append-only events:

    <solutions_dir>/<solution_id>/spec.json      immutable identity
    <solutions_dir>/<solution_id>/events.jsonl   "proposed" (lineage, autonomy)
                                                 and "executed" (results) events

Lineage (parent solution, change description) is an event, not part of the
identity: one solution can be reached from several parents, which is the edge
structure a future solution graph needs. ``initialization`` and
``custom_components`` are reserved for warm starts and sandboxed code (scope
S2); both must stay empty until those milestones exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA = "gotrain.solution/1"
DEFAULT_DIR = "artifacts/solutions"
ENVIRONMENT_PACKAGES = ("torch", "torchvision", "segmentation-models-pytorch", "numpy", "geobench")


def solutions_dir() -> Path:
    return Path(os.environ.get("GOTRAIN_SOLUTIONS_DIR", DEFAULT_DIR))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_contract_sha256(dataset_spec) -> str:
    return _sha256_text(canonical_json(asdict(dataset_spec)))


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

def build_solution_spec(dataset_id: str, cfg) -> dict[str, Any]:
    """The immutable identity of a training configuration for one dataset."""
    from .components import selected_view
    from .model_registry import get_model_spec
    from .registry import get_dataset_spec

    dataset = get_dataset_spec(dataset_id)
    model = get_model_spec(cfg.model)
    pipeline = cfg.resolved_pipeline()
    view = selected_view(pipeline, model)
    indices = dataset.profile_band_indices(view)
    normalization = (
        "train_channel_stats"
        if "train_channel_stats" in (pipeline["normalization"], dataset.normalization)
        else dataset.normalization
    )
    return {
        "schema": SCHEMA,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "benchmark_name": dataset.benchmark_name,
            "task_type": dataset.task_type,
            "num_classes": dataset.num_classes,
            "registry_contract_sha256": registry_contract_sha256(dataset),
        },
        "inputs": {
            "input_view": view,
            "band_indices": list(indices),
            "band_names": dataset.channel_names(indices),
            "normalization": normalization,
            "native_resolution": list(dataset.native_resolution),
            "image_size": cfg.image_size,
        },
        "model": {
            "model_id": model.model_id,
            "architecture": model.architecture,
            "pretrained": cfg.pretrained,
            "pretraining_source": model.pretraining_source if cfg.pretrained else None,
            "checkpoint_id": model.checkpoint_id if cfg.pretrained else None,
            "checkpoint_url": model.checkpoint_url if cfg.pretrained else None,
            "input_adaptation": model.preprocessing,
        },
        "pipeline": {
            key: pipeline[key]
            for key in ("augmentations", "loss", "optimizer", "scheduler", "dropout")
        },
        "training": {
            key: getattr(cfg, key)
            for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed",
                        "num_workers", "mixed_precision", "early_stopping",
                        "early_stopping_patience", "early_stopping_min_delta")
        },
        "initialization": {"warm_start_from_solution": None, "warm_start_checkpoint": None},
        "custom_components": [],
    }


def solution_id(spec: dict[str, Any]) -> str:
    return "sol_" + _sha256_text(canonical_json(spec))[:20]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _paths(sid: str) -> tuple[Path, Path]:
    if not sid.startswith("sol_") or not sid[4:].isalnum():
        raise ValueError(f"Invalid solution id {sid!r}.")
    directory = solutions_dir() / sid
    return directory / "spec.json", directory / "events.jsonl"


def solution_exists(sid: str) -> bool:
    try:
        return _paths(sid)[0].is_file()
    except ValueError:
        return False


def _append_event(sid: str, event: dict[str, Any]) -> None:
    _, events = _paths(sid)
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def propose_solution(spec: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    """Store the spec (once, immutably) and record who proposed it and why."""
    context = dict(context or {})
    if spec.get("custom_components"):
        raise ValueError("custom_components are reserved for scope S2 and must be empty.")
    if any(spec.get("initialization", {}).values()):
        raise ValueError("Warm starts are reserved for a later milestone; initialization must be empty.")
    parent = context.get("parent_solution_id")
    if parent is not None and not solution_exists(parent):
        raise ValueError(f"Unknown parent solution {parent!r}.")
    sid = solution_id(spec)
    spec_path, _ = _paths(sid)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if spec_path.exists():
        if canonical_json(json.loads(spec_path.read_text(encoding="utf-8"))) != canonical_json(spec):
            raise ValueError(f"Stored spec for {sid} differs; solution specs are immutable.")
    else:
        temporary = spec_path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, spec_path)
    _append_event(sid, {
        "type": "proposed",
        "at": _now(),
        "solution_id": sid,
        "parent_solution_id": parent,
        "change_description": context.get("change_description"),
        "campaign_id": context.get("campaign_id"),
        "feedback_version": context.get("feedback_version"),
        "selection_policy": context.get("selection_policy"),
        "autonomy": context.get("autonomy"),
        "proposer": context.get("proposer", "direct"),
    })
    return sid


def record_run(sid: str, run: dict[str, Any]) -> None:
    _append_event(sid, {"type": "executed", "at": _now(), "solution_id": sid, **run})


def load_solution(sid: str) -> dict[str, Any]:
    spec_path, events_path = _paths(sid)
    if not spec_path.is_file():
        raise ValueError(f"Unknown solution {sid!r}.")
    events = []
    if events_path.is_file():
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    return {
        "solution_id": sid,
        "spec": json.loads(spec_path.read_text(encoding="utf-8")),
        "proposals": [event for event in events if event["type"] == "proposed"],
        "runs": [event for event in events if event["type"] == "executed"],
    }


def campaign_solution_ids(campaign_id: str) -> list[str]:
    """Solutions proposed in a campaign, in first-proposal order."""
    root = solutions_dir()
    found = []
    for directory in root.glob("sol_*") if root.is_dir() else ():
        events = directory / "events.jsonl"
        if not events.is_file():
            continue
        for line in events.read_text(encoding="utf-8").splitlines():
            event = json.loads(line) if line.strip() else {}
            if event.get("type") == "proposed" and event.get("campaign_id") == campaign_id:
                found.append((event["at"], directory.name))
                break
    return [sid for _, sid in sorted(found)]


# ---------------------------------------------------------------------------
# Execution evidence
# ---------------------------------------------------------------------------

def environment_info() -> dict[str, Any]:
    packages = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # torch missing or CUDA probe failed
        info["cuda_available"] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5, check=True).stdout.strip()
        info["git"] = {"commit": commit, "uncommitted_changes": bool(dirty)}
    except Exception:
        info["git"] = None
    return info


def dataset_fingerprint(task) -> dict[str, Any]:
    """Cheap identity of the local dataset files: task_specs.pkl hash plus names and sizes."""
    getter = getattr(task, "get_dataset_dir", None)
    try:
        directory = Path(getter()) if callable(getter) else None
    except Exception:
        directory = None
    if directory is None or not directory.is_dir():
        return {"status": "unavailable", "reason": "task does not expose a local dataset directory"}
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(directory).as_posix()}:{path.stat().st_size}\n".encode())
    specs = directory / "task_specs.pkl"
    if specs.is_file():
        digest.update(file_sha256(specs).encode())
    return {
        "status": "computed",
        "method": "sha256 of relative paths, sizes, and task_specs.pkl content",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }
