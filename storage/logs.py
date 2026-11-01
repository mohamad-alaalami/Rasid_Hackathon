"""Back up agent campaign logs and experiment records to the store."""

from __future__ import annotations

from pathlib import Path

LOGS_PREFIX = "logs"
EXPERIMENT_LOG_SUFFIXES = {".json", ".log"}


def push_logs(store, artifacts_dir: str | Path = "artifacts", experiments_dir: str | Path = "experiments") -> dict:
    """Mirror campaign artifacts and per-experiment JSON/log files (not checkpoints)."""
    uploaded = []
    sources = (
        (Path(artifacts_dir), lambda path: not path.name.startswith(".")),
        (Path(experiments_dir), lambda path: path.suffix in EXPERIMENT_LOG_SUFFIXES),
    )
    for root, include in sources:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and include(path):
                key = f"{LOGS_PREFIX}/{root.name}/{path.relative_to(root).as_posix()}"
                store.upload_file(path, key)
                uploaded.append(key)
    return {"status": "uploaded", "files": len(uploaded), "uri": store.describe(LOGS_PREFIX)}
