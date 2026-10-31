"""Upload GEO-Bench datasets to the store and download them onto any instance.

A dataset is stored as one uncompressed tar archive plus a manifest written
last, so a manifest only exists for a fully uploaded archive. Downloads are
verified by SHA-256 and extracted into $GEO_BENCH_DIR, where GEO-Bench and the
trainer already look, so training code needs no changes.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .backends import sha256_file

DATASETS_PREFIX = "datasets"


def geo_bench_root(geo_bench_dir: str | Path | None = None) -> Path:
    """Mirror geobench.config: $GEO_BENCH_DIR, else ~/dataset/geobench."""
    root = geo_bench_dir or os.environ.get("GEO_BENCH_DIR") or "~/dataset/geobench"
    return Path(root).expanduser()


def _benchmark_name(dataset_id: str, benchmark_name: str | None) -> str:
    if benchmark_name:
        return benchmark_name
    from ml.registry import get_dataset_spec

    return get_dataset_spec(dataset_id).benchmark_name


def local_dataset_dir(dataset_id, benchmark_name=None, geo_bench_dir=None) -> Path:
    return geo_bench_root(geo_bench_dir) / _benchmark_name(dataset_id, benchmark_name) / dataset_id


def _keys(dataset_id: str, benchmark_name: str) -> tuple[str, str]:
    base = f"{DATASETS_PREFIX}/{benchmark_name}/{dataset_id}"
    return f"{base}.tar", f"{base}.manifest.json"


def _is_dataset_dir(path: Path) -> bool:
    return (path / "task_specs.pkl").is_file()


def get_stored_dataset(store, dataset_id: str, benchmark_name: str | None = None):
    _, manifest_key = _keys(dataset_id, _benchmark_name(dataset_id, benchmark_name))
    return store.read_json(manifest_key)


def list_stored_datasets(store) -> list[dict]:
    return [
        store.read_json(key)
        for key in store.list_keys(DATASETS_PREFIX)
        if key.endswith(".manifest.json")
    ]


def push_dataset(store, dataset_id, *, benchmark_name=None, geo_bench_dir=None, overwrite=False):
    """Archive a local GEO-Bench dataset directory and upload it to the store."""
    benchmark_name = _benchmark_name(dataset_id, benchmark_name)
    source = local_dataset_dir(dataset_id, benchmark_name, geo_bench_dir)
    if not _is_dataset_dir(source):
        raise FileNotFoundError(f"No GEO-Bench dataset (task_specs.pkl) found at {source}.")
    archive_key, manifest_key = _keys(dataset_id, benchmark_name)
    existing = store.read_json(manifest_key)
    if existing is not None and not overwrite:
        return {"status": "already_stored", "manifest": existing}

    files = [path for path in sorted(source.rglob("*")) if path.is_file()]
    staging_root = source.parent / f".gotrain-staging-{uuid.uuid4().hex}"
    staging_root.mkdir()
    try:
        archive = staging_root / f"{dataset_id}.tar"
        with tarfile.open(archive, "w") as tar:
            for path in files:
                tar.add(path, arcname=f"{dataset_id}/{path.relative_to(source).as_posix()}")
        manifest = {
            "dataset_id": dataset_id,
            "benchmark_name": benchmark_name,
            "archive_key": archive_key,
            "archive_sha256": sha256_file(archive),
            "archive_bytes": archive.stat().st_size,
            "file_count": len(files),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "uri": store.describe(archive_key),
        }
        store.upload_file(archive, archive_key)
        store.write_json(manifest_key, manifest)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {"status": "uploaded", "manifest": manifest}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Refusing unsupported archive entry: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Refusing archive entry outside destination: {member.name}")
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(destination, **options)


def pull_dataset(store, dataset_id, *, benchmark_name=None, geo_bench_dir=None):
    """Download a stored dataset into $GEO_BENCH_DIR unless it is already there."""
    benchmark_name = _benchmark_name(dataset_id, benchmark_name)
    destination = local_dataset_dir(dataset_id, benchmark_name, geo_bench_dir)
    if _is_dataset_dir(destination):
        return {"status": "already_local", "path": str(destination)}
    archive_key, manifest_key = _keys(dataset_id, benchmark_name)
    manifest = store.read_json(manifest_key)
    if manifest is None:
        raise FileNotFoundError(f"{dataset_id} is not in the store ({store.describe(manifest_key)}).")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # The dot prefix keeps GEO-Bench's task iterator from seeing partial data.
    staging_root = Path(tempfile.mkdtemp(prefix=".gotrain-staging-", dir=destination.parent))
    try:
        archive = staging_root / f"{dataset_id}.tar"
        store.download_file(archive_key, archive)
        actual = sha256_file(archive)
        if actual != manifest["archive_sha256"]:
            raise ValueError(f"Checksum mismatch for {dataset_id}: expected {manifest['archive_sha256']}, got {actual}.")
        extracted = staging_root / "extracted"
        _safe_extract(archive, extracted)
        if not _is_dataset_dir(extracted / dataset_id):
            raise ValueError(f"Archive for {dataset_id} does not contain task_specs.pkl.")
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(extracted / dataset_id, destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {
        "status": "downloaded",
        "path": str(destination),
        "archive_bytes": manifest["archive_bytes"],
        "file_count": manifest["file_count"],
    }


def sync_dataset(store, dataset_id, *, benchmark_name=None, geo_bench_dir=None):
    """Make the dataset both local and stored: download or upload as needed."""
    benchmark_name = _benchmark_name(dataset_id, benchmark_name)
    local = local_dataset_dir(dataset_id, benchmark_name, geo_bench_dir)
    if not _is_dataset_dir(local):
        return pull_dataset(store, dataset_id, benchmark_name=benchmark_name, geo_bench_dir=geo_bench_dir)
    pushed = push_dataset(store, dataset_id, benchmark_name=benchmark_name, geo_bench_dir=geo_bench_dir)
    status = "local_and_stored" if pushed["status"] == "already_stored" else "uploaded"
    return {"status": status, "path": str(local), "manifest": pushed["manifest"]}
