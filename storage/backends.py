"""Storage backends for GoTrain datasets, model versions, and logs.

The backend is selected by $GOTRAIN_STORE:

- ``s3://bucket/prefix``: Amazon S3 (requires boto3, preinstalled on SageMaker)
- ``/any/directory``: a local or shared file system

When $GOTRAIN_STORE is unset, storage is disabled and GoTrain behaves exactly
as it did before the store existed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

STORE_ENV = "GOTRAIN_STORE"
CACHE_ENV = "GOTRAIN_CACHE_DIR"


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV) or "~/.cache/gotrain").expanduser()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_key(key: str) -> str:
    parts = PurePosixPath(key).parts
    if not key or key.startswith("/") or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid storage key: {key!r}")
    return "/".join(parts)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class LocalStore:
    """Directory-backed store; also suitable for EFS/FSx shared file systems."""

    kind = "local"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @property
    def uri(self) -> str:
        return str(self.root)

    def describe(self, key: str) -> str:
        return str(self._path(key))

    def _path(self, key: str) -> Path:
        return self.root / _clean_key(key)

    def _staging(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")

    def upload_file(self, local_path: str | Path, key: str) -> None:
        target = self._path(key)
        staging = self._staging(target)
        shutil.copyfile(local_path, staging)
        os.replace(staging, target)

    def download_file(self, key: str, local_path: str | Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(f"Not found in store: {self.describe(key)}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_json(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        target = self._path(key)
        staging = self._staging(target)
        staging.write_bytes(_json_bytes(value))
        os.replace(staging, target)

    def create_json(self, key: str, value: Any) -> bool:
        """Write only if the key does not exist; return False if it already does."""
        target = self._path(key)
        staging = self._staging(target)
        staging.write_bytes(_json_bytes(value))
        try:
            os.link(staging, target)
            return True
        except FileExistsError:
            return False
        finally:
            staging.unlink(missing_ok=True)

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Store:
    """Amazon S3 store using the SageMaker instance's IAM role credentials."""

    kind = "s3"

    def __init__(self, bucket: str, prefix: str = "", client: Any = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("boto3 is required for s3:// stores.") from exc
            self._client = boto3.client("s3")
        return self._client

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}" if self.prefix else f"s3://{self.bucket}"

    def describe(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"

    def _key(self, key: str) -> str:
        key = _clean_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    @staticmethod
    def _error_code(exc: Exception) -> str:
        return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))

    def _is_missing(self, exc: Exception) -> bool:
        return self._error_code(exc) in {"404", "NoSuchKey", "NotFound"}

    def upload_file(self, local_path: str | Path, key: str) -> None:
        self.client.upload_file(str(local_path), self.bucket, self._key(key))

    def download_file(self, key: str, local_path: str | Path) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.client.download_file(self.bucket, self._key(key), str(staging))
        except Exception as exc:
            staging.unlink(missing_ok=True)
            if self._is_missing(exc):
                raise FileNotFoundError(f"Not found in store: {self.describe(key)}") from exc
            raise
        os.replace(staging, target)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as exc:
            if self._is_missing(exc):
                return False
            raise

    def read_json(self, key: str) -> Any | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            if self._is_missing(exc):
                return None
            raise
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=_json_bytes(value),
            ContentType="application/json",
        )

    def create_json(self, key: str, value: Any) -> bool:
        """Conditional create (S3 If-None-Match); return False if the key exists."""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self._key(key),
                Body=_json_bytes(value),
                ContentType="application/json",
                IfNoneMatch="*",
            )
            return True
        except Exception as exc:
            if self._error_code(exc) in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                return False
            if type(exc).__name__ != "ParamValidationError":
                raise
        # botocore predates conditional writes: fall back to check-then-write.
        if self.exists(key):
            return False
        self.write_json(key, value)
        return True

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._key(prefix).rstrip("/") + "/"
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            keys.extend(item["Key"][strip:] for item in page.get("Contents", []))
        return sorted(keys)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


def get_store(uri: str | None = None):
    """Return the configured store, or None when storage is disabled."""
    uri = (uri if uri is not None else os.environ.get(STORE_ENV, "")).strip()
    if not uri:
        return None
    if uri.startswith("s3://"):
        bucket, _, prefix = uri[len("s3://"):].partition("/")
        if not bucket:
            raise ValueError(f"Invalid S3 store URI: {uri!r}")
        return S3Store(bucket, prefix)
    if "://" in uri:
        raise ValueError(f"Unsupported store URI {uri!r}; use s3://bucket/prefix or a directory.")
    return LocalStore(uri)
