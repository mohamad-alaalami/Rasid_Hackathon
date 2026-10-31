import io
import json
import os
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage.backends import LocalStore, S3Store, get_store
from storage.datasets import _safe_extract, pull_dataset, push_dataset, sync_dataset
from storage.logs import push_logs
from storage.model_versions import (
    download_version,
    get_champion,
    list_versions,
    promote_version,
    publish_experiment,
    register_experiment,
    run_final_test,
    summarize,
)


def make_experiment(root, experiment_id, score, dataset_id="m-eurosat"):
    directory = Path(root) / experiment_id
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(f"weights-{experiment_id}".encode())
    (directory / "config.json").write_text(json.dumps({"dataset_id": dataset_id, "epochs": 2}))
    (directory / "metrics.json").write_text(json.dumps([{"epoch": 1, "val_accuracy": score}]))
    result = {
        "status": "completed",
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "model": "resnet18_rgb",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "best_val_score": score,
    }
    (directory / "result.json").write_text(json.dumps(result))
    return result, directory


class StoreConfigurationTests(unittest.TestCase):
    def test_store_is_disabled_without_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_store())
        self.assertIsNone(get_store("  "))

    def test_store_uri_parsing(self):
        s3 = get_store("s3://bucket/team/gotrain/")
        self.assertIsInstance(s3, S3Store)
        self.assertEqual((s3.bucket, s3.prefix), ("bucket", "team/gotrain"))
        self.assertEqual(s3.describe("models/x.json"), "s3://bucket/team/gotrain/models/x.json")
        self.assertIsInstance(get_store("/tmp/store"), LocalStore)
        with self.assertRaises(ValueError):
            get_store("gs://bucket")

    def test_keys_cannot_escape_the_store(self):
        store = LocalStore(tempfile.mkdtemp())
        for key in ("../x", "/abs", "a/../../b", ""):
            with self.assertRaises(ValueError):
                store.write_json(key, {})

    def test_create_json_is_exclusive(self):
        store = LocalStore(tempfile.mkdtemp())
        self.assertTrue(store.create_json("a/b.json", {"n": 1}))
        self.assertFalse(store.create_json("a/b.json", {"n": 2}))
        self.assertEqual(store.read_json("a/b.json"), {"n": 1})
        self.assertEqual(store.list_keys("a"), ["a/b.json"])


class ModelVersionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = LocalStore(root / "store")
        self.experiments = root / "experiments"
        self.cache = patch.dict(os.environ, {"GOTRAIN_CACHE_DIR": str(root / "cache")})
        self.cache.start()

    def tearDown(self):
        self.cache.stop()
        self.temporary.cleanup()

    def register(self, experiment_id, score, dataset_id="m-eurosat"):
        result, directory = make_experiment(self.experiments, experiment_id, score, dataset_id)
        return register_experiment(self.store, result, directory)

    def test_versions_are_numbered_per_dataset_with_provenance(self):
        first = self.register("exp_a", 0.80)
        second = self.register("exp_b", 0.85)
        other = self.register("exp_c", 0.60, dataset_id="m-pv4ger")
        self.assertEqual((first["name"], second["name"]), ("m-eurosat/v0001", "m-eurosat/v0002"))
        self.assertEqual(other["name"], "m-pv4ger/v0001")
        self.assertEqual(second["status"], "ready")
        self.assertEqual(second["experiment_id"], "exp_b")
        self.assertEqual(second["config"]["epochs"], 2)
        self.assertEqual(len(second["artifacts"]["best.pt"]["sha256"]), 64)
        self.assertTrue(self.store.exists("models/m-eurosat/v0002/best.pt"))
        self.assertEqual([v["version"] for v in list_versions(self.store, "m-eurosat")], [1, 2])

    def test_failed_runs_never_become_versions(self):
        with self.assertRaises(ValueError):
            register_experiment(self.store, {"status": "failed", "dataset_id": "m-eurosat"}, ".")

    def test_publish_is_a_noop_without_store_and_never_raises(self):
        result, directory = make_experiment(self.experiments, "exp_a", 0.8)
        with patch.dict(os.environ, {"GOTRAIN_STORE": ""}):
            self.assertEqual(publish_experiment(result, directory), {})
        with patch.dict(os.environ, {"GOTRAIN_STORE": str(self.store.root)}):
            published = publish_experiment(result, directory)
        self.assertEqual(published["model_version"]["name"], "m-eurosat/v0001")
        (directory / "best.pt").unlink()
        with patch.dict(os.environ, {"GOTRAIN_STORE": str(self.store.root)}):
            failed = publish_experiment(result, directory)
        self.assertIn("model_version_error", failed)

    def test_promotion_download_and_one_time_final_test(self):
        self.register("exp_a", 0.80)
        self.register("exp_b", 0.85)
        with self.assertRaises(ValueError):
            run_final_test(self.store, "m-eurosat", lambda path: {})
        champion = promote_version(self.store, "m-eurosat", 2, "best validation accuracy")
        self.assertTrue(champion["is_best_validation_version"])
        local = download_version(self.store, "m-eurosat")
        self.assertEqual(Path(local["files"]["best.pt"]).read_bytes(), b"weights-exp_b")

        seen = []
        record = run_final_test(
            self.store, "m-eurosat", lambda path: seen.append(path) or {"test_score": 0.83}
        )
        self.assertEqual((record["status"], record["name"]), ("completed", "m-eurosat/v0002"))
        self.assertEqual(len(seen), 1)
        with self.assertRaisesRegex(ValueError, "already evaluated"):
            run_final_test(self.store, "m-eurosat", lambda path: {}, retry_unfinished=True)
        with self.assertRaisesRegex(ValueError, "locked"):
            promote_version(self.store, "m-eurosat", 1, "late change")
        self.assertEqual(summarize(self.store)[0]["test_score"], 0.83)

    def test_crashed_final_test_needs_explicit_retry(self):
        self.register("exp_a", 0.80)
        promote_version(self.store, "m-eurosat", 1, "only version")

        def crash(path):
            raise RuntimeError("CUDA error")

        with self.assertRaises(RuntimeError):
            run_final_test(self.store, "m-eurosat", crash)
        with self.assertRaisesRegex(ValueError, "retry-unfinished"):
            run_final_test(self.store, "m-eurosat", lambda path: {"test_score": 0.7})
        record = run_final_test(
            self.store, "m-eurosat", lambda path: {"test_score": 0.7}, retry_unfinished=True
        )
        self.assertEqual(record["status"], "completed")

    def test_download_rejects_tampered_artifacts(self):
        self.register("exp_a", 0.80)
        (self.store.root / "models/m-eurosat/v0001/best.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            download_version(self.store, "m-eurosat", 1)

    def test_promotion_history_is_kept(self):
        self.register("exp_a", 0.80)
        self.register("exp_b", 0.85)
        promote_version(self.store, "m-eurosat", 1, "first")
        champion = promote_version(self.store, "m-eurosat", "latest", "better")
        self.assertEqual(champion["version"], 2)
        self.assertEqual([event["version"] for event in champion["history"]], [1, 2])
        self.assertEqual(get_champion(self.store, "m-eurosat")["reason"], "better")


class DatasetSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = LocalStore(root / "store")
        self.source_root = root / "source"
        self.target_root = root / "target"
        dataset = self.source_root / "classification_v1.0" / "m-eurosat"
        (dataset / "nested").mkdir(parents=True)
        (dataset / "task_specs.pkl").write_bytes(pickle.dumps({"fake": True}))
        (dataset / "nested" / "sample.hdf5").write_bytes(b"pixels")

    def tearDown(self):
        self.temporary.cleanup()

    def test_push_then_pull_roundtrip_into_geo_bench_layout(self):
        pushed = push_dataset(self.store, "m-eurosat", geo_bench_dir=self.source_root)
        self.assertEqual(pushed["status"], "uploaded")
        self.assertEqual(pushed["manifest"]["file_count"], 2)
        self.assertEqual(
            push_dataset(self.store, "m-eurosat", geo_bench_dir=self.source_root)["status"],
            "already_stored",
        )
        pulled = pull_dataset(self.store, "m-eurosat", geo_bench_dir=self.target_root)
        self.assertEqual(pulled["status"], "downloaded")
        target = self.target_root / "classification_v1.0" / "m-eurosat"
        self.assertEqual((target / "nested" / "sample.hdf5").read_bytes(), b"pixels")
        self.assertEqual([p.name for p in target.parent.iterdir()], ["m-eurosat"])
        self.assertEqual(
            pull_dataset(self.store, "m-eurosat", geo_bench_dir=self.target_root)["status"],
            "already_local",
        )

    def test_sync_uploads_or_downloads_as_needed(self):
        self.assertEqual(
            sync_dataset(self.store, "m-eurosat", geo_bench_dir=self.source_root)["status"],
            "uploaded",
        )
        self.assertEqual(
            sync_dataset(self.store, "m-eurosat", geo_bench_dir=self.source_root)["status"],
            "local_and_stored",
        )
        self.assertEqual(
            sync_dataset(self.store, "m-eurosat", geo_bench_dir=self.target_root)["status"],
            "downloaded",
        )

    def test_pull_of_unknown_dataset_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            pull_dataset(self.store, "m-pv4ger", geo_bench_dir=self.target_root)

    def test_corrupted_archive_is_rejected_and_nothing_is_installed(self):
        push_dataset(self.store, "m-eurosat", geo_bench_dir=self.source_root)
        archive = self.store.root / "datasets/classification_v1.0/m-eurosat.tar"
        archive.write_bytes(archive.read_bytes()[:-10] + b"0123456789")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            pull_dataset(self.store, "m-eurosat", geo_bench_dir=self.target_root)
        self.assertEqual(list((self.target_root / "classification_v1.0").iterdir()), [])

    def test_archive_path_traversal_is_refused(self):
        archive = Path(self.temporary.name) / "evil.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            _safe_extract(archive, Path(self.temporary.name) / "out")


class LogBackupTests(unittest.TestCase):
    def test_logs_are_mirrored_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts/campaigns").mkdir(parents=True)
            (root / "artifacts/campaigns/E3.jsonl").write_text("{}\n")
            (root / "artifacts/campaigns/.experiments.lock").write_text("")
            make_experiment(root / "experiments", "exp_a", 0.8)
            store = LocalStore(root / "store")
            result = push_logs(store, root / "artifacts", root / "experiments")
            keys = store.list_keys("logs")
        self.assertEqual(result["files"], 4)
        self.assertIn("logs/artifacts/campaigns/E3.jsonl", keys)
        self.assertNotIn("logs/experiments/exp_a/best.pt", keys)


class S3StoreTests(unittest.TestCase):
    def setUp(self):
        try:
            import boto3
            from moto import mock_aws
        except ImportError:
            self.skipTest("boto3 and moto are not installed")
        self.mock = mock_aws()
        self.mock.start()
        os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")
        client = boto3.client("s3", region_name="eu-central-1")
        client.create_bucket(
            Bucket="gotrain-test",
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
        )
        self.store = S3Store("gotrain-test", "team", client=client)

    def tearDown(self):
        self.mock.stop()

    def test_s3_roundtrip_versions_and_exclusive_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, directory = make_experiment(temporary, "exp_a", 0.8)
            record = register_experiment(self.store, result, directory)
            self.assertEqual(record["uri"], "s3://gotrain-test/team/models/m-eurosat/v0001")
            self.assertTrue(self.store.exists("models/m-eurosat/v0001/best.pt"))
            self.assertFalse(self.store.create_json("models/m-eurosat/v0001/version.json", {}))
            self.assertIsNone(self.store.read_json("missing.json"))
            local = download_version(self.store, "m-eurosat", 1, Path(temporary) / "dl")
            self.assertEqual(Path(local["files"]["best.pt"]).read_bytes(), b"weights-exp_a")


if __name__ == "__main__":
    unittest.main()
