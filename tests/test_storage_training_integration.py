"""End-to-end: real trainer -> model version -> promotion -> one-time test evaluation.

Uses tiny synthetic data in place of GEO-Bench so it runs on CPU in seconds.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    import torch
    import torchvision  # noqa: F401
except ImportError:
    torch = None


def fake_bundle(dataset_id, config):
    from ml.dataset import DataBundle

    generator = torch.Generator().manual_seed(0)
    batch = [(torch.randn(4, 3, 64, 64, generator=generator), torch.tensor([0, 1, 2, 3]))]
    # A real DataBundle, so the fake fails loudly if the loader contract changes.
    return DataBundle(
        train_loader=batch,
        val_loader=batch,
        task_type="classification",
        num_classes=10,
        input_channels=3,
        class_names=[],
        ignore_index=None,
        primary_metric="accuracy",
        native_resolution=(64, 64),
        effective_resolution=(64, 64),
        preprocessing={
            "band_indices": [3, 2, 1],
            "normalization": "imagenet_rgb",
            "effective_resolution": [64, 64],
        },
        rgb_positions=(0, 1, 2),  # B04, B03, B02 in the selected RGB view
        pipeline=config.resolved_pipeline(),
    )


def fake_task(dataset_id, benchmark_dir=None):
    rng = np.random.default_rng(0)

    def sample(label):
        bands = [SimpleNamespace(data=rng.uniform(0, 3000, (64, 64))) for _ in range(13)]
        return SimpleNamespace(bands=bands, label=label)

    splits_seen = []

    def get_dataset(split, band_names=None):
        splits_seen.append(split)
        return [sample(label) for label in (0, 1, 2)]

    fake_task.splits_seen = splits_seen
    return SimpleNamespace(get_dataset=get_dataset)


@unittest.skipIf(torch is None, "torch, torchvision, and numpy are required")
class StoreTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store_dir = root / "store"
        self.environment = patch.dict(os.environ, {
            "GOTRAIN_STORE": str(self.store_dir),
            "GOTRAIN_CACHE_DIR": str(root / "cache"),
            "GOTRAIN_SOLUTIONS_DIR": str(root / "solutions"),
        })
        self.environment.start()
        self.config = {
            "model": "resnet18_rgb",
            "pretrained": False,
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "output_dir": str(root / "experiments"),
        }

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def cli(self, *argv):
        from storage.__main__ import main

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(list(argv))
        return json.loads(output.getvalue())

    def test_training_versions_promotes_and_tests_once(self):
        from ml.trainer import train_model

        with patch("ml.dataset.create_dataloaders", fake_bundle):
            first = train_model("m-eurosat", self.config)
            second = train_model("m-eurosat", {**self.config, "learning_rate": 1e-3})
        self.assertEqual(first["status"], "completed", first.get("message"))
        self.assertEqual(first["model_version"]["name"], "m-eurosat/v0001")
        self.assertEqual(second["model_version"]["name"], "m-eurosat/v0002")
        saved = json.loads((Path(first["checkpoint_path"]).parent / "result.json").read_text())
        self.assertEqual(saved["model_version"]["version"], 1)

        versions = self.cli("versions", "m-eurosat")
        self.assertEqual([v["experiment_id"] for v in versions],
                         [first["experiment_id"], second["experiment_id"]])
        self.cli("promote", "m-eurosat", "2", "--reason", "test fixture")

        with patch("ml.dataset.find_task", fake_task):
            with self.assertRaises(SystemExit):
                self.cli("final-test", "m-eurosat")
            record = self.cli("final-test", "m-eurosat", "--yes", "--num-workers", "0")
        self.assertEqual(fake_task.splits_seen, ["test"])
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["name"], "m-eurosat/v0002")
        self.assertEqual(record["evaluation"]["num_samples"], 3)
        self.assertIn(record["evaluation"]["test_score"], {0.0, 1 / 3, 2 / 3, 1.0})
        summary = self.cli("summary")[0]
        self.assertEqual((summary["champion"], summary["test_status"]), ("m-eurosat/v0002", "completed"))

    def test_training_is_unchanged_without_store(self):
        from ml.trainer import train_model

        with patch.dict(os.environ, {"GOTRAIN_STORE": ""}):
            with patch("ml.dataset.create_dataloaders", fake_bundle):
                result = train_model("m-eurosat", self.config)
        self.assertEqual(result["status"], "completed", result.get("message"))
        self.assertNotIn("model_version", result)
        self.assertNotIn("model_version_error", result)
        self.assertFalse(self.store_dir.exists())


if __name__ == "__main__":
    unittest.main()
