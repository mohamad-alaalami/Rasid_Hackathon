import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.campaign import CampaignRunner
from agent.experiments import load_campaign_experiments
from agent.feedback import build_campaign_report


def configuration(model="resnet18_rgb", learning_rate=0.0003):
    return {
        "architecture": model,
        "learning_rate": learning_rate,
        "epochs": 2,
        "batch_size": 2,
        "weight_decay": 0.0001,
        "pretrained": True,
        "image_size": None,
    }


class CampaignRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.patch = patch("agent.experiments.RESULTS_DIR", Path(self.temporary.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def executor(dataset_id, config, seed):
        score = {0.0003: 0.71, 0.001: 0.78, 0.0001: 0.74}[config["learning_rate"]]
        return {
            "status": "completed",
            "evaluation_split": "validation",
            "validation_score": score,
            "primary_metric": "accuracy",
            "train_loss": [1.0, 0.5],
            "val_loss": [0.9, 0.6],
            "validation_metric_curve": [score - 0.1, score],
            "higher_is_better": True,
            "detailed_diagnostics": {"learning_rate_history": [config["learning_rate"]] * 2},
            "checkpoint_path": f"experiments/{score}/best.pt",
        }

    def runner(self, version="E3", max_experiments=3, executor=None):
        return CampaignRunner(
            version=version,
            campaign_id="campaign-1",
            seed=42,
            max_experiments=max_experiments,
            executor=executor or self.executor,
        )

    def test_complete_multi_experiment_campaign_and_best_checkpoint(self):
        runner = self.runner()
        for learning_rate in (0.0003, 0.001, 0.0001):
            self.assertEqual(
                runner.run("m-eurosat", configuration(learning_rate=learning_rate))["status"],
                "completed",
            )
        report = runner.report()
        self.assertEqual(len(report["experiments"]), 3)
        self.assertEqual(report["best_validation_score"], 0.78)
        self.assertEqual(report["best_checkpoint_path"], "experiments/0.78/best.pt")

    def test_budget_exhaustion_and_duplicate_prevention(self):
        runner = self.runner(max_experiments=1)
        first = runner.run("m-eurosat", configuration())
        duplicate = runner.run("m-eurosat", configuration())
        other = runner.run("m-eurosat", configuration(learning_rate=0.001))
        self.assertEqual(first["status"], "completed")
        self.assertEqual(duplicate["error_code"], "DUPLICATE_CONFIGURATION")
        self.assertEqual(other["error_code"], "BUDGET_EXHAUSTED")

    def test_failed_training_is_persisted_without_fake_score(self):
        def failing_executor(dataset_id, config, seed):
            raise RuntimeError("CUDA out of memory")

        result = self.runner(executor=failing_executor).run("m-eurosat", configuration())
        records = load_campaign_experiments("E3", "campaign-1", 42)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(records[0]["status"], "failed")
        self.assertIsNone(records[0]["validation_score"])
        self.assertIn("CUDA out of memory", records[0]["error"])

    def test_persistence_across_runner_instances_and_dataset_lock(self):
        self.runner().run("m-eurosat", configuration())
        resumed = self.runner()
        self.assertEqual(len(resumed.report()["experiments"]), 1)
        rejected = resumed.run("m-pv4ger", configuration(learning_rate=0.001))
        self.assertEqual(rejected["error_code"], "CAMPAIGN_DATASET_LOCKED")

    def test_feedback_isolation_uses_real_record_fields(self):
        e0 = self.runner(version="E0").run("m-eurosat", configuration())["feedback"]
        self.assertNotIn("training_diagnostics", e0)
        self.assertNotIn("experiment_memory", e0)
        self.assertEqual(e0["primary_metric"], "accuracy")

        e1 = self.runner(version="E1").run("m-eurosat", configuration())["feedback"]
        self.assertIn("training_diagnostics", e1)
        self.assertNotIn("experiment_memory", e1)

        e2_runner = self.runner(version="E2")
        e2_runner.run("m-eurosat", configuration())
        e2 = e2_runner.run(
            "m-eurosat", configuration(learning_rate=0.001)
        )["feedback"]
        self.assertEqual(len(e2["experiment_memory"]), 1)

        e3 = self.runner(version="E3").run("m-eurosat", configuration())["feedback"]
        self.assertEqual(
            e3["detailed_diagnostics_available"], ["learning_rate_history"]
        )

    def test_campaign_report_respects_feedback_isolation(self):
        forbidden_curves = {
            "train_loss", "val_loss", "validation_metric_curve",
            "detailed_diagnostics",
        }

        for version in ("E0", "E1", "E2", "E3"):
            with self.subTest(version=version):
                records = [
                    {
                        "experiment_id": 1,
                        "dataset": "m-eurosat",
                        "configuration": configuration(),
                        "status": "completed",
                        "primary_metric": "accuracy",
                        "validation_score": 0.7,
                        "train_loss": [1.0, 0.5],
                        "val_loss": [0.9, 0.6],
                        "validation_metric_curve": [0.6, 0.7],
                        "detailed_diagnostics": {
                            "learning_rate_history": [0.0003, 0.0003]
                        },
                    },
                    {
                        "experiment_id": 2,
                        "dataset": "m-eurosat",
                        "configuration": configuration(learning_rate=0.001),
                        "status": "completed",
                        "primary_metric": "accuracy",
                        "validation_score": 0.8,
                        "train_loss": [0.8, 0.4],
                        "val_loss": [0.7, 0.5],
                        "validation_metric_curve": [0.7, 0.8],
                        "detailed_diagnostics": {
                            "learning_rate_history": [0.001, 0.001]
                        },
                    },
                ]
                report = build_campaign_report(
                    version, "campaign-1", records, records[-1]
                )
                experiment = report["experiments"][-1]

                self.assertTrue(forbidden_curves.isdisjoint(experiment))
                if version == "E0":
                    self.assertNotIn("training_diagnostics", experiment)
                    self.assertNotIn(
                        "detailed_diagnostics_available", experiment
                    )
                else:
                    self.assertIn("training_diagnostics", experiment)

                if version in {"E0", "E1"}:
                    self.assertEqual(len(report["experiments"]), 1)
                    self.assertNotIn("best_validation_score", report)
                    self.assertNotIn("completed", report)
                else:
                    self.assertEqual(len(report["experiments"]), 2)
                    self.assertEqual(report["best_validation_score"], 0.8)
                    self.assertEqual(report["completed"], 2)

                if version == "E3":
                    self.assertEqual(
                        experiment["detailed_diagnostics_available"],
                        ["learning_rate_history"],
                    )
                else:
                    self.assertNotIn(
                        "detailed_diagnostics_available", experiment
                    )


if __name__ == "__main__":
    unittest.main()
