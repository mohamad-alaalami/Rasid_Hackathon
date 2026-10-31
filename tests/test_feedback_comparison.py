import json
import tempfile
import unittest
from pathlib import Path

from agent.feedback_comparison import (
    MODES,
    aggregate_comparison,
    summarize_trial,
    write_reports,
)


def record(experiment_id, score=None, status="completed", runtime=2.0):
    return {
        "experiment_id": experiment_id,
        "status": status,
        "validation_score": score,
        "runtime_seconds": runtime,
        "training_seconds": runtime,
        "checkpoint_path": f"exp/{experiment_id}/best.pt" if status == "completed" else None,
        "configuration": {"architecture": "resnet18_rgb", "learning_rate": 0.0003},
    }


def trial(mode, seed, scores=(0.7, 0.8), campaign_suffix=""):
    records = [record(index + 1, score) for index, score in enumerate(scores)]
    value = summarize_trial(
        mode=mode,
        seed=seed,
        campaign_id=f"campaign-{seed}-{mode}{campaign_suffix}",
        records=records,
        attempts=[],
        agent_seconds=5.0,
        decision_trace={"text": "mocked only in unit test"},
    )
    value["initial_configuration"] = {
        "architecture": "resnet18_rgb",
        "epochs": 2,
        "batch_size": 16,
    }
    return value


class FeedbackComparisonTests(unittest.TestCase):
    def metadata(self, seeds=(42,)):
        return {"seeds": list(seeds), "dataset": "m-pv4ger"}

    def test_fair_initialization_and_isolated_campaign_ids(self):
        trials = [trial(mode, 42) for mode in MODES]
        report = aggregate_comparison(trials, self.metadata())
        self.assertEqual(report["winner"], "E0")
        self.assertEqual(len({item["campaign_id"] for item in trials}), 4)

    def test_unmatched_initial_configuration_is_rejected(self):
        trials = [trial(mode, 42) for mode in MODES]
        trials[-1]["initial_configuration"]["epochs"] = 3
        with self.assertRaisesRegex(ValueError, "unmatched initial"):
            aggregate_comparison(trials, self.metadata())

    def test_metric_aggregation_mean_and_variation(self):
        trials = []
        for seed, e3_scores in ((42, (0.7, 0.9)), (43, (0.7, 0.8))):
            for mode in MODES:
                scores = e3_scores if mode == "E3" else (0.7, 0.75)
                trials.append(trial(mode, seed, scores))
        report = aggregate_comparison(trials, self.metadata((42, 43)))
        self.assertEqual(report["winner"], "E3")
        self.assertAlmostEqual(report["modes"]["E3"]["validation_improvement"]["mean"], 0.15)
        self.assertGreater(report["modes"]["E3"]["validation_improvement"]["stdev"], 0)

    def test_failure_and_rejected_attempt_accounting(self):
        records = [record(1, 0.7), record(2, None, status="failed")]
        attempts = [
            {"response": {"status": "rejected", "error_code": "DUPLICATE_CONFIGURATION"}},
            {"response": {"status": "rejected", "error_code": "INVALID_MODEL_CONFIGURATION"}},
        ]
        result = summarize_trial(
            mode="E0", seed=42, campaign_id="isolated", records=records,
            attempts=attempts, agent_seconds=3.0, decision_trace={},
        )
        self.assertEqual(result["failed_runs"], 1)
        self.assertEqual(result["duplicate_attempts"], 1)
        self.assertEqual(result["invalid_attempts"], 1)
        self.assertEqual(result["validation_improvement"], 0.0)

    def test_json_csv_and_markdown_reports_are_persisted(self):
        report = aggregate_comparison(
            [trial(mode, 42) for mode in MODES], self.metadata()
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_reports(report, Path(temporary))
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["winner"], report["winner"])


if __name__ == "__main__":
    unittest.main()
