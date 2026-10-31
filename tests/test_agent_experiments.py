import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.experiments import finalize_training, reserve_training
from agent.feedback import build_feedback


class AgentExperimentControlTests(unittest.TestCase):
    def setUp(self):
        test_tmp = os.environ.get("GOTRAIN_TEST_TMP")
        self.temporary = tempfile.TemporaryDirectory(dir=test_tmp)
        self.results_patch = patch(
            "agent.experiments.RESULTS_DIR", Path(self.temporary.name)
        )
        self.results_patch.start()
        self.configuration = {
            "architecture": "resnet18_rgb",
            "learning_rate": 0.0003,
            "epochs": 1,
            "batch_size": 2,
        }

    def tearDown(self):
        self.results_patch.stop()
        self.temporary.cleanup()

    def test_duplicate_configuration_is_rejected_before_training(self):
        first = reserve_training(
            "E2", "campaign", "m-eurosat", 42, self.configuration
        )
        duplicate = reserve_training(
            "E2", "campaign", "m-eurosat", 42, self.configuration
        )
        self.assertEqual(first["status"], "reserved")
        self.assertEqual(duplicate["error_code"], "DUPLICATE_CONFIGURATION")

    def test_validation_result_can_finalize_reservation(self):
        reservation = reserve_training(
            "E1", "campaign", "m-eurosat", 42, self.configuration
        )
        result = finalize_training(
            "E1",
            "campaign",
            "m-eurosat",
            42,
            reservation["experiment_id"],
            {
                "status": "completed",
                "evaluation_split": "validation",
                "validation_score": 0.8,
                "runtime_seconds": 1.0,
            },
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["validation_score"], 0.8)

    def test_feedback_levels_do_not_leak_history_to_e0(self):
        experiment = {
            "status": "completed",
            "evaluation_split": "validation",
            "configuration": self.configuration,
            "validation_score": 0.8,
        }
        feedback = build_feedback("E0", experiment, remaining_budget=1, history=[{}])
        self.assertNotIn("experiment_memory", feedback)
        self.assertNotIn("training_diagnostics", feedback)


if __name__ == "__main__":
    unittest.main()
