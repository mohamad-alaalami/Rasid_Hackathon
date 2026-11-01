import unittest

from ml.trainer import _update_early_stopping, train_model


class TrainerContractTests(unittest.TestCase):
    def test_early_stopping_patience_updates_only_for_meaningful_improvement(self):
        reference, waiting = _update_early_stopping(0.80, None, 0, 0.01)
        self.assertEqual((reference, waiting), (0.80, 0))
        reference, waiting = _update_early_stopping(0.805, reference, waiting, 0.01)
        self.assertEqual((reference, waiting), (0.80, 1))
        reference, waiting = _update_early_stopping(0.82, reference, waiting, 0.01)
        self.assertEqual((reference, waiting), (0.82, 0))
    def test_invalid_model_dataset_pair_returns_structured_error(self):
        result = train_model(
            "m-eurosat",
            {"model": "unet_r34", "input_mode": "registry", "epochs": 1},
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["dataset_id"], "m-eurosat")
        self.assertIn("experiment_id", result)
        self.assertIn("error_type", result)

    def test_unknown_model_returns_structured_error(self):
        result = train_model(
            "m-eurosat",
            {"model": "invented-model", "input_mode": "registry", "epochs": 1},
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unknown model", result["message"])


if __name__ == "__main__":
    unittest.main()
