import unittest

from ml.config import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def test_valid_minimal_configuration(self):
        config = TrainingConfig.from_mapping({"epochs": 1})
        self.assertEqual(config.model, "resnet18_rgb")
        self.assertEqual(config.epochs, 1)
        self.assertIsNone(config.image_size)

    def test_image_size_override_is_optional_and_explicit(self):
        config = TrainingConfig.from_mapping({"image_size": 128})
        self.assertEqual(config.image_size, 128)

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported configuration keys"):
            TrainingConfig.from_mapping({"magic": True})

    def test_invalid_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            TrainingConfig.from_mapping({"model": "unknown"})

    def test_model_dataset_compatibility(self):
        config = TrainingConfig.from_mapping({"model": "fcn_resnet50"})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            config.validate_for_dataset("m-eurosat")

    def test_string_epoch_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "epochs has invalid type"):
            TrainingConfig.from_mapping({"epochs": "5"})

    def test_boolean_epoch_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "epochs has invalid type"):
            TrainingConfig.from_mapping({"epochs": True})

    def test_early_stopping_configuration(self):
        config = TrainingConfig.from_mapping({
            "early_stopping": True,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.001,
        })
        self.assertTrue(config.early_stopping)
        self.assertEqual(config.early_stopping_patience, 2)
        self.assertEqual(config.early_stopping_min_delta, 0.001)

    def test_invalid_early_stopping_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "patience"):
            TrainingConfig.from_mapping({"early_stopping_patience": 0})
        with self.assertRaisesRegex(ValueError, "min_delta"):
            TrainingConfig.from_mapping({"early_stopping_min_delta": -0.1})


if __name__ == "__main__":
    unittest.main()

