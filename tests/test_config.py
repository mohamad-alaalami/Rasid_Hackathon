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


if __name__ == "__main__":
    unittest.main()

