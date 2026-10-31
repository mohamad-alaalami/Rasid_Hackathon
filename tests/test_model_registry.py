import unittest

from ml.model_registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_model_options,
    validate_model_for_dataset,
)


class ModelRegistryTests(unittest.TestCase):
    def test_six_requested_model_configurations_exist(self):
        requested = {
            "resnet18_rgb",
            "resnet18_s2",
            "resnet50_s2",
            "convnext_tiny_rgb",
            "unet_r34",
            "deeplabv3plus_r50",
        }
        self.assertTrue(requested.issubset(MODEL_REGISTRY))

    def test_invalid_task_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_model_for_dataset("unet_r34", "m-eurosat")

    def test_checkpoint_provenance_is_exact(self):
        moco = get_model_spec("resnet18_s2")
        dino = get_model_spec("resnet50_s2")
        self.assertEqual(moco.checkpoint_id, "B13_rn18_moco_0099_ckpt.pth")
        self.assertEqual(dino.checkpoint_id, "B13_rn50_dino_0099.pth")
        self.assertIn("drive.google.com", moco.checkpoint_url)

    def test_options_include_runtime_eligibility(self):
        options = list_model_options("m-eurosat")
        self.assertTrue(options)
        self.assertTrue(all("eligible" in option for option in options))


if __name__ == "__main__":
    unittest.main()
