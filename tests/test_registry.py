import unittest

from ml.dataset import resolve_effective_resolution
from ml.registry import get_dataset_spec, list_dataset_specs


class DatasetRegistryTests(unittest.TestCase):
    def test_all_eight_datasets_are_registered(self):
        self.assertEqual(
            {item["dataset_id"] for item in list_dataset_specs()},
            {"m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake",
             "m-forestnet", "m-brick-kiln", "m-nz-cattle", "m-cashew-plant"},
        )

    def test_task_specific_defaults(self):
        self.assertEqual(get_dataset_spec("m-eurosat").task_type, "classification")
        self.assertEqual(get_dataset_spec("m-eurosat").available_channels, 13)
        self.assertEqual(get_dataset_spec("m-eurosat").input_channels, 3)
        self.assertEqual(get_dataset_spec("m-chesapeake").task_type, "segmentation")
        self.assertEqual(get_dataset_spec("m-chesapeake").input_channels, 4)
        self.assertEqual(get_dataset_spec("m-chesapeake").num_classes, 7)

    def test_verified_native_resolutions(self):
        self.assertEqual(get_dataset_spec("m-eurosat").native_resolution, (64, 64))
        self.assertEqual(get_dataset_spec("m-pv4ger").native_resolution, (320, 320))
        self.assertEqual(get_dataset_spec("m-pv4ger-seg").native_resolution, (320, 320))
        self.assertEqual(get_dataset_spec("m-chesapeake").native_resolution, (256, 256))

    def test_pv4ger_defaults_to_native_320_not_64(self):
        spec = get_dataset_spec("m-pv4ger")
        self.assertEqual(resolve_effective_resolution(spec, None), (320, 320))
        self.assertNotEqual(resolve_effective_resolution(spec, None), (64, 64))

    def test_explicit_resolution_override_is_respected(self):
        spec = get_dataset_spec("m-pv4ger")
        self.assertEqual(resolve_effective_resolution(spec, 128), (128, 128))

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered datasets"):
            get_dataset_spec("unknown")


if __name__ == "__main__":
    unittest.main()
