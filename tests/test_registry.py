import unittest

from ml.registry import get_dataset_spec, list_dataset_specs


class DatasetRegistryTests(unittest.TestCase):
    def test_selected_four_datasets_are_registered(self):
        self.assertEqual(
            {item["dataset_id"] for item in list_dataset_specs()},
            {"m-eurosat", "m-pv4ger", "m-pv4ger-seg", "m-chesapeake"},
        )

    def test_task_specific_defaults(self):
        self.assertEqual(get_dataset_spec("m-eurosat").task_type, "classification")
        self.assertEqual(get_dataset_spec("m-eurosat").available_channels, 13)
        self.assertEqual(get_dataset_spec("m-eurosat").input_channels, 3)
        self.assertEqual(get_dataset_spec("m-chesapeake").task_type, "segmentation")
        self.assertEqual(get_dataset_spec("m-chesapeake").input_channels, 4)
        self.assertEqual(get_dataset_spec("m-chesapeake").num_classes, 7)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered datasets"):
            get_dataset_spec("unknown")


if __name__ == "__main__":
    unittest.main()
