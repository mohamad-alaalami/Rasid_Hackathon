"""Regression tests for all eight registered GEO-Bench datasets.

Fake GEO-Bench tasks mirror the SageMaker-verified contracts (channel count,
dtype, resolution, label encoding). The real loader, band selection,
normalization, mask validation, first-convolution adaptation, and input
padding run with real torch; no dataset, network, GPU, or test split is used.
"""

import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ml.config import TrainingConfig
from ml.dataset import GeoBenchDataset, compute_channel_statistics, create_dataloaders
from ml.model_registry import (
    MODEL_REGISTRY,
    compatible_models,
    get_model_spec,
    list_model_options,
    validate_model_for_dataset,
)
from ml.models import _adapt_first_convolution, apply_input_padding
from ml.registry import DATASET_REGISTRY, SSL4EO_S2_ORDER, get_dataset_spec, list_dataset_specs

# SageMaker-verified contracts (dtype per stored channel).
VERIFIED = {
    "m-eurosat": ("classification", 13, (64, 64), 10, "float32"),
    "m-pv4ger": ("classification", 3, (320, 320), 2, "uint8"),
    "m-pv4ger-seg": ("segmentation", 3, (320, 320), 2, "uint8"),
    "m-chesapeake": ("segmentation", 4, (256, 256), 7, "uint8"),
    "m-forestnet": ("classification", 6, (332, 332), 12, "uint8"),
    "m-brick-kiln": ("classification", 13, (64, 64), 2, "float64"),
    "m-nz-cattle": ("segmentation", 3, (500, 500), 2, "uint8"),
    "m-cashew-plant": ("segmentation", 13, (256, 256), 7, "float32"),
}


class FakeBand:
    def __init__(self, data, name=None):
        self.data = data
        self.band_info = types.SimpleNamespace(name=name)


class FakeSplit:
    """Deterministic samples: channel c is centred on 10*(c+1) so order is traceable."""

    def __init__(self, dataset_id, size, seed, mask_values=None):
        self.spec = get_dataset_spec(dataset_id)
        self.size, self.seed = size, seed
        self.mask_values = mask_values
        self.dtype = VERIFIED[dataset_id][4]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        spec = self.spec
        rng = np.random.default_rng(self.seed * 1000 + index)
        height, width = spec.native_resolution
        bands = []
        for channel in range(spec.available_channels):
            values = 10.0 * (channel + 1) + rng.normal(0.0, 2.0, (height, width))
            bands.append(FakeBand(np.clip(values, 0, 255).astype(self.dtype), f"stored_{channel}"))
        if spec.task_type == "classification":
            label = np.int64(index % spec.num_classes)
        else:
            choices = self.mask_values or list(range(spec.num_classes))
            label = FakeBand(rng.choice(choices, size=(height, width)).astype(np.int64))
        return types.SimpleNamespace(bands=bands, label=label)


class FakeTask:
    def __init__(self, dataset_id, mask_values=None):
        spec = get_dataset_spec(dataset_id)
        self.dataset_name = dataset_id
        self.label_type = types.SimpleNamespace(
            n_classes=spec.num_classes,
            class_names=[f"class_{i}" for i in range(spec.num_classes)],
        )
        self.dataset_id, self.mask_values = dataset_id, mask_values
        self.splits_requested = []

    def get_dataset(self, split, band_names=None):
        self.splits_requested.append(split)
        if split == "test":
            raise AssertionError("The sealed test split must never be read here.")
        if band_names is not None:
            raise AssertionError("Channels must be selected by index, never by stored name.")
        return FakeSplit(self.dataset_id, 4 if split == "train" else 2,
                         seed=1 if split == "train" else 2, mask_values=self.mask_values)


def loaders(dataset_id, model_id, task=None):
    task = task or FakeTask(dataset_id)
    config = TrainingConfig.from_mapping(
        {"model": model_id, "batch_size": 2, "num_workers": 0, "device": "cpu"}
    )
    with patch("ml.dataset.find_task", lambda *args: task):
        return create_dataloaders(dataset_id, config), task


class RegistryContractTests(unittest.TestCase):
    def test_every_dataset_matches_its_verified_contract(self):
        self.assertEqual(set(DATASET_REGISTRY), set(VERIFIED))
        for dataset_id, (task, channels, resolution, classes, dtype) in VERIFIED.items():
            spec = get_dataset_spec(dataset_id)
            with self.subTest(dataset_id):
                self.assertEqual(spec.task_type, task)
                self.assertEqual(spec.available_channels, channels)
                self.assertEqual(len(spec.band_names), channels)
                self.assertEqual(spec.native_resolution, resolution)
                self.assertEqual(spec.num_classes, classes)
                self.assertEqual(
                    spec.benchmark_name,
                    "classification_v1.0" if task == "classification" else "segmentation_v1.0",
                )

    def test_brick_kiln_uses_the_corrected_geobench_channel_order(self):
        spec = get_dataset_spec("m-brick-kiln")
        # geo-bench issue #29 / PR #30, README "Known issues".
        self.assertEqual(spec.band_names, (
            "B01", "B02", "B03", "B04", "B05", "B07", "B08A", "B08", "B11", "B12",
            "TCI_R", "TCI_G", "TCI_B",
        ))
        self.assertNotIn("B06", spec.band_names)
        self.assertNotIn("B09", spec.band_names)
        self.assertNotIn("B10", spec.band_names)
        self.assertEqual(spec.channel_names(spec.profile_band_indices("rgb")),
                         ["TCI_R", "TCI_G", "TCI_B"])
        ms = spec.profile_band_indices("multispectral")
        self.assertFalse(any(name.startswith("TCI") for name in spec.channel_names(ms)))
        self.assertEqual(spec.rgb_positions(ms), (3, 2, 1))  # B04, B03, B02
        self.assertNotIn("sentinel2_13", spec.input_profiles)

    def test_eurosat_mapping_agrees_with_the_upstream_fix(self):
        spec = get_dataset_spec("m-eurosat")
        self.assertEqual(spec.band_names[12], "B08A")  # geo-bench issue #28
        self.assertEqual(spec.channel_names(spec.sentinel2_13_indices), list(SSL4EO_S2_ORDER))
        self.assertEqual(spec.channel_names(spec.rgb_indices), ["B04", "B03", "B02"])
        self.assertEqual(spec.sentinel2_13_indices, (0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11))

    def test_cashew_excludes_cloud_probability_and_has_no_b10(self):
        spec = get_dataset_spec("m-cashew-plant")
        names = spec.channel_names(spec.band_indices)
        self.assertEqual(len(names), 12)
        self.assertNotIn("CLOUD_PROBABILITY", names)
        self.assertNotIn("B10", spec.band_names)
        self.assertEqual(spec.rgb_positions(spec.band_indices), (3, 2, 1))
        self.assertNotIn("sentinel2_13", spec.input_profiles)

    def test_forestnet_and_nz_cattle_band_order(self):
        forest = get_dataset_spec("m-forestnet")
        self.assertEqual(forest.band_names, ("red", "green", "blue", "nir", "swir1", "swir2"))
        self.assertEqual(forest.rgb_positions(forest.multispectral_indices), (0, 1, 2))
        cattle = get_dataset_spec("m-nz-cattle")
        self.assertEqual(cattle.channel_names(cattle.band_indices), ["red", "green", "blue"])

    def test_listing_exposes_views_and_notes(self):
        listed = {item["dataset_id"]: item for item in list_dataset_specs()}
        self.assertEqual(listed["m-brick-kiln"]["input_views"]["rgb"], ["TCI_R", "TCI_G", "TCI_B"])
        self.assertTrue(listed["m-cashew-plant"]["channel_notes"])


class EligibilityTests(unittest.TestCase):
    EXPECTED = {
        "m-eurosat": {"resnet18_rgb", "convnext_tiny_rgb", "resnet18_s2", "resnet50_s2"},
        "m-pv4ger": {"resnet18_rgb", "convnext_tiny_rgb"},
        "m-forestnet": {"resnet18_rgb", "convnext_tiny_rgb", "resnet18_ms", "resnet50_ms"},
        "m-brick-kiln": {"resnet18_rgb", "convnext_tiny_rgb", "resnet18_ms", "resnet50_ms"},
        "m-pv4ger-seg": {"unet_r34", "deeplabv3plus_r50", "fcn_resnet50"},
        "m-chesapeake": {"unet_r34", "deeplabv3plus_r50", "fcn_resnet50"},
        "m-nz-cattle": {"unet_r34", "deeplabv3plus_r50", "fcn_resnet50"},
        "m-cashew-plant": {"unet_r34", "deeplabv3plus_r50", "fcn_resnet50"},
    }

    def test_capability_rule_yields_expected_options(self):
        for dataset_id, expected in self.EXPECTED.items():
            with self.subTest(dataset_id):
                self.assertEqual(set(compatible_models(dataset_id)), expected)

    def test_no_model_names_datasets(self):
        for spec in MODEL_REGISTRY.values():
            self.assertFalse(hasattr(spec, "datasets"))

    def test_incompatible_requests_are_rejected(self):
        for model_id, dataset_id in (
            ("resnet18_ms", "m-pv4ger"),         # no multispectral view
            ("resnet18_ms", "m-eurosat"),        # existing dataset unchanged
            ("resnet18_s2", "m-brick-kiln"),     # no B06/B09/B10: not SSL4EO 13-band
            ("resnet18_s2", "m-cashew-plant"),   # no B10 and wrong task
            ("unet_r34", "m-forestnet"),
            ("resnet18_rgb", "m-nz-cattle"),
        ):
            with self.subTest((model_id, dataset_id)):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    validate_model_for_dataset(model_id, dataset_id)

    def test_options_describe_the_exact_input_channels(self):
        options = {o["model_id"]: o for o in list_model_options("m-brick-kiln")}
        self.assertEqual(options["resnet18_ms"]["dataset_input"]["input_channels"], 10)
        self.assertEqual(options["resnet18_rgb"]["dataset_input"]["band_names"],
                         ["TCI_R", "TCI_G", "TCI_B"])
        self.assertTrue(all("eligible" in option for option in options.values()))

    def test_segmentation_encoders_pad_to_a_multiple(self):
        self.assertEqual(get_model_spec("unet_r34").input_multiple, 32)
        self.assertEqual(get_model_spec("deeplabv3plus_r50").input_multiple, 32)
        self.assertEqual(get_model_spec("fcn_resnet50").input_multiple, 1)


class LoaderRegressionTests(unittest.TestCase):
    def test_every_dataset_and_compatible_input_view_loads_valid_batches(self):
        for dataset_id in DATASET_REGISTRY:
            spec = get_dataset_spec(dataset_id)
            for model_id in compatible_models(dataset_id):
                with self.subTest((dataset_id, model_id)):
                    data, task = loaders(dataset_id, model_id)
                    expected = spec.profile_band_indices(get_model_spec(model_id).input_profile)
                    images, targets = next(iter(data.train_loader))
                    self.assertEqual(tuple(images.shape),
                                     (2, len(expected), *spec.native_resolution))
                    self.assertEqual(images.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(images).all())
                    self.assertEqual(targets.dtype, torch.long)
                    self.assertTrue(int(targets.min()) >= 0)
                    self.assertTrue(int(targets.max()) < spec.num_classes)
                    if spec.task_type == "segmentation":
                        self.assertEqual(tuple(targets.shape), (2, *spec.native_resolution))
                    else:
                        self.assertEqual(tuple(targets.shape), (2,))
                    self.assertEqual(data.input_channels, len(expected))
                    self.assertEqual(data.preprocessing["band_indices"], list(expected))
                    self.assertEqual(data.preprocessing["band_names"], spec.channel_names(expected))
                    self.assertNotIn("test", task.splits_requested)
                    next(iter(data.val_loader))

    def test_original_datasets_keep_their_fixed_normalization(self):
        cases = {
            ("m-pv4ger", "resnet18_rgb"): "imagenet_rgb",
            ("m-eurosat", "resnet18_rgb"): "imagenet_rgb",
            ("m-eurosat", "resnet18_s2"): "none",
            ("m-pv4ger-seg", "unet_r34"): "imagenet_rgb",
            ("m-chesapeake", "unet_r34"): "imagenet_rgb_plus_nir",
        }
        for (dataset_id, model_id), normalization in cases.items():
            with self.subTest((dataset_id, model_id)):
                data, _ = loaders(dataset_id, model_id)
                self.assertEqual(data.preprocessing["normalization"], normalization)
                if model_id != "resnet18_s2":  # S2 checkpoints load 13-band filters directly
                    self.assertEqual(data.rgb_positions, (0, 1, 2))
        data, _ = loaders("m-eurosat", "resnet18_s2")
        self.assertEqual(data.preprocessing["band_indices"], [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 9, 10, 11])

    def test_new_datasets_standardize_with_training_split_statistics(self):
        data, task = loaders("m-brick-kiln", "resnet18_ms")
        stats = data.preprocessing["normalization"]
        self.assertEqual(stats["type"], "train_channel_stats")
        self.assertEqual(stats["split"], "train")
        self.assertEqual(stats["train_size"], 4)
        # Channel c is centred on 10*(c+1): raw units are preserved in the stats.
        np.testing.assert_allclose(stats["mean"], [10.0 * (c + 1) for c in range(10)], atol=0.5)
        np.testing.assert_allclose(stats["std"], [2.0] * 10, rtol=0.1)
        images, _ = next(iter(data.train_loader))
        self.assertLess(float(images.mean().abs()), 0.2)
        self.assertEqual(data.rgb_positions, (3, 2, 1))
        self.assertEqual(data.preprocessing["excluded_channels"], ["TCI_R", "TCI_G", "TCI_B"])

    def test_selected_physical_channels_arrive_in_view_order(self):
        identity = {"type": "train_channel_stats", "mean": [0.0] * 3, "std": [1.0] * 3}
        spec = get_dataset_spec("m-brick-kiln")
        dataset = GeoBenchDataset(FakeSplit("m-brick-kiln", 1, 1), spec, spec.rgb_indices,
                                  identity, spec.native_resolution, augment=False)
        image, _ = dataset[0]
        np.testing.assert_allclose(image.mean(dim=(1, 2)).numpy(), [110.0, 120.0, 130.0], atol=0.5)
        cashew = get_dataset_spec("m-cashew-plant")
        identity12 = {"type": "train_channel_stats", "mean": [0.0] * 12, "std": [1.0] * 12}
        image, _ = GeoBenchDataset(FakeSplit("m-cashew-plant", 1, 1), cashew, cashew.band_indices,
                                   identity12, cashew.native_resolution, augment=False)[0]
        self.assertEqual(image.shape[0], 12)
        self.assertLess(float(image[-1].mean()), 125.0)  # cloud probability (130) not included

    def test_statistics_never_read_validation_or_test(self):
        split = FakeSplit("m-forestnet", 300, 1)
        stats = compute_channel_statistics(split, (0, 1, 2), max_samples=10)
        self.assertEqual(stats["samples"], 10)
        self.assertEqual(stats["train_size"], 300)

    def test_constant_channels_do_not_divide_by_zero(self):
        class Constant(FakeSplit):
            def __getitem__(self, index):
                sample = super().__getitem__(index)
                sample.bands[0].data = np.full_like(sample.bands[0].data, 7)
                return sample
        stats = compute_channel_statistics(Constant("m-forestnet", 2, 1), (0, 1))
        self.assertEqual(stats["std"][0], 1.0)
        self.assertEqual(stats["constant_channels"], [0])

    def test_checkpoint_preprocessing_survives_weights_only_loading(self):
        data, _ = loaders("m-cashew-plant", "unet_r34")
        buffer = __import__("io").BytesIO()
        torch.save({"preprocessing": data.preprocessing}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)["preprocessing"]
        spec = get_dataset_spec("m-cashew-plant")
        # The sealed evaluator rebuilds the dataset exactly like this.
        test_view = GeoBenchDataset(FakeSplit("m-cashew-plant", 1, 3), spec,
                                    tuple(restored["band_indices"]), restored["normalization"],
                                    tuple(restored["effective_resolution"]), augment=False)
        image, mask = test_view[0]
        self.assertEqual(tuple(image.shape), (12, 256, 256))
        self.assertEqual(tuple(mask.shape), (256, 256))


class MaskAndLabelTests(unittest.TestCase):
    def test_cashew_sparse_classes_are_valid_and_out_of_range_rejected(self):
        task = FakeTask("m-cashew-plant", mask_values=[0, 1, 2, 3, 5])
        data, _ = loaders("m-cashew-plant", "unet_r34", task)
        _, masks = next(iter(data.train_loader))
        self.assertEqual(data.num_classes, 7)  # classes 4 and 6 absent here, not globally
        self.assertTrue(set(masks.unique().tolist()) <= {0, 1, 2, 3, 5})
        bad = FakeTask("m-cashew-plant", mask_values=[0, 7])
        data, _ = loaders("m-cashew-plant", "unet_r34", bad)
        with self.assertRaisesRegex(ValueError, "outside the registry contract"):
            next(iter(data.train_loader))

    def test_nz_cattle_binary_mask_band(self):
        data, _ = loaders("m-nz-cattle", "deeplabv3plus_r50")
        _, masks = next(iter(data.train_loader))
        self.assertTrue(set(masks.unique().tolist()) <= {0, 1})

    def test_classification_labels_must_be_integers_in_range(self):
        spec = get_dataset_spec("m-forestnet")
        identity = {"type": "train_channel_stats", "mean": [0.0] * 6, "std": [1.0] * 6}

        class BadLabel(FakeSplit):
            def __getitem__(self, index):
                sample = super().__getitem__(index)
                sample.label = 12 if index == 0 else 1.5
                return sample
        dataset = GeoBenchDataset(BadLabel("m-forestnet", 2, 1), spec, spec.band_indices,
                                  identity, spec.native_resolution, augment=False)
        with self.assertRaisesRegex(ValueError, "outside"):
            dataset[0]
        with self.assertRaisesRegex(TypeError, "integer"):
            dataset[1]

    def test_channel_count_contract_is_enforced(self):
        class Missing(FakeSplit):
            def __getitem__(self, index):
                sample = super().__getitem__(index)
                sample.bands = sample.bands[:12]
                return sample

        class Task(FakeTask):
            def get_dataset(self, split, band_names=None):
                return Missing("m-brick-kiln", 2, 1)
        with self.assertRaisesRegex(ValueError, "Registry says 13 channels"):
            loaders("m-brick-kiln", "resnet18_ms", Task("m-brick-kiln"))


class ModelAdaptationTests(unittest.TestCase):
    def conv(self):
        module = types.SimpleNamespace()
        module.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, bias=False)
        return module

    def test_rgb_filters_go_to_verified_rgb_channels(self):
        module = self.conv()
        old = module.conv1.weight.detach().clone()
        _adapt_first_convolution(module, "conv1", 12, (3, 2, 1))
        new = module.conv1.weight.detach()
        self.assertEqual(new.shape[1], 12)
        torch.testing.assert_close(new[:, 3], old[:, 0])  # B04 <- red
        torch.testing.assert_close(new[:, 2], old[:, 1])  # B03 <- green
        torch.testing.assert_close(new[:, 1], old[:, 2])  # B02 <- blue
        mean = old.mean(dim=1)
        for channel in (0, 4, 5, 11):
            torch.testing.assert_close(new[:, channel], mean)

    def test_default_positions_reproduce_the_original_chesapeake_adaptation(self):
        module = self.conv()
        old = module.conv1.weight.detach().clone()
        _adapt_first_convolution(module, "conv1", 4)
        new = module.conv1.weight.detach()
        torch.testing.assert_close(new[:, :3], old)
        torch.testing.assert_close(new[:, 3], old.mean(dim=1))

    def test_three_rgb_channels_in_order_are_untouched(self):
        module = self.conv()
        original = module.conv1
        _adapt_first_convolution(module, "conv1", 3, (0, 1, 2))
        self.assertIs(module.conv1, original)

    def test_padding_preserves_native_resolution_and_state_dict(self):
        class NeedsMultiple(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 2, 1)

            def forward(self, images):
                if images.shape[-1] % 32 or images.shape[-2] % 32:
                    raise RuntimeError("input must be divisible by 32")
                return self.head(images)

        model = NeedsMultiple()
        keys = set(model.state_dict())
        apply_input_padding(model, 32)
        self.assertEqual(tuple(model(torch.zeros(1, 3, 500, 500)).shape), (1, 2, 500, 500))
        self.assertEqual(tuple(model(torch.zeros(1, 3, 256, 256)).shape), (1, 2, 256, 256))
        self.assertEqual(set(model.state_dict()), keys)
        self.assertEqual(model.gotrain_input_padding["multiple"], 32)


if __name__ == "__main__":
    unittest.main()
