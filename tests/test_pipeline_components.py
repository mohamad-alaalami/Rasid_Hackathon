"""Composable pipeline components: validation, compatibility, augmentation
synchronization, spectral safety, builder wiring, and loader wiring.

Pure torch; no dataset, network, GPU, or test split is used.
"""

import math
import unittest

import torch

from ml.components import (
    CATALOG,
    DEFAULT_AUGMENTATIONS,
    PipelineError,
    apply_augmentations,
    build_loss,
    build_optimizer,
    build_scheduler,
    class_weights,
    resolve_pipeline,
)
from ml.config import TrainingConfig

try:
    from test_geobench_datasets import FakeTask, loaders
except ImportError:  # run as tests.<module> from the repo root
    from tests.test_geobench_datasets import FakeTask, loaders


def config(model="resnet18_rgb", epochs=3, **pipeline):
    return TrainingConfig.from_mapping({"model": model, "epochs": epochs,
                                        "pipeline": pipeline or None, "num_workers": 0,
                                        "batch_size": 2, "device": "cpu"})


class ResolutionTests(unittest.TestCase):
    def test_defaults_reproduce_the_historical_pipeline(self):
        resolved = resolve_pipeline(None)
        self.assertEqual(resolved["augmentations"], list(DEFAULT_AUGMENTATIONS))
        self.assertEqual(resolved["loss"], {"name": "cross_entropy", "label_smoothing": 0.0,
                                            "class_weighting": "none"})
        self.assertEqual(resolved["optimizer"], {"name": "adamw", "beta1": 0.9, "beta2": 0.999})
        self.assertEqual(resolved["scheduler"], {"name": "cosine"})
        self.assertEqual(resolved["dropout"], 0.0)
        self.assertIsNone(resolved["input_view"])
        self.assertEqual(TrainingConfig().pipeline, None)

    def test_every_problem_is_reported_explicitly(self):
        with self.assertRaises(PipelineError) as raised:
            resolve_pipeline({
                "loss": {"name": "hinge"},
                "optimizer": {"name": "sgd", "momentum": 1.5},
                "augmentations": [{"name": "hflip"}, {"name": "hflip"}, {"name": "hue_shift"}],
                "scheduler": {"name": "step", "warmup": 2},
                "dropout": 0.9,
                "mixup": True,
                "custom_components": [{"code": "x"}],
            })
        text = " | ".join(raised.exception.errors)
        for fragment in ("unknown loss 'hinge'", "momentum=1.5", "listed more than once",
                         "unknown augmentation 'hue_shift'", "unsupported parameter 'warmup'",
                         "dropout=0.9", "unsupported pipeline key 'mixup'", "reserved for scope S2"):
            self.assertIn(fragment, text)

    def test_no_spectrally_unsafe_augmentation_is_offered(self):
        names = {name for kind, name in CATALOG if kind == "augmentation"}
        self.assertFalse(names & {"hue", "saturation", "color_jitter", "channel_shuffle",
                                  "channel_dropout"})
        photometric = [c for c in CATALOG.values() if c.effect == "photometric"]
        self.assertTrue(all("shared" in c.description or "per-channel sigma" in c.description
                            for c in photometric))


class CompatibilityTests(unittest.TestCase):
    def assert_rejected(self, dataset_id, fragment, model="resnet18_rgb", epochs=3, **pipeline):
        with self.assertRaises(PipelineError) as raised:
            config(model, epochs, **pipeline).validate_for_dataset(dataset_id)
        self.assertIn(fragment, " | ".join(raised.exception.errors))

    def test_meaningless_combinations_are_rejected(self):
        self.assert_rejected("m-eurosat", "does not support classification",
                             loss={"name": "dice_cross_entropy"})
        self.assert_rejected("m-cashew-plant", "dropout is supported only", model="unet_r34",
                             dropout=0.2)
        self.assert_rejected("m-eurosat", "train_channel_stats is incompatible",
                             model="resnet18_s2", normalization="train_channel_stats")
        self.assert_rejected("m-brick-kiln", "cannot take the 'multispectral' view",
                             input_view="multispectral")
        self.assert_rejected("m-nz-cattle", "no verified 'multispectral' view", model="unet_r34",
                             input_view="multispectral")
        self.assert_rejected("m-forestnet", "warmup_cosine needs epochs", model="resnet18_ms",
                             epochs=1, scheduler={"name": "warmup_cosine", "warmup_epochs": 1})

    def test_valid_compositions_are_accepted(self):
        for model, dataset_id, pipeline in (
            ("resnet18_ms", "m-brick-kiln", {"input_view": "rgb"}),
            ("unet_r34", "m-cashew-plant", {"input_view": "rgb",
                                            "loss": {"name": "dice_cross_entropy"}}),
            ("resnet18_rgb", "m-pv4ger", {"normalization": "train_channel_stats",
                                          "optimizer": "sgd", "scheduler": "step",
                                          "dropout": 0.3,
                                          "loss": {"name": "focal", "gamma": 1.0}}),
            ("deeplabv3plus_r50", "m-nz-cattle", {"augmentations": [
                {"name": "rot90"}, {"name": "random_resized_crop", "min_scale": 0.5},
                {"name": "brightness"}, {"name": "gaussian_noise"}]}),
        ):
            with self.subTest((model, dataset_id)):
                self.assertEqual(config(model, **pipeline).validate_for_dataset(dataset_id), model)


def historical_flips(image, target, segmentation):
    """The pre-Phase-1A augmentation, verbatim."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        if segmentation:
            target = torch.flip(target, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
        if segmentation:
            target = torch.flip(target, dims=(-2,))
    return image, target


def blocky_mask(size=64, block=16, classes=4):
    rows = torch.arange(size)[:, None] // block
    cols = torch.arange(size)[None, :] // block
    return ((rows + 2 * cols) % classes).long()


class AugmentationTests(unittest.TestCase):
    def test_default_flips_match_historical_draws_and_output(self):
        image = torch.rand(3, 16, 16)
        mask = torch.randint(0, 2, (16, 16))
        for seed in range(20):
            torch.manual_seed(seed)
            old = historical_flips(image, mask, True)
            old_next = torch.rand(())
            torch.manual_seed(seed)
            new = apply_augmentations(image, mask, list(DEFAULT_AUGMENTATIONS), "segmentation")
            self.assertTrue(torch.equal(old[0], new[0]) and torch.equal(old[1], new[1]))
            self.assertEqual(float(old_next), float(torch.rand(())))  # same RNG consumption

    def test_spatial_ops_move_masks_with_images(self):
        mask = blocky_mask()
        image = torch.stack([mask.float(), torch.rand(64, 64)])
        for name in ("hflip", "vflip", "rot90"):
            for seed in range(5):
                torch.manual_seed(seed)
                out_image, out_mask = apply_augmentations(
                    image, mask, [{"name": name, "p": 1.0}], "segmentation")
                self.assertTrue(torch.equal(out_image[0].long(), out_mask), name)
                self.assertFalse(torch.equal(out_mask, mask) and name != "rot90" and seed == 0)

    def test_random_resized_crop_keeps_image_and_mask_aligned(self):
        mask = blocky_mask()
        image = mask.float()[None]
        for seed in range(10):
            torch.manual_seed(seed)
            out_image, out_mask = apply_augmentations(
                image, mask, [{"name": "random_resized_crop", "p": 1.0, "min_scale": 0.3}],
                "segmentation")
            self.assertEqual(tuple(out_mask.shape), (64, 64))
            agreement = (out_image[0].round().long() == out_mask).float().mean()
            self.assertGreater(float(agreement), 0.9)

    def test_photometric_ops_respect_spectral_relationships(self):
        image = torch.rand(10, 32, 32) + 0.5
        mask = blocky_mask(32, 8)
        torch.manual_seed(0)
        bright, same_mask = apply_augmentations(
            image, mask, [{"name": "brightness", "p": 1.0, "max_delta": 0.3}], "segmentation")
        self.assertTrue(torch.equal(same_mask, mask))
        ratio = bright / image
        self.assertTrue(torch.allclose(ratio, ratio.flatten()[0].expand_as(ratio)))
        torch.manual_seed(1)
        contrasted, _ = apply_augmentations(
            image, mask, [{"name": "contrast", "p": 1.0, "max_delta": 0.3}], "segmentation")
        torch.testing.assert_close(contrasted.mean(dim=(1, 2)), image.mean(dim=(1, 2)))
        constant = torch.full((2, 8, 8), 5.0)
        noisy, _ = apply_augmentations(
            constant, torch.zeros(8, 8).long(), [{"name": "gaussian_noise", "p": 1.0,
                                                  "sigma_fraction": 0.2}], "segmentation")
        self.assertTrue(torch.equal(noisy, constant))  # zero-variance band gets zero noise

    def test_classification_targets_are_untouched(self):
        torch.manual_seed(0)
        _, target = apply_augmentations(torch.rand(3, 8, 8), torch.tensor(4),
                                        [{"name": "rot90", "p": 1.0}], "classification")
        self.assertEqual(int(target), 4)


class BuilderTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(0)
        return torch.nn.Linear(4, 3)

    def test_optimizers_are_built_as_configured(self):
        sgd = build_optimizer(resolve_pipeline({"optimizer": {"name": "sgd", "momentum": 0.8,
                                                              "nesterov": True}}),
                              self.model().parameters(), 0.1, 0.01)
        self.assertIsInstance(sgd, torch.optim.SGD)
        self.assertEqual((sgd.defaults["momentum"], sgd.defaults["nesterov"],
                          sgd.defaults["weight_decay"]), (0.8, True, 0.01))
        adamw = build_optimizer(resolve_pipeline({"optimizer": {"name": "adamw", "beta1": 0.8}}),
                                self.model().parameters(), 0.001, 0.0)
        self.assertIsInstance(adamw, torch.optim.AdamW)
        self.assertEqual(adamw.defaults["betas"], (0.8, 0.999))

    def lrs(self, scheduler, epochs=6):
        optimizer = torch.optim.SGD(self.model().parameters(), lr=1.0)
        built = build_scheduler(resolve_pipeline({"scheduler": scheduler}), optimizer, epochs)
        values = []
        for _ in range(epochs):
            values.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            built.step()
        return values

    def test_schedulers_change_the_learning_rate_trajectory(self):
        reference = torch.optim.SGD(self.model().parameters(), lr=1.0)
        historical = torch.optim.lr_scheduler.CosineAnnealingLR(reference, T_max=6)
        expected = []
        for _ in range(6):
            expected.append(reference.param_groups[0]["lr"])
            reference.step()
            historical.step()
        self.assertEqual(self.lrs("cosine"), expected)
        self.assertEqual(self.lrs("constant"), [1.0] * 6)
        self.assertEqual([round(v, 6) for v in self.lrs({"name": "step", "step_size": 2,
                                                         "gamma": 0.5})],
                         [1.0, 1.0, 0.5, 0.5, 0.25, 0.25])
        warm = self.lrs({"name": "warmup_cosine", "warmup_epochs": 2})
        self.assertLess(warm[0], warm[1])
        self.assertAlmostEqual(warm[2], 1.0)
        self.assertLess(warm[-1], warm[2])

    def test_losses_are_built_and_behave_as_configured(self):
        logits = torch.randn(6, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        default = build_loss(resolve_pipeline(None), 3)
        torch.testing.assert_close(default(logits, targets),
                                   torch.nn.functional.cross_entropy(logits, targets))
        smoothed = build_loss(resolve_pipeline({"loss": {"name": "cross_entropy",
                                                         "label_smoothing": 0.2}}), 3)
        self.assertEqual(smoothed.label_smoothing, 0.2)
        focal0 = build_loss(resolve_pipeline({"loss": {"name": "focal", "gamma": 0.0}}), 3)
        torch.testing.assert_close(focal0(logits, targets), default(logits, targets))
        focal2 = build_loss(resolve_pipeline({"loss": {"name": "focal", "gamma": 2.0}}), 3)
        self.assertLess(float(focal2(logits, targets)), float(default(logits, targets)))
        weighted = build_loss(resolve_pipeline({"loss": {"name": "cross_entropy",
                                                         "class_weighting": "inverse_frequency"}}),
                              3, class_counts=[100, 10, 1])
        self.assertGreater(float(weighted.weight[2]), float(weighted.weight[0]))

    def test_dice_cross_entropy_and_ignore_index(self):
        pipeline = resolve_pipeline({"loss": {"name": "dice_cross_entropy", "dice_weight": 0.0}})
        logits = torch.randn(2, 3, 4, 4)
        targets = torch.randint(0, 3, (2, 4, 4))
        torch.testing.assert_close(build_loss(pipeline, 3)(logits, targets),
                                   torch.nn.functional.cross_entropy(logits, targets))
        dice = build_loss(resolve_pipeline({"loss": {"name": "dice_cross_entropy",
                                                     "dice_weight": 1.0}}), 3, ignore_index=255)
        perfect = torch.nn.functional.one_hot(targets, 3).permute(0, 3, 1, 2).float() * 50
        self.assertLess(float(dice(perfect, targets)), 0.01)
        ignored = targets.clone()
        ignored[0] = 255
        self.assertTrue(math.isfinite(float(dice(logits, ignored))))

    def test_class_weights_are_mean_one_floored_and_capped(self):
        weights = class_weights([1000, 100, 0], "inverse_frequency")
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertLessEqual(float(weights.max() / weights.min()), 10.0 + 1e-4)
        self.assertIsNone(class_weights([1, 2], "none"))


class LoaderWiringTests(unittest.TestCase):
    def test_input_view_and_normalization_choices_change_the_data(self):
        data, _ = loaders_with("m-brick-kiln", "resnet18_ms", input_view="rgb")
        images, _ = next(iter(data.train_loader))
        self.assertEqual(images.shape[1], 3)
        self.assertEqual(data.preprocessing["band_names"], ["TCI_R", "TCI_G", "TCI_B"])
        self.assertEqual(data.preprocessing["input_view"], "rgb")
        default, _ = loaders("m-pv4ger", "resnet18_rgb")
        stats, _ = loaders_with("m-pv4ger", "resnet18_rgb", normalization="train_channel_stats")
        self.assertEqual(default.preprocessing["normalization"], "imagenet_rgb")
        self.assertEqual(stats.preprocessing["normalization"]["type"], "train_channel_stats")

    def test_class_counts_only_when_weighting_is_requested(self):
        plain, _ = loaders("m-cashew-plant", "unet_r34")
        self.assertIsNone(plain.class_counts)
        weighted, _ = loaders_with("m-cashew-plant", "unet_r34",
                                   loss={"name": "dice_cross_entropy",
                                         "class_weighting": "inverse_frequency"})
        self.assertEqual(len(weighted.class_counts), 7)
        self.assertGreater(sum(weighted.class_counts), 0)

    def test_augmentations_reach_training_but_never_validation(self):
        data, _ = loaders_with("m-forestnet", "resnet18_ms",
                               augmentations=[{"name": "brightness", "p": 1.0}])
        self.assertEqual(data.train_loader.dataset.augmentations[0]["name"], "brightness")
        self.assertTrue(data.train_loader.dataset.augment)
        self.assertFalse(data.val_loader.dataset.augment)


def loaders_with(dataset_id, model_id, **pipeline):
    from unittest.mock import patch

    from ml.dataset import create_dataloaders

    task = FakeTask(dataset_id)
    cfg = TrainingConfig.from_mapping({"model": model_id, "batch_size": 2, "num_workers": 0,
                                       "device": "cpu", "pipeline": pipeline})
    cfg.validate_for_dataset(dataset_id)
    with patch("ml.dataset.find_task", lambda *args: task):
        return create_dataloaders(dataset_id, cfg), task


if __name__ == "__main__":
    unittest.main()
