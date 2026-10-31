import unittest

import numpy as np

from ml.dataset import inspect_sample_resolution
from ml.registry import get_dataset_spec


class _Band:
    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.uint8)


class _SegmentationSample:
    def __init__(self, image_shape, mask_shape=None):
        self.bands = [_Band(image_shape) for _ in range(4)]
        self.label = _Band(mask_shape or image_shape)


class DatasetResolutionInspectionTests(unittest.TestCase):
    def test_pv4ger_seg_actual_sample_is_320(self):
        spec = get_dataset_spec("m-pv4ger-seg")
        sample = _SegmentationSample((320, 320))
        self.assertEqual(
            inspect_sample_resolution(sample, spec, spec.band_indices),
            (320, 320),
        )

    def test_chesapeake_actual_sample_is_256(self):
        spec = get_dataset_spec("m-chesapeake")
        sample = _SegmentationSample((256, 256))
        self.assertEqual(
            inspect_sample_resolution(sample, spec, spec.band_indices),
            (256, 256),
        )

    def test_segmentation_image_mask_mismatch_is_rejected(self):
        spec = get_dataset_spec("m-pv4ger-seg")
        sample = _SegmentationSample((320, 320), (64, 64))
        with self.assertRaisesRegex(ValueError, "image/mask mismatch"):
            inspect_sample_resolution(sample, spec, spec.band_indices)


if __name__ == "__main__":
    unittest.main()
