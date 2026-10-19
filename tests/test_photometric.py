"""Unit tests for explicit photometric and clarity diagnostics."""

import unittest
import numpy as np
import cv2

from photometric_quality import (
    compute_sharpness_laplacian_var,
    compute_exposure_statistics,
    compute_illumination_asymmetry,
    analyze_photometrics,
    PhotometricDiagnostics,
)


class TestPhotometricQuality(unittest.TestCase):
    def setUp(self):
        # Create a standard synthetic face crop (64x64, 3 channels)
        self.synth_crop = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Add high-contrast sharp edges to half of it
        self.synth_crop[10:50, 10:50] = 220
        self.synth_crop[20:40, 20:40] = 40

    def test_sharpness_none_and_empty(self):
        self.assertIsNone(compute_sharpness_laplacian_var(None))
        self.assertIsNone(compute_sharpness_laplacian_var(np.array([], dtype=np.uint8)))
        self.assertIsNone(compute_sharpness_laplacian_var(np.zeros((1, 1, 3), dtype=np.uint8)))

    def test_sharpness_blur_vs_sharp(self):
        sharp_var = compute_sharpness_laplacian_var(self.synth_crop)
        self.assertIsNotNone(sharp_var)
        self.assertGreater(sharp_var, 0)

        # Blurred version should have significantly lower variance
        blurred_crop = cv2.GaussianBlur(self.synth_crop, (15, 15), 0)
        blur_var = compute_sharpness_laplacian_var(blurred_crop)
        self.assertIsNotNone(blur_var)
        self.assertLess(blur_var, sharp_var)

    def test_exposure_statistics_extreme_cases(self):
        # All-dark image
        black_img = np.zeros((32, 32, 3), dtype=np.uint8)
        mean_lum, under_ratio, over_ratio = compute_exposure_statistics(black_img)
        self.assertEqual(mean_lum, 0.0)
        self.assertEqual(under_ratio, 1.0)
        self.assertEqual(over_ratio, 0.0)

        # All-bright image
        white_img = np.full((32, 32, 3), 255, dtype=np.uint8)
        mean_lum, under_ratio, over_ratio = compute_exposure_statistics(white_img)
        self.assertEqual(mean_lum, 255.0)
        self.assertEqual(under_ratio, 0.0)
        self.assertEqual(over_ratio, 1.0)

    def test_exposure_statistics_invalid_inputs(self):
        mean_lum, under, over = compute_exposure_statistics(None)
        self.assertIsNone(mean_lum)
        self.assertIsNone(under)
        self.assertIsNone(over)

    def test_illumination_asymmetry_symmetric(self):
        # Symmetric image has 0.0 asymmetry
        symmetric = np.full((32, 32, 3), 100, dtype=np.uint8)
        asym = compute_illumination_asymmetry(symmetric)
        self.assertEqual(asym, 0.0)

    def test_illumination_asymmetry_asymmetric(self):
        # Left side very dark (10), right side bright (200)
        asymmetric = np.zeros((32, 32, 3), dtype=np.uint8)
        asymmetric[:, :16] = 10
        asymmetric[:, 16:] = 200
        asym = compute_illumination_asymmetry(asymmetric)
        self.assertIsNotNone(asym)
        self.assertGreater(asym, 0.5)

    def test_analyze_photometrics_full(self):
        diag = analyze_photometrics(self.synth_crop)
        self.assertIsInstance(diag, PhotometricDiagnostics)
        self.assertIsNotNone(diag.sharpness_laplacian_var)
        self.assertIsNotNone(diag.mean_luminance)
        self.assertIsNotNone(diag.underexposure_ratio)
        self.assertIsNotNone(diag.overexposure_ratio)
        self.assertIsNotNone(diag.illumination_asymmetry)

    def test_analyze_photometrics_none(self):
        diag = analyze_photometrics(None)
        self.assertIsNone(diag.sharpness_laplacian_var)
        self.assertIsNone(diag.mean_luminance)


if __name__ == '__main__':
    unittest.main()
