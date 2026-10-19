"""Explicit, explainable image-quality diagnostics using OpenCV and NumPy.

Calculates photometric and clarity measurements on cropped face regions:
1. Laplacian variance as an explainable sharpness/blur indicator.
2. Exposure statistics (mean luminance, underexposure ratio, overexposure ratio).
3. Left/right face-region luminance comparison to quantify illumination asymmetry.

Note: These measurements are diagnostic indicators, NOT calibrated biometric quality
scores. They provide explainable signals for surveillance/capture inspection without
inventing arbitrary rejection thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PhotometricDiagnostics:
    """Diagnostic photometric measurements on a detected face crop."""

    sharpness_laplacian_var: float | None = None
    mean_luminance: float | None = None
    underexposure_ratio: float | None = None  # Fraction of pixels with luminance < 15
    overexposure_ratio: float | None = None   # Fraction of pixels with luminance > 240
    illumination_asymmetry: float | None = None  # Normalized relative difference |L - R| / avg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_sharpness_laplacian_var(face_bgr: np.ndarray | None) -> float | None:
    """Compute the variance of the Laplacian of the grayscale face crop.

    Higher values typically indicate sharper edges; lower values indicate blur.
    Returns None if the image is empty, None, or invalid.
    """
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return None
    try:
        if face_bgr.ndim == 3 and face_bgr.shape[2] == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        elif face_bgr.ndim == 2:
            gray = face_bgr
        else:
            return None

        if gray.shape[0] < 2 or gray.shape[1] < 2:
            return None

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(laplacian.var())
        return round(var, 2) if np.isfinite(var) else None
    except Exception:
        return None


def compute_exposure_statistics(
    face_bgr: np.ndarray | None,
    *,
    underexposed_threshold: int = 15,
    overexposed_threshold: int = 240,
) -> tuple[float | None, float | None, float | None]:
    """Compute mean luminance and ratios of underexposed and overexposed pixels in the face crop.

    Returns:
        (mean_luminance, underexposure_ratio, overexposure_ratio)
        Values are between 0.0 and 255.0 for luminance, and 0.0 and 1.0 for ratios.
    """
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return None, None, None
    try:
        if face_bgr.ndim == 3 and face_bgr.shape[2] == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        elif face_bgr.ndim == 2:
            gray = face_bgr
        else:
            return None, None, None

        total_pixels = gray.size
        if total_pixels == 0:
            return None, None, None

        mean_lum = float(np.mean(gray))
        under_count = int(np.count_nonzero(gray < underexposed_threshold))
        over_count = int(np.count_nonzero(gray > overexposed_threshold))

        under_ratio = float(under_count / total_pixels)
        over_ratio = float(over_count / total_pixels)

        return (
            round(mean_lum, 2) if np.isfinite(mean_lum) else None,
            round(under_ratio, 4) if np.isfinite(under_ratio) else None,
            round(over_ratio, 4) if np.isfinite(over_ratio) else None,
        )
    except Exception:
        return None, None, None


def compute_illumination_asymmetry(face_bgr: np.ndarray | None) -> float | None:
    """Compute left/right luminance comparison on the vertical halves of the face crop.

    Metric: |mean(Left) - mean(Right)| / max(1.0, (mean(Left) + mean(Right)) / 2.0)
    Ranges from 0.0 (perfectly symmetric) to ~2.0 (one side entirely black, one bright).
    Returns None if the image is invalid or too small.
    """
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return None
    try:
        if face_bgr.ndim == 3 and face_bgr.shape[2] == 3:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        elif face_bgr.ndim == 2:
            gray = face_bgr
        else:
            return None

        h, w = gray.shape[:2]
        if w < 4 or h < 2:
            return None

        mid_x = w // 2
        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:]

        left_mean = float(np.mean(left_half))
        right_mean = float(np.mean(right_half))

        avg_mean = (left_mean + right_mean) / 2.0
        denom = max(1.0, avg_mean)
        asymmetry = abs(left_mean - right_mean) / denom

        return round(float(asymmetry), 4) if np.isfinite(asymmetry) else None
    except Exception:
        return None


def analyze_photometrics(face_bgr: np.ndarray | None) -> PhotometricDiagnostics:
    """Analyze all photometric and sharpness diagnostics on a face crop."""
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return PhotometricDiagnostics()

    sharpness = compute_sharpness_laplacian_var(face_bgr)
    mean_lum, under_ratio, over_ratio = compute_exposure_statistics(face_bgr)
    asymmetry = compute_illumination_asymmetry(face_bgr)

    return PhotometricDiagnostics(
        sharpness_laplacian_var=sharpness,
        mean_luminance=mean_lum,
        underexposure_ratio=under_ratio,
        overexposure_ratio=over_ratio,
        illumination_asymmetry=asymmetry,
    )
