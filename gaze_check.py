"""Optional gaze direction estimation using UniFace MobileGaze.

MobileGaze predicts gaze line-of-sight direction (pitch and yaw) from a detected face crop.
Crucial Distinction:
- Head Pose: Estimates rigid orientation of the skull / head (pitch, yaw, roll).
- Gaze Direction: Estimates eye line-of-sight direction relative to the camera (pitch, yaw).

UniFace returns pitch and yaw in radians. This module converts them to degrees for
consistent dashboard interpretation and reporting.

Sign and units:
- Gaze pitch: vertical angle (degrees).
- Gaze yaw: horizontal angle (degrees).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GazeAngles:
    """Estimated 2D gaze angles in degrees."""

    pitch: float  # Vertical gaze angle (degrees)
    yaw: float    # Horizontal gaze angle (degrees)


class GazeChecker:
    """Loads UniFace MobileGaze once and reuses it. ``estimator`` may be injected for testing."""

    def __init__(
        self,
        estimator: Any | None = None,
        providers: list[str] | None = None,
        model_name: Any | None = None,
    ) -> None:
        if estimator is None:
            try:
                from uniface.constants import GazeWeights
                from uniface.gaze import MobileGaze

                chosen_model = model_name or GazeWeights.RESNET34
                estimator = MobileGaze(model_name=chosen_model, providers=providers)
            except Exception as exc:
                raise RuntimeError(f'Failed to initialize UniFace MobileGaze: {exc}') from exc
        self.estimator = estimator

    def estimate(self, face_crop: np.ndarray) -> GazeAngles:
        """Estimate gaze direction angles in degrees from a BGR face crop.

        Args:
            face_crop: Cropped bounding-box region of the face (BGR format).

        Returns:
            GazeAngles with pitch and yaw in degrees.
        """
        if face_crop is None or not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
            raise ValueError('face crop is empty or invalid for gaze estimation')

        h, w = face_crop.shape[:2]
        if h < 4 or w < 4:
            raise ValueError(f'face crop too small for gaze estimation: {w}x{h}')

        result = self.estimator.estimate(face_crop)

        # UniFace MobileGaze returns GazeResult(pitch=float, yaw=float) in RADIANS.
        pitch_deg = float(math.degrees(result.pitch))
        yaw_deg = float(math.degrees(result.yaw))

        if not (math.isfinite(pitch_deg) and math.isfinite(yaw_deg)):
            raise ValueError(f'gaze model returned non-finite angles: pitch={pitch_deg}, yaw={yaw_deg}')

        return GazeAngles(pitch=round(pitch_deg, 2), yaw=round(yaw_deg, 2))
