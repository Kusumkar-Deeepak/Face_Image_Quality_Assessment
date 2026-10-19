"""Optional head-pose estimation (pitch / yaw / roll in degrees).

Uses ``uniface.headpose.HeadPose().estimate(face_crop_bgr) -> HeadPoseResult`` (verified in UniFace 4.0.0).
UniFace documents the input as a *bounding-box crop of the original image* in BGR. The 112x112 aligned face
must NOT be used here: alignment removes the in-plane rotation and rescales the face, which would destroy the
pose signal.

This estimates HEAD orientation only. It says nothing about gaze direction.
Sign convention per UniFace docs: pitch > 0 looking down, yaw > 0 looking right, roll > 0 tilting clockwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseAngles:
    pitch: float
    yaw: float
    roll: float


class HeadPoseChecker:
    """Loads the head-pose model once and reuses it. ``estimator`` may be injected (tests)."""

    def __init__(self, estimator: Any | None = None, providers: list[str] | None = None) -> None:
        if estimator is None:
            from uniface.headpose import HeadPose

            estimator = HeadPose(providers=providers)
        self.estimator = estimator

    def estimate(self, image: np.ndarray, bbox: Any) -> PoseAngles:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (int(round(float(v))) for v in np.asarray(bbox, dtype=np.float64))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError(f'face crop too small for head-pose estimation: {(x1, y1, x2, y2)}')
        result = self.estimator.estimate(image[y1:y2, x1:x2])
        angles = PoseAngles(float(result.pitch), float(result.yaw), float(result.roll))
        if not all(np.isfinite([angles.pitch, angles.yaw, angles.roll])):
            raise ValueError(f'head-pose model returned non-finite angles: {angles}')
        return angles
