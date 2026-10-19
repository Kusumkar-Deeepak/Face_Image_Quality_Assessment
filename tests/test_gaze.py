"""Unit tests for UniFace MobileGaze gaze_check module."""

import math
import unittest
from dataclasses import dataclass
import numpy as np

from gaze_check import GazeAngles, GazeChecker


@dataclass
class MockGazeResult:
    pitch: float
    yaw: float


class MockEstimator:
    def __init__(self, pitch_rad: float = 0.0, yaw_rad: float = 0.0, fail: bool = False):
        self.pitch_rad = pitch_rad
        self.yaw_rad = yaw_rad
        self.fail = fail

    def estimate(self, face_crop: np.ndarray) -> MockGazeResult:
        if self.fail:
            raise RuntimeError('Mock estimation failure')
        return MockGazeResult(pitch=self.pitch_rad, yaw=self.yaw_rad)


class TestGazeCheck(unittest.TestCase):
    def setUp(self):
        self.crop = np.zeros((64, 64, 3), dtype=np.uint8)

    def test_mock_degrees_conversion(self):
        # 30 degrees = pi / 6 radians ~= 0.5235987755982988
        # -45 degrees = -pi / 4 radians ~= -0.7853981633974483
        mock = MockEstimator(pitch_rad=math.radians(30.0), yaw_rad=math.radians(-45.0))
        checker = GazeChecker(estimator=mock)
        angles = checker.estimate(self.crop)

        self.assertIsInstance(angles, GazeAngles)
        self.assertAlmostEqual(angles.pitch, 30.0, places=1)
        self.assertAlmostEqual(angles.yaw, -45.0, places=1)

    def test_invalid_and_empty_crop(self):
        mock = MockEstimator()
        checker = GazeChecker(estimator=mock)

        with self.assertRaises(ValueError):
            checker.estimate(None)

        with self.assertRaises(ValueError):
            checker.estimate(np.array([], dtype=np.uint8))

        with self.assertRaises(ValueError):
            checker.estimate(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_estimator_failure_propagates(self):
        mock = MockEstimator(fail=True)
        checker = GazeChecker(estimator=mock)

        with self.assertRaises(RuntimeError):
            checker.estimate(self.crop)

    def test_non_finite_angles_raise(self):
        mock = MockEstimator(pitch_rad=float('nan'), yaw_rad=0.0)
        checker = GazeChecker(estimator=mock)

        with self.assertRaises(ValueError):
            checker.estimate(self.crop)


if __name__ == '__main__':
    unittest.main()
