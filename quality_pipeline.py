from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection import DetectionOutcome, DetectorConfig, FaceDetectorChain
from gaze_check import GazeAngles, GazeChecker
from models import get_model
from photometric_quality import PhotometricDiagnostics, analyze_photometrics
from pose_check import HeadPoseChecker, PoseAngles
from uniface import face_alignment

STATUSES = {'OK', 'READ_ERROR', 'NO_FACE', 'ALIGNMENT_ERROR', 'MODEL_ERROR'}
DECISIONS = {'ACCEPT', 'PASS', 'REJECT', 'REVIEW', 'NO_FACE', 'NOT_EVALUATED', 'ERROR'}
POSE_STATUSES = {'DISABLED', 'SKIPPED', 'NOT_APPLICABLE', 'OK', 'ERROR'}  # SKIPPED: enabled but not run (earlier failure)
GAZE_STATUSES = {'DISABLED', 'SKIPPED', 'NOT_APPLICABLE', 'OK', 'ERROR'}
ALIGNED_SIZE = 112  # official eDifFIQA input: aligned 112x112 RGB face


@dataclass
class EvaluationResult:
    image: str
    image_width: int | None
    image_height: int | None
    detector: str
    face_index: int | None
    confidence: float | None
    bbox: list[float] | None
    quality_score: float | None
    status: str
    decision: str
    error_reason: str | None
    processing_time_ms: float
    # Fields below were added after the original schema; all have defaults.
    decision_reason: str | None = None
    fallback_used: bool = False
    detection_note: str | None = None
    detection_time_ms: float | None = None
    quality_time_ms: float | None = None
    pose_time_ms: float | None = None
    gaze_time_ms: float | None = None
    photometrics_time_ms: float | None = None
    face_area_ratio: float | None = None  # clipped bbox area / image area
    inter_eye_distance_px: float | None = None  # from detector landmarks, original-image pixels
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    pose_status: str = 'DISABLED'
    # Explicit photometric diagnostics
    sharpness_laplacian_var: float | None = None
    mean_luminance: float | None = None
    underexposure_ratio: float | None = None
    overexposure_ratio: float | None = None
    illumination_asymmetry: float | None = None
    # Gaze estimation
    gaze_pitch: float | None = None
    gaze_yaw: float | None = None
    gaze_status: str = 'DISABLED'

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f'Unknown processing status: {self.status}')
        if self.decision not in DECISIONS:
            raise ValueError(f'Unknown quality decision: {self.decision}')
        if self.pose_status not in POSE_STATUSES:
            raise ValueError(f'Unknown pose status: {self.pose_status}')
        if self.gaze_status not in GAZE_STATUSES:
            raise ValueError(f'Unknown gaze status: {self.gaze_status}')

    @property
    def gate_decision(self) -> str:
        """Deterministic 6-state gate decision: PASS, REVIEW, REJECT, NO_FACE, NOT_EVALUATED, ERROR."""
        if self.status == 'NO_FACE':
            return 'NO_FACE'
        if self.status in ('READ_ERROR', 'ALIGNMENT_ERROR', 'MODEL_ERROR'):
            return 'ERROR'
        if self.decision in ('ACCEPT', 'PASS'):
            return 'PASS'
        if self.decision == 'REVIEW':
            return 'REVIEW'
        if self.decision == 'REJECT':
            return 'REJECT'
        return 'NOT_EVALUATED'

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['gate_decision'] = self.gate_decision
        return data


def _optional_number(config: dict[str, Any], key: str, *, positive: bool = False) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} must be a finite number or null, got {value!r}')
    if positive and value <= 0:
        raise ValueError(f'{key} must be > 0, got {value!r}')
    return float(value)


@dataclass(frozen=True)
class QualityThresholds:
    """Decision thresholds. ``None`` means "not calibrated"; no defaults are invented.

    Score rule: ``score <= reject_max`` -> REJECT, ``score >= accept_min`` -> ACCEPT, otherwise REVIEW.
    Pose limits are absolute angles in degrees; exceeding one yields REVIEW, never an automatic REJECT.
    """

    accept_min: float | None = None
    reject_max: float | None = None
    max_abs_pitch: float | None = None
    max_abs_yaw: float | None = None
    max_abs_roll: float | None = None

    def __post_init__(self) -> None:
        for name in ('accept_min', 'reject_max'):
            _optional_number({name: getattr(self, name)}, name)
        for name in ('max_abs_pitch', 'max_abs_yaw', 'max_abs_roll'):
            _optional_number({name: getattr(self, name)}, name, positive=True)
        if self.accept_min is not None and self.reject_max is not None and self.reject_max >= self.accept_min:
            raise ValueError(
                f'reject_max ({self.reject_max}) must be strictly lower than accept_min ({self.accept_min})'
            )

    @property
    def has_pose_limits(self) -> bool:
        return any(v is not None for v in (self.max_abs_pitch, self.max_abs_yaw, self.max_abs_roll))

    @classmethod
    def from_file(cls, path: str | Path | None) -> 'QualityThresholds':
        if path is None:
            return cls()
        config_path = Path(path).expanduser()
        try:
            with config_path.open(encoding='utf-8') as config_file:
                config = json.load(config_file)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid JSON in threshold file {config_path}: {error}') from error
        if not isinstance(config, dict):
            raise ValueError(f'Threshold file {config_path} must contain a JSON object')
        unknown = set(config) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown threshold keys in {config_path}: {sorted(unknown)}')
        return cls(**{key: config.get(key) for key in (f.name for f in fields(cls))})

    def decide(self, score: float | None) -> tuple[str, str]:
        """Return ``(decision, reason)`` for a quality score."""
        if score is None:
            return 'NOT_EVALUATED', 'no quality score'
        if self.accept_min is None and self.reject_max is None:
            return 'NOT_EVALUATED', 'no calibrated score thresholds configured'
        if self.reject_max is not None and score <= self.reject_max:
            return 'REJECT', f'score {score:.4f} <= reject_max {self.reject_max}'
        if self.accept_min is not None and score >= self.accept_min:
            return 'ACCEPT', f'score {score:.4f} >= accept_min {self.accept_min}'
        return 'REVIEW', f'score {score:.4f} is between the configured thresholds'

    def decision(self, score: float | None) -> str:
        return self.decide(score)[0]

    def pose_violations(self, pitch: float, yaw: float, roll: float) -> list[str]:
        out = []
        for label, value, limit in (
            ('pitch', pitch, self.max_abs_pitch),
            ('yaw', yaw, self.max_abs_yaw),
            ('roll', roll, self.max_abs_roll),
        ):
            if limit is not None and abs(value) > limit:
                out.append(f'|{label}| {abs(value):.1f} > {limit}')
        return out


class QualityPipeline:
    def __init__(
        self,
        *,
        variant: str = 's',
        weights: str | Path | None = None,
        device: str = 'cpu',
        detector: str | None = None,
        fallback_detector: str | None = None,
        detector_config: DetectorConfig | None = None,
        thresholds: QualityThresholds | None = None,
        head_pose: bool = False,
        pose_estimator: Any | None = None,
        gaze: bool = False,
        gaze_estimator: Any | None = None,
    ) -> None:
        config = (detector_config or DetectorConfig()).with_overrides(primary=detector, fallback=fallback_detector)
        self.detection = FaceDetectorChain(config)
        self.device = device
        self.detector_name = config.primary
        self.fallback_detector_name = config.fallback
        self.thresholds = thresholds or QualityThresholds()
        if self.thresholds.has_pose_limits and not (head_pose or pose_estimator is not None):
            raise ValueError('pose limits are configured but head-pose estimation is disabled; enable head_pose')
        weights_path = Path(weights or f'weights/ediffiqa_{variant}.pth')
        if not weights_path.exists():
            try:
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_{variant}.pth'
                torch.hub.download_url_to_file(url, str(weights_path))
            except Exception:
                pass

        try:
            self.model = get_model(variant, return_feat=False)
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(device).eval()
        except Exception as error:
            raise RuntimeError(f'Failed to load eDifFIQA-{variant.upper()} from {weights_path}: {error}') from error

        # Official eDifFIQA preprocessing (matches upstream main.py): RGB, 112x112, (x/255 - 0.5) / 0.5.
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((ALIGNED_SIZE, ALIGNED_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.pose_checker: HeadPoseChecker | None = None
        if head_pose or pose_estimator is not None:
            self.pose_checker = HeadPoseChecker(estimator=pose_estimator)

        self.gaze_checker: GazeChecker | None = None
        if gaze or gaze_estimator is not None:
            self.gaze_checker = GazeChecker(estimator=gaze_estimator)

    @property
    def detectors(self) -> dict[str, Any]:
        """Built detector instances by name (exposed so tests/diagnostics can inject or inspect them)."""
        return self.detection.detectors

    @detectors.setter
    def detectors(self, value: dict[str, Any]) -> None:
        self.detection.detectors = value

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def _check_aligned(aligned: np.ndarray) -> None:
        """Raise ValueError unless ``aligned`` satisfies the eDifFIQA input requirements."""
        if not isinstance(aligned, np.ndarray) or aligned.dtype != np.uint8 or aligned.ndim != 3 or aligned.shape[2] != 3:
            raise ValueError('aligned face must be a 3-channel uint8 BGR image')
        if aligned.shape[:2] != (ALIGNED_SIZE, ALIGNED_SIZE):
            raise ValueError(
                f'aligned face must be exactly {ALIGNED_SIZE}x{ALIGNED_SIZE}, got {aligned.shape[1]}x{aligned.shape[0]}'
            )
        if int(aligned.max()) == int(aligned.min()):
            raise ValueError('aligned face is blank (constant pixel value)')

    def _score_aligned(self, aligned: np.ndarray) -> float:
        """Return the raw eDifFIQA output for an aligned 112x112 BGR face. Not a probability."""
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            score = self.model(tensor).squeeze().cpu().item()
        if not np.isfinite(score):
            raise ValueError(f'model returned non-finite score: {score}')
        return float(score)

    score_aligned = _score_aligned

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, image_path: str | Path, *, aligned: bool = False) -> list[EvaluationResult]:
        started = time.perf_counter()
        display_path = str(image_path)
        path = Path(image_path).expanduser()
        label = 'aligned' if aligned else self.detector_name
        if not path.exists():
            return [self._result(display_path, label, None, None, None, 'READ_ERROR', f'path does not exist: {path}', started)]
        if not path.is_file():
            return [self._result(display_path, label, None, None, None, 'READ_ERROR', f'path is not a file: {path}', started)]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return [self._result(display_path, label, None, None, None, 'READ_ERROR', f'could not decode image: {path}', started)]
        return self.evaluate_array(image, name=display_path, aligned=aligned, started=started)

    def evaluate_array(
        self, image: np.ndarray, *, name: str, aligned: bool = False, started: float | None = None
    ) -> list[EvaluationResult]:
        """Evaluate an already-decoded BGR uint8 image (used for in-memory degraded variants)."""
        started = time.perf_counter() if started is None else started
        height, width = image.shape[:2]

        if aligned:
            try:
                self._check_aligned(image)
            except ValueError as error:
                return [self._result(name, 'aligned', width, height, 1, 'ALIGNMENT_ERROR', str(error), started, pose_status='NOT_APPLICABLE', gaze_status='NOT_APPLICABLE')]
            return [self._score_result(name, width, height, 'aligned', 1, image, started, pose_status='NOT_APPLICABLE', gaze_status='NOT_APPLICABLE')]

        outcome: DetectionOutcome = self.detection.detect(image)
        detection_ms = (time.perf_counter() - started) * 1000
        shared = dict(fallback_used=outcome.fallback_used, detection_note=outcome.note, detection_time_ms=detection_ms)
        if outcome.error:
            return [self._result(name, outcome.detector, width, height, None, 'MODEL_ERROR', outcome.error, started, **shared)]
        if not outcome.faces:
            fallback_text = f'; fallback={self.fallback_detector_name}' if self.fallback_detector_name else ''
            reason = f'no usable face detected by {self.detector_name}{fallback_text} ({outcome.note})'
            return [self._result(name, outcome.detector, width, height, None, 'NO_FACE', reason, started, **shared)]

        results = []
        for index, face in enumerate(outcome.faces, start=1):
            face_started = time.perf_counter()
            bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
            landmarks = np.asarray(face.landmarks, dtype=np.float32)
            geometry = dict(
                confidence=float(face.confidence),
                bbox=bbox,
                face_area_ratio=self._area_ratio(bbox, width, height),
                inter_eye_distance_px=float(np.linalg.norm(landmarks[1] - landmarks[0])),
                **shared,
            )
            try:
                aligned_face, inverse = face_alignment(image, landmarks)
                if aligned_face is None or aligned_face.size == 0:
                    raise ValueError('alignment returned an empty image')
                if aligned_face.shape != (ALIGNED_SIZE, ALIGNED_SIZE, 3) or not np.isfinite(inverse).all():
                    raise ValueError('alignment produced an invalid transform or output size')
                self._check_aligned(aligned_face)
            except Exception as error:
                elapsed = detection_ms + (time.perf_counter() - face_started) * 1000
                results.append(self._result(name, outcome.detector, width, height, index, 'ALIGNMENT_ERROR', str(error), None, elapsed_ms=elapsed, **geometry))
                continue
            results.append(
                self._score_result(
                    name, width, height, outcome.detector, index, aligned_face, face_started,
                    image=image, offset_ms=detection_ms, **geometry,
                )
            )
        return results

    @staticmethod
    def _area_ratio(bbox: list[float], width: int, height: int) -> float:
        x1, y1, x2, y2 = bbox
        w = max(0.0, min(x2, width) - max(x1, 0.0))
        h = max(0.0, min(y2, height) - max(y1, 0.0))
        return float(w * h / (width * height))

    def _score_result(
        self,
        name: str,
        width: int,
        height: int,
        detector: str,
        face_index: int,
        aligned: np.ndarray,
        started: float,
        *,
        image: np.ndarray | None = None,
        offset_ms: float = 0.0,
        pose_status: str | None = None,
        gaze_status: str | None = None,
        **extra: Any,
    ) -> EvaluationResult:
        def elapsed() -> float:
            return offset_ms + (time.perf_counter() - started) * 1000

        t_score_start = time.perf_counter()
        try:
            score = self._score_aligned(aligned)
            quality_ms = (time.perf_counter() - t_score_start) * 1000
        except Exception as error:
            return self._result(
                name, detector, width, height, face_index, 'MODEL_ERROR', str(error), None,
                elapsed_ms=elapsed(), pose_status=pose_status, gaze_status=gaze_status, **extra,
            )

        # Extract face crop from the original image (or use aligned as fallback)
        crop: np.ndarray | None = None
        if image is not None and 'bbox' in extra and extra['bbox'] is not None:
            try:
                h_img, w_img = image.shape[:2]
                bx1, by1, bx2, by2 = [int(round(float(v))) for v in extra['bbox']]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w_img, bx2), min(h_img, by2)
                if bx2 - bx1 >= 2 and by2 - by1 >= 2:
                    crop = image[by1:by2, bx1:bx2]
            except Exception:
                crop = None
        target_crop = crop if (crop is not None and crop.size > 0) else aligned

        # Photometric & sharpness diagnostics on the face crop
        t_photo_start = time.perf_counter()
        photo_diag = analyze_photometrics(target_crop)
        photo_ms = (time.perf_counter() - t_photo_start) * 1000

        angles: PoseAngles | None = None
        pose_error: str | None = None
        pose_ms: float | None = None
        if pose_status is None:
            pose_status = 'DISABLED'
            if self.pose_checker is not None and image is not None and 'bbox' in extra:
                t_pose_start = time.perf_counter()
                try:
                    angles = self.pose_checker.estimate(image, extra['bbox'])
                    pose_status = 'OK'
                except Exception as error:  # pose failure must not discard a valid quality score
                    pose_status, pose_error = 'ERROR', str(error)
                pose_ms = (time.perf_counter() - t_pose_start) * 1000

        gaze_angles: GazeAngles | None = None
        gaze_ms: float | None = None
        if gaze_status is None:
            gaze_status = 'DISABLED'
            if self.gaze_checker is not None and target_crop is not None and target_crop.size > 0:
                t_gaze_start = time.perf_counter()
                try:
                    gaze_angles = self.gaze_checker.estimate(target_crop)
                    gaze_status = 'OK'
                except Exception:
                    gaze_status = 'ERROR'
                gaze_ms = (time.perf_counter() - t_gaze_start) * 1000

        return self._result(
            name, detector, width, height, face_index, 'OK', None, None,
            elapsed_ms=elapsed(), quality_score=score,
            quality_time_ms=quality_ms,
            pose_time_ms=pose_ms,
            gaze_time_ms=gaze_ms,
            photometrics_time_ms=photo_ms,
            pose_status=pose_status, angles=angles, pose_error=pose_error,
            gaze_status=gaze_status, gaze_angles=gaze_angles,
            sharpness_laplacian_var=photo_diag.sharpness_laplacian_var,
            mean_luminance=photo_diag.mean_luminance,
            underexposure_ratio=photo_diag.underexposure_ratio,
            overexposure_ratio=photo_diag.overexposure_ratio,
            illumination_asymmetry=photo_diag.illumination_asymmetry,
            **extra,
        )

    def _final_decision(
        self, status: str, score: float | None, pose_status: str, angles: PoseAngles | None, pose_error: str | None
    ) -> tuple[str, str]:
        if status != 'OK':
            return 'NOT_EVALUATED', f'not evaluated: processing status {status}'
        decision, reason = self.thresholds.decide(score)
        if not self.thresholds.has_pose_limits or pose_status == 'NOT_APPLICABLE':
            return decision, reason
        flags: list[str] = []
        if pose_status == 'ERROR':
            flags.append(f'head pose could not be verified ({pose_error})')
        elif angles is not None:
            flags.extend(self.thresholds.pose_violations(angles.pitch, angles.yaw, angles.roll))
        if flags:
            if decision in ('ACCEPT', 'NOT_EVALUATED'):
                decision = 'REVIEW'
            reason = f'{reason}; pose: {"; ".join(flags)}'
        return decision, reason

    def _result(
        self,
        image: str,
        detector: str,
        width: int | None,
        height: int | None,
        face_index: int | None,
        status: str,
        error_reason: str | None,
        started: float | None,
        *,
        elapsed_ms: float | None = None,
        confidence: float | None = None,
        bbox: list[float] | None = None,
        quality_score: float | None = None,
        quality_time_ms: float | None = None,
        pose_time_ms: float | None = None,
        gaze_time_ms: float | None = None,
        photometrics_time_ms: float | None = None,
        pose_status: str | None = None,
        angles: PoseAngles | None = None,
        pose_error: str | None = None,
        gaze_status: str | None = None,
        gaze_angles: GazeAngles | None = None,
        gaze_pitch: float | None = None,
        gaze_yaw: float | None = None,
        sharpness_laplacian_var: float | None = None,
        mean_luminance: float | None = None,
        underexposure_ratio: float | None = None,
        overexposure_ratio: float | None = None,
        illumination_asymmetry: float | None = None,
        **extra: Any,
    ) -> EvaluationResult:
        if pose_status is None:
            pose_status = 'DISABLED' if self.pose_checker is None else 'SKIPPED'
        if gaze_status is None:
            gaze_status = 'DISABLED' if self.gaze_checker is None else 'SKIPPED'
        if elapsed_ms is None:
            elapsed_ms = (time.perf_counter() - started) * 1000 if started is not None else 0.0
        decision, decision_reason = self._final_decision(status, quality_score, pose_status, angles, pose_error)
        if gaze_angles is not None:
            gaze_pitch = gaze_angles.pitch
            gaze_yaw = gaze_angles.yaw
        return EvaluationResult(
            image=image,
            image_width=width,
            image_height=height,
            detector=detector,
            face_index=face_index,
            confidence=confidence,
            bbox=bbox,
            quality_score=quality_score,
            status=status,
            decision=decision,
            error_reason=error_reason,
            processing_time_ms=elapsed_ms,
            decision_reason=decision_reason,
            quality_time_ms=quality_time_ms,
            pose_time_ms=pose_time_ms,
            gaze_time_ms=gaze_time_ms,
            photometrics_time_ms=photometrics_time_ms,
            pitch=None if angles is None else angles.pitch,
            yaw=None if angles is None else angles.yaw,
            roll=None if angles is None else angles.roll,
            pose_status=pose_status,
            sharpness_laplacian_var=sharpness_laplacian_var,
            mean_luminance=mean_luminance,
            underexposure_ratio=underexposure_ratio,
            overexposure_ratio=overexposure_ratio,
            illumination_asymmetry=illumination_asymmetry,
            gaze_pitch=gaze_pitch,
            gaze_yaw=gaze_yaw,
            gaze_status=gaze_status,
            **extra,
        )
