"""Configurable face detection with an optional RetinaFace fallback.

Detector APIs used here were verified against UniFace 4.0.0:

* ``uniface.detection.SCRFD(model_name, confidence_threshold, nms_threshold, input_size, providers)``
* ``uniface.detection.RetinaFace(model_name, confidence_threshold, nms_threshold, input_size, providers, ...)``
* ``detector.detect(bgr_uint8_image) -> list[Face]`` where ``Face`` has ``bbox`` (4,) as
  ``[x1, y1, x2, y2]``, ``confidence`` (float) and ``landmarks`` (5, 2).

Nothing in this module lowers a confidence threshold on its own: defaults equal the UniFace defaults.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

DETECTOR_NAMES = ('scrfd', 'retinaface')
DETECTION_MODES = ('fallback', 'complementary')


def _as_size(value: Any, name: str) -> tuple[int, int]:
    """Accept ``640`` or ``[640, 640]``; UniFace expects ``(width, height)``."""
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an int or a [width, height] pair')
    if isinstance(value, int):
        value = (value, value)
    try:
        width, height = (int(v) for v in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be an int or a [width, height] pair') from error
    if width <= 0 or height <= 0:
        raise ValueError(f'{name} must be positive, got {(width, height)}')
    return width, height


@dataclass(frozen=True)
class DetectorSettings:
    """Per-detector settings. Defaults equal UniFace 4.0.0 defaults."""

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: tuple[int, int] = (640, 640)
    model_name: str | None = None  # None -> UniFace default (SCRFD-10G / RetinaFace MNET_V2)

    def __post_init__(self) -> None:
        for key in ('confidence_threshold', 'nms_threshold'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{key} must be a number in [0, 1], got {value!r}')
        object.__setattr__(self, 'input_size', _as_size(self.input_size, 'input_size'))


@dataclass(frozen=True)
class DetectorConfig:
    """Primary/fallback selection plus per-detector settings."""

    primary: str = 'scrfd'
    fallback: str | None = None
    mode: str = 'fallback'
    scrfd: DetectorSettings = field(default_factory=DetectorSettings)
    retinaface: DetectorSettings = field(default_factory=DetectorSettings)
    dedupe_iou: float = 0.5
    providers: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.primary not in DETECTOR_NAMES:
            raise ValueError(f'primary detector must be one of {DETECTOR_NAMES}, got {self.primary!r}')
        if self.fallback is not None and self.fallback not in DETECTOR_NAMES:
            raise ValueError(f'fallback detector must be None or one of {DETECTOR_NAMES}, got {self.fallback!r}')
        if self.fallback == self.primary:
            raise ValueError('fallback detector must differ from the primary detector')
        if self.mode not in DETECTION_MODES:
            raise ValueError(f'mode must be one of {DETECTION_MODES}, got {self.mode!r}')
        if not 0.0 < self.dedupe_iou <= 1.0:
            raise ValueError(f'dedupe_iou must be in (0, 1], got {self.dedupe_iou!r}')

    def settings_for(self, name: str) -> DetectorSettings:
        return self.scrfd if name == 'scrfd' else self.retinaface

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DetectorConfig':
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f'Unknown detector config keys: {sorted(unknown)}')
        kwargs: dict[str, Any] = dict(data)
        for name in DETECTOR_NAMES:
            if name in kwargs and isinstance(kwargs[name], dict):
                allowed_s = {f.name for f in fields(DetectorSettings)}
                bad = set(kwargs[name]) - allowed_s
                if bad:
                    raise ValueError(f'Unknown keys in detector config "{name}": {sorted(bad)}')
                kwargs[name] = DetectorSettings(**kwargs[name])
        if kwargs.get('providers') is not None:
            kwargs['providers'] = tuple(kwargs['providers'])
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path | None) -> 'DetectorConfig':
        if path is None:
            return cls()
        with Path(path).expanduser().open(encoding='utf-8') as config_file:
            return cls.from_dict(json.load(config_file))

    def with_overrides(self, **overrides: Any) -> 'DetectorConfig':
        """Return a copy with top-level fields replaced (``None`` values are ignored)."""
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return DetectorConfig(**values)


def build_detector(name: str, config: DetectorConfig) -> Any:
    """Construct a UniFace detector from the configuration (weights are loaded once here)."""
    from uniface import constants
    from uniface.detection import SCRFD, RetinaFace

    settings = config.settings_for(name)
    kwargs: dict[str, Any] = {
        'confidence_threshold': settings.confidence_threshold,
        'nms_threshold': settings.nms_threshold,
        'input_size': settings.input_size,
    }
    if config.providers is not None:
        kwargs['providers'] = list(config.providers)
    if name == 'scrfd':
        if settings.model_name is not None:
            kwargs['model_name'] = constants.SCRFDWeights(settings.model_name)
        return SCRFD(**kwargs)
    if settings.model_name is not None:
        kwargs['model_name'] = constants.RetinaFaceWeights(settings.model_name)
    return RetinaFace(**kwargs)


# --------------------------------------------------------------------------- validation


def validate_face(face: Any, width: int, height: int) -> str | None:
    """Return ``None`` if the detection is structurally usable, otherwise the reason it is not.

    Only parameter-free structural checks are used, so no tuned threshold is hidden here.
    """
    try:
        bbox = np.asarray(face.bbox, dtype=np.float64)
        landmarks = np.asarray(face.landmarks, dtype=np.float64)
        confidence = float(face.confidence)
    except Exception as error:  # missing attribute / non-numeric content
        return f'detection is malformed: {error}'

    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return f'bbox must be 4 finite numbers, got shape {bbox.shape}'
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return f'bbox has non-positive size: {bbox.tolist()}'
    if min(x2, width) - max(x1, 0.0) <= 0 or min(y2, height) - max(y1, 0.0) <= 0:
        return f'bbox lies outside the {width}x{height} image: {bbox.tolist()}'
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return f'confidence is not in [0, 1]: {confidence}'
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        return f'landmarks must be a finite (5, 2) array, got shape {landmarks.shape}'
    if (np.ptp(landmarks, axis=0) < 1e-6).any():
        return 'landmarks are degenerate (all points share one x or y coordinate)'
    cx, cy = landmarks.mean(axis=0)
    tol_x = max(2.0, (x2 - x1) * 0.05)
    tol_y = max(2.0, (y2 - y1) * 0.05)
    if not (x1 - tol_x <= cx <= x2 + tol_x and y1 - tol_y <= cy <= y2 + tol_y):
        return 'landmark centroid lies outside the bounding box'
    return None


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe_faces(faces: list[Any], iou_threshold: float) -> tuple[list[Any], int]:
    """Drop overlapping detections, keeping the highest-confidence one. Returns (kept, n_dropped)."""
    kept: list[Any] = []
    for face in sorted(faces, key=lambda f: float(f.confidence), reverse=True):
        box = np.asarray(face.bbox, dtype=np.float64)
        if all(bbox_iou(box, np.asarray(k.bbox, dtype=np.float64)) <= iou_threshold for k in kept):
            kept.append(face)
    return kept, len(faces) - len(kept)


# --------------------------------------------------------------------------- detector chain


@dataclass
class DetectionOutcome:
    detector: str  # label recorded in results, e.g. 'scrfd' or 'retinaface(fallback)'
    faces: list[Any]
    error: str | None = None  # set only when detection could not run at all
    fallback_used: bool = False
    note: str | None = None  # per-detector counts and rejection reasons, for NO_FACE explanations


class FaceDetectorChain:
    """Run the primary detector; run the fallback only if the primary yields no usable face."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.detectors: dict[str, Any] = {}  # built lazily, then reused

    def _get(self, name: str) -> Any:
        if name not in self.detectors:
            self.detectors[name] = build_detector(name, self.config)
        return self.detectors[name]

    def _run(self, name: str, image: np.ndarray) -> tuple[list[Any], str]:
        """Detect, validate and dedupe with one detector. Raises if the detector itself fails."""
        height, width = image.shape[:2]
        raw = list(self._get(name).detect(image))
        valid, rejected = [], []
        for face in raw:
            reason = validate_face(face, width, height)
            (rejected if reason else valid).append(reason or face)
        valid, duplicates = dedupe_faces(valid, self.config.dedupe_iou)
        parts = [f'{len(raw)} raw']
        if rejected:
            parts.append(f'{len(rejected)} invalid ({"; ".join(sorted(set(rejected)))})')
        if duplicates:
            parts.append(f'{duplicates} duplicate')
        parts.append(f'{len(valid)} usable')
        return valid, f'{name}: ' + ', '.join(parts)

    def detect(self, image: np.ndarray) -> DetectionOutcome:
        primary = self.config.primary
        notes: list[str] = []
        primary_error = None

        if self.config.mode == 'complementary' and self.config.fallback is not None:
            fallback = self.config.fallback
            primary_faces: list[Any] = []
            try:
                primary_faces, note = self._run(primary, image)
                notes.append(note)
            except Exception as error:
                primary_error = f'{primary} detection failed: {error}'
                notes.append(primary_error)

            fallback_faces: list[Any] = []
            fallback_error = None
            try:
                fallback_faces, note = self._run(fallback, image)
                notes.append(note)
            except Exception as error:
                fallback_error = f'{fallback} detection failed: {error}'
                notes.append(fallback_error)

            if primary_error and fallback_error:
                combined = f'{primary_error}; {fallback_error}'
                return DetectionOutcome(f'{primary}+{fallback}', [], error=combined, fallback_used=True, note='; '.join(notes))

            combined_faces = primary_faces + fallback_faces
            merged_faces, duplicates = dedupe_faces(combined_faces, self.config.dedupe_iou)
            label = f'{primary}+{fallback}(complementary)'
            notes.append(
                f'complementary merge: {len(primary_faces)} {primary} + {len(fallback_faces)} {fallback} '
                f'-> {len(merged_faces)} usable ({duplicates} cross-detector overlap deduplicated)'
            )
            return DetectionOutcome(label, merged_faces, fallback_used=True, note='; '.join(notes))

        # Standard fallback mode
        try:
            faces, note = self._run(primary, image)
            notes.append(note)
        except Exception as error:
            faces = []
            primary_error = f'{primary} detection failed: {error}'
            notes.append(primary_error)
        if faces:
            return DetectionOutcome(primary, faces, note='; '.join(notes))
        if self.config.fallback is None:
            return DetectionOutcome(primary, [], error=primary_error, note='; '.join(notes))

        fallback = self.config.fallback
        label = f'{fallback}(fallback)'
        try:
            faces, note = self._run(fallback, image)
            notes.append(note)
        except Exception as error:
            fallback_error = f'{fallback} fallback detection failed: {error}'
            notes.append(fallback_error)
            combined = f'{primary_error}; {fallback_error}' if primary_error else fallback_error
            # A crashed fallback means "no face" cannot be claimed, so surface it as an error.
            return DetectionOutcome(label, [], error=combined, fallback_used=True, note='; '.join(notes))
        return DetectionOutcome(label, faces, fallback_used=True, note='; '.join(notes))
