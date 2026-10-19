from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    model_loaded: bool
    model_variant: str
    supported_detectors: list[str]
    timestamp: str


class DetectorSettingsSchema(BaseModel):
    confidence_threshold: float
    nms_threshold: float
    input_size: list[int]
    model_name: str | None = None


class ConfigResponse(BaseModel):
    primary_detector: str
    fallback_detector: str | None = None
    detection_mode: str = "fallback"
    scrfd: DetectorSettingsSchema
    retinaface: DetectorSettingsSchema
    head_pose_enabled: bool
    gaze_enabled: bool = False
    thresholds: dict[str, float | None]
    is_calibrated: bool


class ThresholdsUpdateRequest(BaseModel):
    accept_min: float | None = None
    reject_max: float | None = None
    max_abs_pitch: float | None = None
    max_abs_yaw: float | None = None
    max_abs_roll: float | None = None


class FaceAnalysisResult(BaseModel):
    request_id: str
    image_name: str
    image_width: int | None = None
    image_height: int | None = None
    detector: str
    detection_mode: str = "fallback"
    fallback_used: bool = False
    face_index: int | None = None
    confidence: float | None = None
    bbox: list[float] | None = None
    quality_score: float | None = None
    status: str  # OK, NO_FACE, READ_ERROR, ALIGNMENT_ERROR, MODEL_ERROR
    decision: str  # ACCEPT, REJECT, REVIEW, NOT_EVALUATED
    gate_decision: str = "NOT_EVALUATED"  # PASS, REVIEW, REJECT, NO_FACE, NOT_EVALUATED, ERROR
    effective_decision: str = "NOT_EVALUATED"  # Result after review_policy
    review_policy: str = "manual_review"  # manual_review, treat_as_pass, treat_as_reject
    policy_override: bool = False
    detection_status: str = "DETECTED"  # DETECTED, NO_FACE, ERROR
    decision_reason: str | None = None
    error_reason: str | None = None
    detection_note: str | None = None
    detection_time_ms: float | None = None
    quality_time_ms: float | None = None
    pose_time_ms: float | None = None
    gaze_time_ms: float | None = None
    photometrics_time_ms: float | None = None
    processing_time_ms: float
    face_area_ratio: float | None = None
    inter_eye_distance_px: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    pose_status: str = "DISABLED"
    sharpness_laplacian_var: float | None = None
    mean_luminance: float | None = None
    underexposure_ratio: float | None = None
    overexposure_ratio: float | None = None
    illumination_asymmetry: float | None = None
    gaze_pitch: float | None = None
    gaze_yaw: float | None = None
    gaze_status: str = "DISABLED"
    face_crop_base64: str | None = None


class BatchAnalysisResult(BaseModel):
    batch_id: str
    total_images: int
    processed_images: int
    successful_evaluations: int
    failures: int
    pass_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    no_face_count: int = 0
    error_count: int = 0
    total_faces: int = 0
    review_policy: str = "manual_review"
    results: list[FaceAnalysisResult]
    created_at: str


class SessionStatsResponse(BaseModel):
    session_start_time: str
    total_processed: int
    faces_detected: int
    detection_failures: int
    quality_decisions_completed: int
    not_evaluated_count: int
    current_config: dict[str, Any]
