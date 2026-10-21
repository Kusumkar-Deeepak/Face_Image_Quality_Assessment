from __future__ import annotations

import base64
import csv
import io
import gc
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PilImage, ImageOps as PilImageOps

# Restrict thread pools to avoid memory blow-up on shared CPU cloud containers (Render 512MB)
if hasattr(torch, "set_num_threads"):
    try:
        torch.set_num_threads(min(2, os.cpu_count() or 1))
    except Exception:
        pass

# Safety limits: Decompression bomb and extreme dimension protection
PilImage.MAX_IMAGE_PIXELS = 100_000_000
MAX_IMAGE_DIMENSION = 16384

# Ensure repository root is on sys.path so existing modules import cleanly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from face_detection import DetectorConfig, DetectorSettings
from quality_pipeline import EvaluationResult, QualityPipeline, QualityThresholds
from api.schemas import (
    BatchAnalysisResult,
    ConfigResponse,
    DetectorSettingsSchema,
    FaceAnalysisResult,
    HealthResponse,
    SessionStatsResponse,
)

logger = logging.getLogger("face_quality_lab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def release_process_memory() -> None:
    """Trigger Python garbage collection and prompt Linux glibc to release free memory to OS."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class PipelineService:
    """Manages the eDifFIQA inference pipeline lifecycle, model reuse, session stats,

    and batch exports.
    """

    def __init__(self, device: str | None = None) -> None:
        self.lock = threading.Lock()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = ROOT_DIR / "weights" / "ediffiqa_s.pth"
        self.detector_config_path = ROOT_DIR / "detector_config.json"
        self.thresholds_path = ROOT_DIR / "quality_thresholds.json"

        # Session metrics
        self.session_start = datetime.now(timezone.utc)
        self.total_processed = 0
        self.faces_detected = 0
        self.detection_failures = 0
        self.decisions_completed = 0
        self.not_evaluated_count = 0

        # Caching & Batches
        self._shared_model: Any | None = None
        self._shared_pose_checker: Any | None = None
        self._shared_gaze_checker: Any | None = None
        self._shared_detectors: dict[str, Any] = {}
        self._pipeline_cache: dict[tuple, QualityPipeline] = {}
        self._batches: OrderedDict[str, BatchAnalysisResult] = OrderedDict()
        self._max_batches = 100

        # Current default settings
        self.current_primary = "scrfd"
        self.current_fallback: str | None = None
        self.current_scrfd_input_size = 640
        self.current_head_pose = False
        self.current_gaze = False

    def initialize(self) -> None:
        """Pre-warm the model and verify weights."""
        logger.info(f"Initializing PipelineService on device: {self.device}")
        try:
            # Pre-load default pipeline
            pipeline = self.get_pipeline(
                detector="scrfd",
                fallback=None,
                scrfd_input_size=640,
                head_pose=False,
            )
            logger.info("Default QualityPipeline initialized successfully.")
            release_process_memory()
        except Exception as exc:
            logger.error(f"Error during pre-warming pipeline: {exc}", exc_info=True)

    def get_health(self) -> HealthResponse:
        has_weights = self.weights_path.exists()
        model_ready = self._shared_model is not None or has_weights
        return HealthResponse(
            status="ok" if model_ready else "degraded",
            device=self.device,
            cuda_available=torch.cuda.is_available(),
            model_loaded=self._shared_model is not None,
            model_variant="s",
            supported_detectors=["scrfd", "retinaface"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_config(self) -> ConfigResponse:
        det_config = DetectorConfig.from_file(
            self.detector_config_path if self.detector_config_path.exists() else None
        )
        thresholds = QualityThresholds.from_file(
            self.thresholds_path if self.thresholds_path.exists() else None
        )
        is_calibrated = (
            thresholds.accept_min is not None or thresholds.reject_max is not None
        )

        return ConfigResponse(
            primary_detector=self.current_primary,
            fallback_detector=self.current_fallback,
            detection_mode=det_config.mode,
            scrfd=DetectorSettingsSchema(
                confidence_threshold=det_config.scrfd.confidence_threshold,
                nms_threshold=det_config.scrfd.nms_threshold,
                input_size=list(det_config.scrfd.input_size),
                model_name=det_config.scrfd.model_name,
            ),
            retinaface=DetectorSettingsSchema(
                confidence_threshold=det_config.retinaface.confidence_threshold,
                nms_threshold=det_config.retinaface.nms_threshold,
                input_size=list(det_config.retinaface.input_size),
                model_name=det_config.retinaface.model_name,
            ),
            head_pose_enabled=self.current_head_pose,
            gaze_enabled=self.current_gaze,
            thresholds={
                "accept_min": thresholds.accept_min,
                "reject_max": thresholds.reject_max,
                "max_abs_pitch": thresholds.max_abs_pitch,
                "max_abs_yaw": thresholds.max_abs_yaw,
                "max_abs_roll": thresholds.max_abs_roll,
            },
            is_calibrated=is_calibrated,
        )

    def update_thresholds(self, req: Any) -> ConfigResponse:
        with self.lock:
            data = req.model_dump()
            # Validate through QualityThresholds dataclass (ensures reject_max < accept_min)
            new_thresholds = QualityThresholds(**data)
            
            with self.thresholds_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Invalidate cached pipelines so new runs use updated thresholds immediately
            self._pipeline_cache.clear()
            logger.info(
                f"Updated quality thresholds: accept_min={new_thresholds.accept_min}, "
                f"reject_max={new_thresholds.reject_max}"
            )
            return self.get_config()

    def get_session_stats(self) -> SessionStatsResponse:
        with self.lock:
            return SessionStatsResponse(
                session_start_time=self.session_start.isoformat(),
                total_processed=self.total_processed,
                faces_detected=self.faces_detected,
                detection_failures=self.detection_failures,
                quality_decisions_completed=self.decisions_completed,
                not_evaluated_count=self.not_evaluated_count,
                current_config={
                    "primary_detector": self.current_primary,
                    "fallback_detector": self.current_fallback,
                    "scrfd_input_size": self.current_scrfd_input_size,
                    "head_pose": self.current_head_pose,
                    "gaze": self.current_gaze,
                },
            )

    def get_pipeline(
        self,
        detector: str | None = None,
        fallback: str | None = None,
        scrfd_input_size: int | None = None,
        head_pose: bool = False,
        gaze: bool = False,
        detection_mode: str = "fallback",
        confidence_threshold: float | None = None,
    ) -> QualityPipeline:
        """Retrieves or creates a QualityPipeline instance, reusing loaded model weights."""
        detector = detector or self.current_primary
        fallback = fallback if fallback is not None else self.current_fallback
        if fallback == "none" or fallback == "":
            fallback = None
        scrfd_input_size = scrfd_input_size or self.current_scrfd_input_size

        cache_key = (
            detector,
            fallback,
            scrfd_input_size,
            head_pose,
            gaze,
            detection_mode,
            confidence_threshold,
        )

        with self.lock:
            if cache_key in self._pipeline_cache:
                return self._pipeline_cache[cache_key]

            # Build detector configuration
            base_config = DetectorConfig.from_file(
                self.detector_config_path if self.detector_config_path.exists() else None
            )
            scrfd_settings = base_config.scrfd
            if scrfd_input_size is not None:
                scrfd_settings = replace(
                    scrfd_settings, input_size=(scrfd_input_size, scrfd_input_size)
                )
            if confidence_threshold is not None:
                scrfd_settings = replace(
                    scrfd_settings, confidence_threshold=confidence_threshold
                )

            retinaface_settings = base_config.retinaface
            if confidence_threshold is not None:
                retinaface_settings = replace(
                    retinaface_settings, confidence_threshold=confidence_threshold
                )

            detector_config = replace(
                base_config,
                primary=detector,
                fallback=fallback,
                mode=detection_mode,
                scrfd=scrfd_settings,
                retinaface=retinaface_settings,
            )

            thresholds = QualityThresholds.from_file(
                self.thresholds_path if self.thresholds_path.exists() else None
            )
            if thresholds.has_pose_limits and not head_pose:
                thresholds = QualityThresholds(
                    accept_min=thresholds.accept_min,
                    reject_max=thresholds.reject_max,
                    max_abs_pitch=None,
                    max_abs_yaw=None,
                    max_abs_roll=None,
                )

            pipeline = QualityPipeline(
                variant="s",
                weights=self.weights_path,
                device=self.device,
                detector=detector,
                fallback_detector=fallback,
                detector_config=detector_config,
                thresholds=thresholds,
                head_pose=head_pose,
                pose_estimator=self._shared_pose_checker,
                gaze=gaze,
                gaze_estimator=self._shared_gaze_checker,
                model=self._shared_model,
                detectors=self._shared_detectors,
            )

            # Store reference to shared models & detectors
            if self._shared_model is None:
                self._shared_model = pipeline.model

            if head_pose and pipeline.pose_checker is not None:
                if self._shared_pose_checker is None:
                    self._shared_pose_checker = pipeline.pose_checker.estimator

            if gaze and pipeline.gaze_checker is not None:
                if self._shared_gaze_checker is None:
                    self._shared_gaze_checker = pipeline.gaze_checker.estimator

            self._pipeline_cache[cache_key] = pipeline
            return pipeline

    @staticmethod
    def extract_crop_base64(image: np.ndarray, bbox: list[float] | None) -> str | None:
        """Extract face bounding box crop and return base64 JPEG data URL."""
        if bbox is None or image is None or image.size == 0:
            return None
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(round(float(c))) for c in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            success, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not success:
                return None
            b64_str = base64.b64encode(buffer).decode("ascii")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception as exc:
            logger.warning(f"Failed to generate face crop: {exc}")
            return None

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        filename: str,
        detector: str | None = None,
        fallback_detector: str | None = None,
        scrfd_input_size: int | None = None,
        head_pose: bool = False,
        gaze: bool = False,
        detection_mode: str = "fallback",
        confidence_threshold: float | None = None,
        review_policy: str = "manual_review",
    ) -> list[FaceAnalysisResult]:
        """Runs the pipeline on an image in-memory with decompression protection and configurable review policy."""
        request_id = str(uuid.uuid4())
        started_ms = time.perf_counter()

        pipeline = self.get_pipeline(
            detector=detector,
            fallback=fallback_detector,
            scrfd_input_size=scrfd_input_size,
            head_pose=head_pose,
            gaze=gaze,
            detection_mode=detection_mode,
            confidence_threshold=confidence_threshold,
        )

        image = None
        decode_error = None
        try:
            with PilImage.open(io.BytesIO(image_bytes)) as pil_img:
                if (
                    pil_img.width > MAX_IMAGE_DIMENSION
                    or pil_img.height > MAX_IMAGE_DIMENSION
                    or (pil_img.width * pil_img.height > 100_000_000)
                ):
                    decode_error = (
                        f"Image dimensions ({pil_img.width}x{pil_img.height}) exceed safe limit "
                        f"(max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} or 100MP)"
                    )
                else:
                    transposed = PilImageOps.exif_transpose(pil_img)
                    # Safe downscale for memory conservation (caps uncompressed RAM spikes on 512MB containers)
                    MAX_PROCESS_DIM = 1920
                    if transposed.width > MAX_PROCESS_DIM or transposed.height > MAX_PROCESS_DIM:
                        transposed.thumbnail((MAX_PROCESS_DIM, MAX_PROCESS_DIM), PilImage.Resampling.LANCZOS)
                    rgb = np.array(transposed.convert("RGB"))
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    del rgb
        except PilImage.DecompressionBombError as bomb_err:
            decode_error = f"Decompression bomb detected: {str(bomb_err)}"
        except Exception:
            try:
                arr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                del arr
                if image is not None:
                    ih, iw = image.shape[:2]
                    if ih > MAX_IMAGE_DIMENSION or iw > MAX_IMAGE_DIMENSION or (ih * iw > 100_000_000):
                        decode_error = f"Image dimensions ({iw}x{ih}) exceed safe limit"
                        image = None
                    elif ih > 1920 or iw > 1920:
                        scale = 1920.0 / max(ih, iw)
                        new_w, new_h = int(iw * scale), int(ih * scale)
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception as exc:
                decode_error = str(exc)

        if image is None:
            elapsed = (time.perf_counter() - started_ms) * 1000
            err_result = FaceAnalysisResult(
                request_id=request_id,
                image_name=filename,
                image_width=None,
                image_height=None,
                detector=detector or self.current_primary,
                detection_mode=detection_mode,
                fallback_used=False,
                face_index=None,
                confidence=None,
                bbox=None,
                quality_score=None,
                status="READ_ERROR",
                decision="NOT_EVALUATED",
                gate_decision="ERROR",
                effective_decision="ERROR",
                review_policy=review_policy,
                policy_override=False,
                detection_status="ERROR",
                decision_reason="Image could not be decoded or exceeded dimension bounds",
                error_reason=decode_error or "Invalid image data or corrupted format",
                processing_time_ms=elapsed,
            )
            self._update_session_stats([err_result])
            return [err_result]

        h, w = image.shape[:2]
        eval_results: list[EvaluationResult] = pipeline.evaluate_array(
            image, name=filename, aligned=False
        )

        analysis_results: list[FaceAnalysisResult] = []
        for res in eval_results:
            crop_b64 = self.extract_crop_base64(image, res.bbox)
            gate_dec = res.gate_decision
            effective_dec = gate_dec
            policy_override = False

            if gate_dec == "REVIEW":
                if review_policy == "treat_as_pass":
                    effective_dec = "PASS"
                    policy_override = True
                elif review_policy == "treat_as_reject":
                    effective_dec = "REJECT"
                    policy_override = True

            detection_status = "DETECTED" if res.status == "OK" else ("NO_FACE" if res.status == "NO_FACE" else "ERROR")

            analysis = FaceAnalysisResult(
                request_id=request_id,
                image_name=filename,
                image_width=res.image_width or w,
                image_height=res.image_height or h,
                detector=res.detector,
                detection_mode=detection_mode,
                fallback_used=res.fallback_used,
                face_index=res.face_index,
                confidence=res.confidence,
                bbox=res.bbox,
                quality_score=res.quality_score,
                status=res.status,
                decision=res.decision,
                gate_decision=gate_dec,
                effective_decision=effective_dec,
                review_policy=review_policy,
                policy_override=policy_override,
                detection_status=detection_status,
                decision_reason=res.decision_reason,
                error_reason=res.error_reason,
                detection_note=res.detection_note,
                detection_time_ms=res.detection_time_ms,
                quality_time_ms=res.quality_time_ms,
                pose_time_ms=res.pose_time_ms,
                gaze_time_ms=res.gaze_time_ms,
                photometrics_time_ms=res.photometrics_time_ms,
                processing_time_ms=res.processing_time_ms,
                face_area_ratio=res.face_area_ratio,
                inter_eye_distance_px=res.inter_eye_distance_px,
                pitch=res.pitch,
                yaw=res.yaw,
                roll=res.roll,
                pose_status=res.pose_status,
                sharpness_laplacian_var=res.sharpness_laplacian_var,
                mean_luminance=res.mean_luminance,
                underexposure_ratio=res.underexposure_ratio,
                overexposure_ratio=res.overexposure_ratio,
                illumination_asymmetry=res.illumination_asymmetry,
                gaze_pitch=res.gaze_pitch,
                gaze_yaw=res.gaze_yaw,
                gaze_status=res.gaze_status,
                face_crop_base64=crop_b64,
            )
            analysis_results.append(analysis)

        self._update_session_stats(analysis_results)
        try:
            del image
            del eval_results
        except Exception:
            pass
        release_process_memory()
        return analysis_results

    def _update_session_stats(self, results: list[FaceAnalysisResult]) -> None:
        with self.lock:
            self.total_processed += 1
            for r in results:
                if r.status == "OK":
                    self.faces_detected += 1
                else:
                    self.detection_failures += 1

                if r.decision != "NOT_EVALUATED":
                    self.decisions_completed += 1
                else:
                    self.not_evaluated_count += 1

    def store_batch(self, batch_result: BatchAnalysisResult) -> None:
        with self.lock:
            self._batches[batch_result.batch_id] = batch_result
            while len(self._batches) > self._max_batches:
                self._batches.popitem(last=False)

    def get_batch(self, batch_id: str) -> BatchAnalysisResult | None:
        with self.lock:
            return self._batches.get(batch_id)

    def export_batch_csv(self, batch_id: str) -> str:
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found")

        output = io.StringIO()
        writer = csv.writer(output)
        # Consistent headers matching batch_evaluate.py format
        writer.writerow(
            [
                "image",
                "width",
                "height",
                "detector",
                "fallback_used",
                "face",
                "confidence",
                "bbox",
                "face_area_ratio",
                "inter_eye_px",
                "quality_score",
                "status",
                "decision",
                "decision_reason",
                "pitch",
                "yaw",
                "roll",
                "pose_status",
                "error_reason",
                "detection_note",
                "detection_ms",
                "total_ms",
            ]
        )

        for r in batch.results:
            bbox_str = json.dumps(r.bbox) if r.bbox else ""
            score_str = f"{r.quality_score:.4f}" if r.quality_score is not None else ""
            conf_str = f"{r.confidence:.6f}" if r.confidence is not None else ""
            writer.writerow(
                [
                    r.image_name,
                    r.image_width or "",
                    r.image_height or "",
                    r.detector,
                    "true" if r.fallback_used else "false",
                    r.face_index or "",
                    conf_str,
                    bbox_str,
                    f"{r.face_area_ratio:.4f}" if r.face_area_ratio is not None else "",
                    (
                        f"{r.inter_eye_distance_px:.2f}"
                        if r.inter_eye_distance_px is not None
                        else ""
                    ),
                    score_str,
                    r.status,
                    r.decision,
                    r.decision_reason or "",
                    f"{r.pitch:.1f}" if r.pitch is not None else "",
                    f"{r.yaw:.1f}" if r.yaw is not None else "",
                    f"{r.roll:.1f}" if r.roll is not None else "",
                    r.pose_status,
                    r.error_reason or "",
                    r.detection_note or "",
                    (
                        f"{r.detection_time_ms:.1f}"
                        if r.detection_time_ms is not None
                        else ""
                    ),
                    f"{r.processing_time_ms:.1f}",
                ]
            )

        return output.getvalue()

    def export_batch_json(self, batch_id: str) -> str:
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found")

        # Strip heavy base64 strings from JSON export report
        export_dict = batch.model_dump()
        for res in export_dict.get("results", []):
            res.pop("face_crop_base64", None)

        return json.dumps(export_dict, indent=2)

    def get_passed_faces_count(self, batch_id: str, review_policy: str = "manual_review") -> int:
        batch = self.get_batch(batch_id)
        if not batch:
            return 0
        count = 0
        for r in batch.results:
            eff = r.gate_decision
            if eff == "REVIEW":
                if review_policy == "treat_as_pass":
                    eff = "PASS"
                elif review_policy == "treat_as_reject":
                    eff = "REJECT"
            if eff == "PASS":
                count += 1
        return count

    def generate_passed_faces_zip(
        self, batch_id: str, review_policy: str = "manual_review"
    ) -> tuple[str | None, int]:
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found")

        # Collect passed faces
        passed_faces: list[FaceAnalysisResult] = []
        for r in batch.results:
            eff = r.gate_decision
            override = False
            if eff == "REVIEW":
                if review_policy == "treat_as_pass":
                    eff = "PASS"
                    override = True
                elif review_policy == "treat_as_reject":
                    eff = "REJECT"
                    override = True
            if eff == "PASS":
                passed_faces.append(
                    r.model_copy(
                        update={
                            "effective_decision": eff,
                            "review_policy": review_policy,
                            "policy_override": override,
                        }
                    )
                )

        if not passed_faces:
            return None, 0

        # Safe temporary zip file
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_", batch_id[:8])
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False, prefix=f"face_fiqa_passed_{safe_prefix}_"
        )
        zip_path = tmp_file.name
        tmp_file.close()

        try:
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                used_names: set[str] = set()
                csv_buffer = io.StringIO()
                csv_writer = csv.writer(csv_buffer)
                csv_writer.writerow(
                    [
                        "archive_filename",
                        "source_image",
                        "face_index",
                        "confidence",
                        "quality_score",
                        "gate_decision",
                        "effective_decision",
                        "review_policy",
                        "policy_override",
                        "pitch",
                        "yaw",
                        "roll",
                        "pose_status",
                        "gaze_pitch",
                        "gaze_yaw",
                        "gaze_status",
                        "sharpness_laplacian_var",
                        "mean_luminance",
                        "decision_reason",
                    ]
                )

                manifest_items: list[dict[str, Any]] = []

                for face in passed_faces:
                    raw_stem = Path(face.image_name).stem if face.image_name else "image"
                    clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem) or "face"
                    f_idx = face.face_index or 1
                    base_name = f"{clean_stem}_face{f_idx:02d}.jpg"

                    # Ensure unique filename in zip archive
                    target_name = base_name
                    counter = 1
                    while target_name in used_names:
                        target_name = f"{clean_stem}_face{f_idx:02d}_{counter}.jpg"
                        counter += 1
                    used_names.add(target_name)

                    # Extract face crop bytes
                    if face.face_crop_base64:
                        try:
                            b64_data = face.face_crop_base64.split(",", 1)[-1]
                            crop_bytes = base64.b64decode(b64_data)
                            zf.writestr(target_name, crop_bytes)
                        except Exception as exc:
                            logger.warning(f"Could not write crop for {target_name}: {exc}")

                    # Manifest records
                    conf_str = f"{face.confidence:.6f}" if face.confidence is not None else ""
                    score_str = (
                        f"{face.quality_score:.4f}" if face.quality_score is not None else ""
                    )
                    pitch_str = f"{face.pitch:.1f}" if face.pitch is not None else ""
                    yaw_str = f"{face.yaw:.1f}" if face.yaw is not None else ""
                    roll_str = f"{face.roll:.1f}" if face.roll is not None else ""
                    gaze_p_str = f"{face.gaze_pitch:.1f}" if face.gaze_pitch is not None else ""
                    gaze_y_str = f"{face.gaze_yaw:.1f}" if face.gaze_yaw is not None else ""
                    sharp_str = (
                        f"{face.sharpness_laplacian_var:.2f}"
                        if face.sharpness_laplacian_var is not None
                        else ""
                    )
                    lum_str = (
                        f"{face.mean_luminance:.2f}"
                        if face.mean_luminance is not None
                        else ""
                    )

                    csv_writer.writerow(
                        [
                            target_name,
                            face.image_name,
                            face.face_index or 1,
                            conf_str,
                            score_str,
                            face.gate_decision,
                            face.effective_decision,
                            face.review_policy,
                            "true" if face.policy_override else "false",
                            pitch_str,
                            yaw_str,
                            roll_str,
                            face.pose_status,
                            gaze_p_str,
                            gaze_y_str,
                            face.gaze_status,
                            sharp_str,
                            lum_str,
                            face.decision_reason or "",
                        ]
                    )

                    manifest_items.append(
                        {
                            "archive_filename": target_name,
                            "source_image": face.image_name,
                            "face_index": face.face_index or 1,
                            "confidence": face.confidence,
                            "quality_score": face.quality_score,
                            "gate_decision": face.gate_decision,
                            "effective_decision": face.effective_decision,
                            "review_policy": face.review_policy,
                            "policy_override": face.policy_override,
                            "pitch": face.pitch,
                            "yaw": face.yaw,
                            "roll": face.roll,
                            "pose_status": face.pose_status,
                            "gaze_pitch": face.gaze_pitch,
                            "gaze_yaw": face.gaze_yaw,
                            "gaze_status": face.gaze_status,
                            "sharpness_laplacian_var": face.sharpness_laplacian_var,
                            "mean_luminance": face.mean_luminance,
                            "decision_reason": face.decision_reason,
                        }
                    )

                # Write manifest files inside archive
                zf.writestr("manifest.csv", csv_buffer.getvalue().encode("utf-8"))
                manifest_json = {
                    "batch_id": batch_id,
                    "review_policy": review_policy,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "total_passed_faces": len(manifest_items),
                    "items": manifest_items,
                }
                zf.writestr(
                    "manifest.json", json.dumps(manifest_json, indent=2).encode("utf-8")
                )

            return zip_path, len(passed_faces)
        except Exception:
            if os.path.exists(zip_path):
                try:
                    os.unlink(zip_path)
                except Exception:
                    pass
            raise


# Global service instance
pipeline_service = PipelineService()
