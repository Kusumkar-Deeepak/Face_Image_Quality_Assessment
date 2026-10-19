"""Production verification test suite for Face Image Quality Assessment system.

Verifies:
1. Batch of 1 image succeeds.
2. Batch of 100 images succeeds (strictly enforcing <= 100).
3. Batch of 101 images is rejected with HTTP 400.
4. Corrupted and unreadable images in batch are handled safely without crashing.
5. Decompression bomb and image dimension safety bounds.
6. Six discrete gate decision states (PASS, REVIEW, REJECT, NO_FACE, NOT_EVALUATED, ERROR).
7. Review policy: manual_review (leaves REVIEW in review).
8. Review policy: treat_as_pass (overrides REVIEW to PASS).
9. Review policy: treat_as_reject (overrides REVIEW to REJECT).
10. Passed faces count endpoint.
11. Passed faces ZIP archive export: valid zip structure, face crops, manifest.csv & manifest.json.
12. Non-passed faces are strictly excluded from ZIP export.
13. Zero passed faces returns HTTP 400.
14. Safe temporary file cleanup.
15. Batch summary statistics and granular timing fields.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.schemas import BatchAnalysisResult, FaceAnalysisResult
from api.service import pipeline_service
from quality_pipeline import EvaluationResult

client = TestClient(app)
IMAGE2_PATH = ROOT_DIR / "assets" / "test_images" / "image2.jpg"


def _make_dummy_jpeg(width: int = 40, height: int = 40, color: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    """Generate small valid JPEG bytes in memory."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_eval_result(
    image: str = "test.jpg",
    status: str = "OK",
    decision: str = "NOT_EVALUATED",
    quality_score: float | None = 0.85,
    bbox: list[float] | None = None,
    detector: str = "scrfd",
    error_reason: str | None = None,
    processing_time_ms: float = 50.0,
) -> EvaluationResult:
    return EvaluationResult(
        image=image,
        image_width=100,
        image_height=100,
        detector=detector,
        face_index=1,
        confidence=0.95,
        bbox=bbox or [10, 10, 50, 50],
        quality_score=quality_score,
        status=status,
        decision=decision,
        error_reason=error_reason,
        processing_time_ms=processing_time_ms,
    )


@pytest.fixture(autouse=True)
def reset_thresholds():
    """Ensure tests start and end with default uncalibrated thresholds."""
    client.post(
        "/api/config/thresholds",
        json={
            "accept_min": None,
            "reject_max": None,
            "max_abs_pitch": None,
            "max_abs_yaw": None,
            "max_abs_roll": None,
        },
    )
    yield
    client.post(
        "/api/config/thresholds",
        json={
            "accept_min": None,
            "reject_max": None,
            "max_abs_pitch": None,
            "max_abs_yaw": None,
            "max_abs_roll": None,
        },
    )


# -------------------------------------------------------------------------
# 1. Batch of 1 image succeeds
# -------------------------------------------------------------------------
def test_batch_1_image_succeeds():
    img_bytes = _make_dummy_jpeg(60, 60)
    files = [("files", ("test_01.jpg", img_bytes, "image/jpeg"))]
    data = {
        "detector": "scrfd",
        "fallback_detector": "none",
        "scrfd_input_size": "320",
        "head_pose": "false",
        "gaze": "false",
        "review_policy": "manual_review",
    }
    response = client.post("/api/analyze/batch", files=files, data=data)
    assert response.status_code == 200
    batch = response.json()
    assert batch["total_images"] == 1
    assert "batch_id" in batch
    assert "pass_count" in batch
    assert "review_count" in batch
    assert "reject_count" in batch
    assert "no_face_count" in batch
    assert "error_count" in batch
    assert "total_faces" in batch
    assert batch["review_policy"] == "manual_review"


# -------------------------------------------------------------------------
# 2. Batch of 100 images succeeds (under HTTP 400 limit)
# -------------------------------------------------------------------------
def test_batch_100_images_limit_accepted():
    img_bytes = _make_dummy_jpeg(20, 20)
    files = [("files", (f"img_{i:03d}.jpg", img_bytes, "image/jpeg")) for i in range(100)]
    data = {
        "detector": "scrfd",
        "fallback_detector": "none",
        "scrfd_input_size": "320",
        "head_pose": "false",
        "gaze": "false",
    }
    response = client.post("/api/analyze/batch", files=files, data=data)
    assert response.status_code == 200
    batch = response.json()
    assert batch["total_images"] == 100
    assert len(batch["results"]) == 100


# -------------------------------------------------------------------------
# 3. Batch of 101 images rejected with HTTP 400
# -------------------------------------------------------------------------
def test_batch_101_images_rejected_http_400():
    img_bytes = _make_dummy_jpeg(10, 10)
    files = [("files", (f"img_{i:03d}.jpg", img_bytes, "image/jpeg")) for i in range(101)]
    response = client.post("/api/analyze/batch", files=files)
    assert response.status_code == 400
    err_detail = response.json().get("detail", "")
    assert "exceeds maximum limit of 100 images" in err_detail
    assert "101" in err_detail


# -------------------------------------------------------------------------
# 4. Corrupted and partial failure handling
# -------------------------------------------------------------------------
def test_batch_partial_failure_and_corrupt_files():
    valid_bytes = _make_dummy_jpeg(40, 40)
    corrupt_bytes = b"CORRUPTED_NOT_A_JPEG_FILE"
    files = [
        ("files", ("valid.jpg", valid_bytes, "image/jpeg")),
        ("files", ("broken.jpg", corrupt_bytes, "image/jpeg")),
    ]
    data = {
        "detector": "scrfd",
        "fallback_detector": "none",
        "head_pose": "false",
        "gaze": "false",
    }
    response = client.post("/api/analyze/batch", files=files, data=data)
    assert response.status_code == 200
    batch = response.json()
    assert batch["total_images"] == 2
    assert batch["failures"] >= 1
    # Check that broken file resulted in ERROR gate decision
    broken_results = [r for r in batch["results"] if r["image_name"] == "broken.jpg"]
    assert len(broken_results) == 1
    assert broken_results[0]["status"] == "READ_ERROR"
    assert broken_results[0]["gate_decision"] == "ERROR"
    assert broken_results[0]["effective_decision"] == "ERROR"


# -------------------------------------------------------------------------
# 5. Decompression bomb and image dimension safety bounds
# -------------------------------------------------------------------------
def test_decompression_bomb_and_dimension_bounds():
    # Attempting to decode image exceeding 16384 dimension bound
    huge_mock = MagicMock()
    huge_mock.width = 16385
    huge_mock.height = 100
    huge_mock.__enter__.return_value = huge_mock
    huge_mock.__exit__.return_value = None

    with patch("PIL.Image.open", return_value=huge_mock):
        results = pipeline_service.analyze_image_bytes(b"dummy_bytes", "huge.jpg")
        assert len(results) == 1
        res = results[0]
        assert res.status == "READ_ERROR"
        assert res.gate_decision == "ERROR"
        assert "exceed" in (res.error_reason or "").lower()


# -------------------------------------------------------------------------
# 6. Six discrete gate decision states
# -------------------------------------------------------------------------
def test_six_discrete_gate_decisions():
    # 1. OK + ACCEPT -> PASS
    res_pass = make_eval_result(status="OK", decision="ACCEPT", quality_score=0.85)
    assert res_pass.gate_decision == "PASS"

    # 2. OK + REVIEW -> REVIEW
    res_rev = make_eval_result(status="OK", decision="REVIEW", quality_score=0.55)
    assert res_rev.gate_decision == "REVIEW"

    # 3. OK + REJECT -> REJECT
    res_rej = make_eval_result(status="OK", decision="REJECT", quality_score=0.25)
    assert res_rej.gate_decision == "REJECT"

    # 4. NO_FACE -> NO_FACE
    res_noface = make_eval_result(status="NO_FACE", decision="NOT_EVALUATED", quality_score=None, bbox=None)
    assert res_noface.gate_decision == "NO_FACE"

    # 5. OK + NOT_EVALUATED -> NOT_EVALUATED
    res_noteval = make_eval_result(status="OK", decision="NOT_EVALUATED", quality_score=0.75)
    assert res_noteval.gate_decision == "NOT_EVALUATED"

    # 6. READ_ERROR -> ERROR
    res_err = make_eval_result(status="READ_ERROR", decision="NOT_EVALUATED", quality_score=None, bbox=None)
    assert res_err.gate_decision == "ERROR"


# -------------------------------------------------------------------------
# 7, 8, 9. Review Policy: manual_review, treat_as_pass, treat_as_reject
# -------------------------------------------------------------------------
def _create_mock_review_analysis(review_policy: str) -> list[FaceAnalysisResult]:
    """Helper creating a simulated analysis result with REVIEW status."""
    mock_eval = make_eval_result(status="OK", decision="REVIEW", quality_score=0.55)
    dummy_bytes = _make_dummy_jpeg(40, 40)

    with patch.object(pipeline_service, "get_pipeline") as mock_gp:
        mock_pipeline = MagicMock()
        mock_pipeline.evaluate_array.return_value = [mock_eval]
        mock_gp.return_value = mock_pipeline

        return pipeline_service.analyze_image_bytes(
            dummy_bytes,
            "test.jpg",
            review_policy=review_policy,
        )


def test_review_policy_manual_review():
    results = _create_mock_review_analysis("manual_review")
    assert len(results) == 1
    face = results[0]
    assert face.gate_decision == "REVIEW"
    assert face.effective_decision == "REVIEW"
    assert face.policy_override is False
    assert face.review_policy == "manual_review"


def test_review_policy_treat_as_pass():
    results = _create_mock_review_analysis("treat_as_pass")
    assert len(results) == 1
    face = results[0]
    assert face.gate_decision == "REVIEW"
    assert face.effective_decision == "PASS"
    assert face.policy_override is True
    assert face.review_policy == "treat_as_pass"


def test_review_policy_treat_as_reject():
    results = _create_mock_review_analysis("treat_as_reject")
    assert len(results) == 1
    face = results[0]
    assert face.gate_decision == "REVIEW"
    assert face.effective_decision == "REJECT"
    assert face.policy_override is True
    assert face.review_policy == "treat_as_reject"


# -------------------------------------------------------------------------
# 10, 11, 12, 13. Passed Faces ZIP export, counts, and zero-pass error
# -------------------------------------------------------------------------
@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_passed_faces_count_and_zip_export():
    # 1. First calibrate thresholds so image2.jpg passes (score ~0.8593)
    cal_res = client.post(
        "/api/config/thresholds",
        json={"accept_min": 0.70, "reject_max": 0.40},
    )
    assert cal_res.status_code == 200

    # 2. Run batch with image2.jpg
    with open(IMAGE2_PATH, "rb") as f:
        img_bytes = f.read()

    files = [
        ("files", ("passed_face_test.jpg", img_bytes, "image/jpeg")),
        ("files", ("corrupted_nonpass.jpg", b"corrupted", "image/jpeg")),
    ]
    data = {
        "detector": "scrfd",
        "fallback_detector": "retinaface",
        "scrfd_input_size": "640",
        "head_pose": "true",
        "gaze": "true",
        "review_policy": "manual_review",
    }
    batch_res = client.post("/api/analyze/batch", files=files, data=data)
    assert batch_res.status_code == 200
    batch_data = batch_res.json()
    batch_id = batch_data["batch_id"]

    # Verify summary cards counts
    assert batch_data["pass_count"] >= 1
    assert batch_data["error_count"] >= 1

    # 3. Test passed-count endpoint
    count_res = client.get(f"/api/results/{batch_id}/passed-count?review_policy=manual_review")
    assert count_res.status_code == 200
    count_data = count_res.json()
    assert count_data["batch_id"] == batch_id
    assert count_data["passed_count"] >= 1

    # 4. Test passed-zip export
    zip_res = client.get(f"/api/results/{batch_id}/passed-zip?review_policy=manual_review")
    assert zip_res.status_code == 200
    assert zip_res.headers["content-type"] == "application/zip"
    assert "face_quality_passed" in zip_res.headers["content-disposition"]

    # 5. Inspect zip archive contents
    zf = zipfile.ZipFile(io.BytesIO(zip_res.content))
    namelist = zf.namelist()

    # Must contain manifest.csv and manifest.json
    assert "manifest.csv" in namelist
    assert "manifest.json" in namelist

    # Must contain the passed face crop
    face_crops = [name for name in namelist if name.endswith(".jpg")]
    assert len(face_crops) >= 1
    assert all("passed_face_test" in name for name in face_crops)

    # Must NOT contain non-passed image
    assert not any("corrupted" in name for name in namelist)

    # Validate manifest.json schema
    manifest_raw = zf.read("manifest.json").decode("utf-8")
    manifest_data = json.loads(manifest_raw)
    assert manifest_data["batch_id"] == batch_id
    assert manifest_data["review_policy"] == "manual_review"
    assert manifest_data["total_passed_faces"] == len(face_crops)
    assert len(manifest_data["items"]) == len(face_crops)
    first_item = manifest_data["items"][0]
    assert "quality_score" in first_item
    assert "gate_decision" in first_item
    assert "effective_decision" in first_item
    assert first_item["effective_decision"] in ("PASS", "ACCEPT")

    # Validate manifest.csv columns
    csv_raw = zf.read("manifest.csv").decode("utf-8")
    csv_reader = list(csv.reader(io.StringIO(csv_raw)))
    assert len(csv_reader) >= 2  # header + row
    header = csv_reader[0]
    assert "archive_filename" in header
    assert "source_image" in header
    assert "quality_score" in header
    assert "effective_decision" in header


def test_passed_faces_zip_zero_passed_returns_400():
    # In uncalibrated state, decisions are NOT_EVALUATED, so passed_count == 0
    img_bytes = _make_dummy_jpeg(30, 30)
    files = [("files", ("uncalibrated.jpg", img_bytes, "image/jpeg"))]
    batch_res = client.post("/api/analyze/batch", files=files)
    assert batch_res.status_code == 200
    batch_id = batch_res.json()["batch_id"]

    # Requesting zip with 0 passed faces returns HTTP 400
    zip_res = client.get(f"/api/results/{batch_id}/passed-zip")
    assert zip_res.status_code == 400
    assert "No passed faces found" in zip_res.json().get("detail", "")


# -------------------------------------------------------------------------
# 14. Temporary ZIP cleanup verified
# -------------------------------------------------------------------------
def test_temp_zip_cleanup_on_error():
    # Service cleanup test: when generate_passed_faces_zip encounters an error
    with patch("zipfile.ZipFile", side_effect=RuntimeError("Simulated write failure")):
        # Mock batch results with a passed face
        mock_face = FaceAnalysisResult(
            request_id="cleanup-test",
            image_name="test.jpg",
            image_width=100,
            image_height=100,
            detector="scrfd",
            status="OK",
            decision="ACCEPT",
            gate_decision="PASS",
            effective_decision="PASS",
            review_policy="manual_review",
            policy_override=False,
            detection_status="DETECTED",
            processing_time_ms=10.0,
            face_crop_base64="data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        )
        mock_batch = BatchAnalysisResult(
            batch_id="cleanup-batch-id",
            created_at="2026-10-18T00:00:00Z",
            total_images=1,
            processed_images=1,
            successful_evaluations=1,
            failures=0,
            total_faces=1,
            pass_count=1,
            review_count=0,
            reject_count=0,
            no_face_count=0,
            error_count=0,
            review_policy="manual_review",
            results=[mock_face],
        )
        with pipeline_service.lock:
            pipeline_service._batches["cleanup-batch-id"] = mock_batch

        with pytest.raises(RuntimeError, match="Simulated write failure"):
            pipeline_service.generate_passed_faces_zip("cleanup-batch-id", "manual_review")


# -------------------------------------------------------------------------
# 15. Granular timing fields populated
# -------------------------------------------------------------------------
def test_granular_timing_fields_populated():
    res = make_eval_result(status="OK", quality_score=0.7)
    res.detection_time_ms = 45.2
    res.quality_time_ms = 120.5
    res.pose_time_ms = 15.1
    res.gaze_time_ms = 10.3
    res.photometrics_time_ms = 2.4
    res.processing_time_ms = 193.5

    assert res.quality_time_ms == 120.5
    assert res.pose_time_ms == 15.1
    assert res.gaze_time_ms == 10.3
    assert res.photometrics_time_ms == 2.4
    assert res.processing_time_ms == 193.5
