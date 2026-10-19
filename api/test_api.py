from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE2_PATH = ROOT_DIR / "assets" / "test_images" / "image2.jpg"


@pytest.fixture(autouse=True)
def reset_thresholds():
    client.post(
        "/api/config/thresholds",
        json={"accept_min": None, "reject_max": None, "max_abs_pitch": None, "max_abs_yaw": None, "max_abs_roll": None},
    )
    yield
    client.post(
        "/api/config/thresholds",
        json={"accept_min": None, "reject_max": None, "max_abs_pitch": None, "max_abs_yaw": None, "max_abs_roll": None},
    )


def test_health_check():
    """Verify backend health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "device" in data
    assert data["model_variant"] == "s"
    assert "scrfd" in data["supported_detectors"]
    assert "retinaface" in data["supported_detectors"]


def test_get_config():
    """Verify config endpoint and uncalibrated status."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["primary_detector"] == "scrfd"
    assert "thresholds" in data
    # Confirms thresholds are uncalibrated (nulls in quality_thresholds.json)
    assert data["is_calibrated"] is False
    assert data["thresholds"]["accept_min"] is None
    assert data["thresholds"]["reject_max"] is None


def test_get_session_stats():
    """Verify session statistics endpoint."""
    response = client.get("/api/session/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_processed" in data
    assert "faces_detected" in data
    assert "detection_failures" in data
    assert "quality_decisions_completed" in data


def test_invalid_image_upload_extension():
    """Test rejection of invalid file format."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_corrupted_image_upload_handled_safely():
    """Test that a malformed image returns READ_ERROR rather than crashing."""
    corrupt_bytes = b"FFD8FFE000104A46494600010101006000600000corrupt"
    files = {"file": ("corrupted.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "READ_ERROR"
    assert data["results"][0]["decision"] == "NOT_EVALUATED"


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_no_face_result_with_default_scrfd():
    """image2.jpg at 640 input size yields NO_FACE under SCRFD."""
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        data = {
            "detector": "scrfd",
            "fallback_detector": "none",
            "scrfd_input_size": "640",
            "head_pose": "false",
        }
        response = client.post("/api/analyze", files=files, data=data)

    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["results"]) == 1
    result = res_data["results"][0]
    assert result["status"] == "NO_FACE"
    assert result["quality_score"] is None
    assert result["decision"] == "NOT_EVALUATED"
    assert "no usable face" in (result["error_reason"] or "")


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_successful_quality_score_with_fallback_and_head_pose():
    """image2.jpg with retinaface fallback yields OK, score ~0.8593, and head pose."""
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        data = {
            "detector": "scrfd",
            "fallback_detector": "retinaface",
            "scrfd_input_size": "640",
            "head_pose": "true",
        }
        response = client.post("/api/analyze", files=files, data=data)

    assert response.status_code == 200
    res_data = response.json()
    assert len(res_data["results"]) >= 1
    primary = res_data["primary_result"]
    assert primary["status"] == "OK"
    assert primary["detector"] == "retinaface(fallback)"
    assert primary["fallback_used"] is True
    assert primary["quality_score"] is not None
    # Verify score is in valid range and close to CLI result 0.8593
    assert 0.85 <= primary["quality_score"] <= 0.87
    # Thresholds are uncalibrated -> NOT_EVALUATED
    assert primary["decision"] == "NOT_EVALUATED"
    assert "no calibrated score thresholds" in (primary["decision_reason"] or "")
    # Check bounding box and crop
    assert primary["bbox"] is not None and len(primary["bbox"]) == 4
    assert primary["face_crop_base64"] is not None
    assert primary["face_crop_base64"].startswith("data:image/jpeg;base64,")
    # Check head pose
    assert primary["pose_status"] == "OK"
    assert primary["pitch"] is not None
    assert primary["yaw"] is not None
    assert primary["roll"] is not None


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_batch_evaluation_with_partial_failure():
    """Batch evaluation processes multiple files, tolerant of partial failures."""
    with open(IMAGE2_PATH, "rb") as f:
        img_bytes = f.read()

    files = [
        ("files", ("image2_valid.jpg", img_bytes, "image/jpeg")),
        ("files", ("broken_file.jpg", b"bad content", "image/jpeg")),
    ]
    data = {
        "detector": "scrfd",
        "fallback_detector": "retinaface",
        "scrfd_input_size": "640",
        "head_pose": "false",
    }
    response = client.post("/api/analyze/batch", files=files, data=data)
    assert response.status_code == 200
    batch = response.json()
    assert batch["total_images"] == 2
    assert batch["successful_evaluations"] >= 1
    assert batch["failures"] >= 1
    batch_id = batch["batch_id"]

    # Verify retrieval
    get_res = client.get(f"/api/results/{batch_id}")
    assert get_res.status_code == 200

    # Verify CSV export
    csv_res = client.get(f"/api/results/{batch_id}/export?format=csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    reader = csv.reader(io.StringIO(csv_res.text))
    rows = list(reader)
    assert len(rows) >= 3  # Header + 2 results
    assert "quality_score" in rows[0]

    # Verify JSON export
    json_res = client.get(f"/api/results/{batch_id}/export?format=json")
    assert json_res.status_code == 200
    json_data = json_res.json()
    assert json_data["batch_id"] == batch_id
    assert len(json_data["results"]) >= 2


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_update_thresholds_and_calibrated_decision():
    """Verify that setting calibrated thresholds activates ACCEPT decisions, and resetting restores NOT_EVALUATED."""
    # 1. Update thresholds to calibrated state
    update_res = client.post(
        "/api/config/thresholds",
        json={"accept_min": 0.70, "reject_max": 0.40, "max_abs_yaw": 30.0},
    )
    assert update_res.status_code == 200
    config_data = update_res.json()
    assert config_data["is_calibrated"] is True
    assert config_data["thresholds"]["accept_min"] == 0.70

    # 2. Analyze image2 (score ~0.8593 >= 0.70) -> decision should now be ACCEPT!
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        data = {"detector": "scrfd", "fallback_detector": "retinaface", "scrfd_input_size": "640"}
        res = client.post("/api/analyze", files=files, data=data)
    assert res.status_code == 200
    primary = res.json()["primary_result"]
    assert primary["status"] == "OK"
    assert primary["decision"] == "ACCEPT"
    assert "score" in primary["decision_reason"]

    # 3. Reset thresholds to uncalibrated
    reset_res = client.post(
        "/api/config/thresholds",
        json={"accept_min": None, "reject_max": None, "max_abs_yaw": None},
    )
    assert reset_res.status_code == 200
    reset_config = reset_res.json()
    assert reset_config["is_calibrated"] is False

    # 4. Analyze image2 again -> decision must honestly return to NOT_EVALUATED
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        res2 = client.post("/api/analyze", files=files, data=data)
    assert res2.status_code == 200
    assert res2.json()["primary_result"]["decision"] == "NOT_EVALUATED"


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_photometric_diagnostics_returned():
    """Verify photometric diagnostics are populated on detected faces."""
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        data = {
            "detector": "scrfd",
            "fallback_detector": "retinaface",
            "scrfd_input_size": "640",
            "head_pose": "false",
            "gaze": "false",
        }
        res = client.post("/api/analyze", files=files, data=data)

    assert res.status_code == 200
    primary = res.json()["primary_result"]
    assert primary["status"] == "OK"
    assert primary["sharpness_laplacian_var"] is not None
    assert primary["sharpness_laplacian_var"] > 0
    assert primary["mean_luminance"] is not None
    assert 0 <= primary["mean_luminance"] <= 255
    assert primary["underexposure_ratio"] is not None
    assert 0 <= primary["underexposure_ratio"] <= 1.0
    assert primary["overexposure_ratio"] is not None
    assert 0 <= primary["overexposure_ratio"] <= 1.0
    assert primary["illumination_asymmetry"] is not None
    assert primary["gaze_status"] == "DISABLED"
    assert primary["gaze_pitch"] is None


@pytest.mark.skipif(not IMAGE2_PATH.exists(), reason="image2.jpg required")
def test_gaze_estimation_endpoint():
    """Verify UniFace MobileGaze estimation when enabled."""
    with open(IMAGE2_PATH, "rb") as f:
        files = {"file": ("image2.jpg", f, "image/jpeg")}
        data = {
            "detector": "scrfd",
            "fallback_detector": "retinaface",
            "scrfd_input_size": "640",
            "head_pose": "true",
            "gaze": "true",
        }
        res = client.post("/api/analyze", files=files, data=data)

    assert res.status_code == 200
    primary = res.json()["primary_result"]
    assert primary["status"] == "OK"
    assert primary["pose_status"] == "OK"
    assert primary["gaze_status"] in ("OK", "ERROR")
    if primary["gaze_status"] == "OK":
        assert primary["gaze_pitch"] is not None
        assert primary["gaze_yaw"] is not None
        # MobileGaze angles in degrees should be within physical limits [-90, 90]
        assert -90.0 <= primary["gaze_pitch"] <= 90.0
        assert -90.0 <= primary["gaze_yaw"] <= 90.0


