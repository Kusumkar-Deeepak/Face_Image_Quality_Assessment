import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from api.schemas import (
    BatchAnalysisResult,
    ConfigResponse,
    FaceAnalysisResult,
    HealthResponse,
    SessionStatsResponse,
    ThresholdsUpdateRequest,
)
from api.service import pipeline_service

logger = logging.getLogger("face_quality_api")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
MAX_BATCH_SIZE = 100
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm model on startup
    logger.info("Starting Face Quality Lab API service...")
    pipeline_service.initialize()
    yield
    logger.info("Shutting down Face Quality Lab API service...")


app = FastAPI(
    title="Face Quality Lab API",
    description="FastAPI service for eDifFIQA face image quality assessment",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",  # Local developer flexibility
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = datetime.now(timezone.utc)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal processing error",
                "request_id": request_id,
            },
        )


@app.get("/api/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return pipeline model availability and hardware acceleration status."""
    return pipeline_service.get_health()


@app.get("/api/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    """Return detector parameters and threshold calibration status."""
    return pipeline_service.get_config()


@app.post("/api/config/thresholds", response_model=ConfigResponse)
def update_thresholds(req: ThresholdsUpdateRequest) -> ConfigResponse:
    """Update quality decision thresholds with validation."""
    try:
        return pipeline_service.update_thresholds(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@app.get("/api/session/stats", response_model=SessionStatsResponse)
def get_session_stats() -> SessionStatsResponse:
    """Return cumulative session statistics."""
    return pipeline_service.get_session_stats()


def _validate_image_file(file: UploadFile) -> None:
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


@app.post("/api/analyze")
async def analyze_single_image(
    file: UploadFile = File(...),
    detector: str = Form("scrfd"),
    fallback_detector: str | None = Form(None),
    detection_mode: str = Form("fallback"),
    confidence_threshold: float | None = Form(None),
    scrfd_input_size: int = Form(640),
    head_pose: bool = Form(False),
    gaze: bool = Form(False),
    review_policy: str = Form("manual_review"),
) -> dict[str, Any]:
    """Analyze a single uploaded face image."""
    _validate_image_file(file)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size ({MAX_FILE_SIZE // (1024*1024)} MB)",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Clean fallback value
    clean_fallback = fallback_detector if fallback_detector not in ("none", "", "null") else None

    results = pipeline_service.analyze_image_bytes(
        image_bytes=content,
        filename=file.filename or "uploaded_image.jpg",
        detector=detector,
        fallback_detector=clean_fallback,
        scrfd_input_size=scrfd_input_size,
        head_pose=head_pose,
        gaze=gaze,
        detection_mode=detection_mode,
        confidence_threshold=confidence_threshold,
        review_policy=review_policy,
    )

    return {
        "filename": file.filename,
        "faces_count": len([r for r in results if r.status == "OK"]),
        "results": [r.model_dump() for r in results],
        "primary_result": results[0].model_dump() if results else None,
    }


@app.post("/api/analyze/batch", response_model=BatchAnalysisResult)
async def analyze_batch_images(
    files: list[UploadFile] = File(...),
    detector: str = Form("scrfd"),
    fallback_detector: str | None = Form(None),
    detection_mode: str = Form("fallback"),
    confidence_threshold: float | None = Form(None),
    scrfd_input_size: int = Form(640),
    head_pose: bool = Form(False),
    gaze: bool = Form(False),
    review_policy: str = Form("manual_review"),
) -> BatchAnalysisResult:
    """Analyze multiple uploaded face images with 100-image enforcement and partial failure tolerance."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch exceeds maximum limit of {MAX_BATCH_SIZE} images (received {len(files)} images).",
        )

    batch_id = str(uuid.uuid4())
    all_results: list[FaceAnalysisResult] = []
    clean_fallback = fallback_detector if fallback_detector not in ("none", "", "null") else None

    for upload in files:
        filename = upload.filename or "unknown.jpg"
        try:
            _validate_image_file(upload)
            content = await upload.read()
            if len(content) > MAX_FILE_SIZE:
                err_result = FaceAnalysisResult(
                    request_id=str(uuid.uuid4()),
                    image_name=filename,
                    detector=detector,
                    detection_mode=detection_mode,
                    status="READ_ERROR",
                    decision="NOT_EVALUATED",
                    gate_decision="ERROR",
                    effective_decision="ERROR",
                    review_policy=review_policy,
                    policy_override=False,
                    detection_status="ERROR",
                    error_reason="File size exceeds limit",
                    processing_time_ms=0.0,
                )
                all_results.append(err_result)
                continue

            results = pipeline_service.analyze_image_bytes(
                image_bytes=content,
                filename=filename,
                detector=detector,
                fallback_detector=clean_fallback,
                scrfd_input_size=scrfd_input_size,
                head_pose=head_pose,
                gaze=gaze,
                detection_mode=detection_mode,
                confidence_threshold=confidence_threshold,
                review_policy=review_policy,
            )
            all_results.extend(results)
        except Exception as exc:
            logger.error(f"Error analyzing batch item {filename}: {exc}")
            err_result = FaceAnalysisResult(
                request_id=str(uuid.uuid4()),
                image_name=filename,
                detector=detector,
                detection_mode=detection_mode,
                status="READ_ERROR",
                decision="NOT_EVALUATED",
                gate_decision="ERROR",
                effective_decision="ERROR",
                review_policy=review_policy,
                policy_override=False,
                detection_status="ERROR",
                error_reason=f"Processing failed: {str(exc)}",
                processing_time_ms=0.0,
            )
            all_results.append(err_result)

    success_count = sum(1 for r in all_results if r.status == "OK")
    failure_count = len(all_results) - success_count
    pass_count = sum(1 for r in all_results if r.effective_decision == "PASS")
    review_count = sum(1 for r in all_results if r.effective_decision == "REVIEW")
    reject_count = sum(1 for r in all_results if r.effective_decision == "REJECT")
    no_face_count = sum(1 for r in all_results if r.effective_decision == "NO_FACE")
    error_count = sum(1 for r in all_results if r.effective_decision == "ERROR")
    total_faces = sum(1 for r in all_results if r.detection_status == "DETECTED")

    batch_result = BatchAnalysisResult(
        batch_id=batch_id,
        total_images=len(files),
        processed_images=len(files),
        successful_evaluations=success_count,
        failures=failure_count,
        pass_count=pass_count,
        review_count=review_count,
        reject_count=reject_count,
        no_face_count=no_face_count,
        error_count=error_count,
        total_faces=total_faces,
        review_policy=review_policy,
        results=all_results,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline_service.store_batch(batch_result)
    return batch_result


@app.get("/api/results/{batch_id}", response_model=BatchAnalysisResult)
def get_batch_result(batch_id: str) -> BatchAnalysisResult:
    """Retrieve results for a previously evaluated batch."""
    batch = pipeline_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch result not found")
    return batch


@app.post("/api/results/save", response_model=BatchAnalysisResult)
def save_batch_result(batch: BatchAnalysisResult) -> BatchAnalysisResult:
    """Store or synchronize a completed client-side batch for CSV/JSON exports."""
    # Ensure summary metrics are populated
    if batch.pass_count == 0 and batch.total_faces == 0 and batch.results:
        batch.pass_count = sum(1 for r in batch.results if r.effective_decision == "PASS")
        batch.review_count = sum(1 for r in batch.results if r.effective_decision == "REVIEW")
        batch.reject_count = sum(1 for r in batch.results if r.effective_decision == "REJECT")
        batch.no_face_count = sum(1 for r in batch.results if r.effective_decision == "NO_FACE")
        batch.error_count = sum(1 for r in batch.results if r.effective_decision == "ERROR")
        batch.total_faces = sum(1 for r in batch.results if r.detection_status == "DETECTED")

    pipeline_service.store_batch(batch)
    return batch


@app.get("/api/results/{batch_id}/passed-count")
def get_passed_count(
    batch_id: str,
    review_policy: str = Query("manual_review"),
) -> dict[str, Any]:
    """Return the count of passed faces for the specified batch and review policy."""
    batch = pipeline_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch result not found")
    count = pipeline_service.get_passed_faces_count(batch_id, review_policy=review_policy)
    return {
        "batch_id": batch_id,
        "passed_count": count,
        "review_policy": review_policy,
    }


@app.get("/api/results/{batch_id}/passed-zip")
def download_passed_zip(
    batch_id: str,
    review_policy: str = Query("manual_review"),
) -> FileResponse:
    """Generate and stream a ZIP archive of all passed face crops with manifest files."""
    batch = pipeline_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch result not found")

    zip_path, passed_count = pipeline_service.generate_passed_faces_zip(
        batch_id, review_policy=review_policy
    )
    if passed_count == 0 or zip_path is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No passed faces found in this batch to export.",
        )

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"face_quality_passed_{batch_id[:8]}.zip",
        background=BackgroundTask(os.unlink, zip_path),
    )


@app.get("/api/results/{batch_id}/export")
def export_batch_result(
    batch_id: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
) -> Response:
    """Export batch results as CSV or JSON report."""
    batch = pipeline_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch result not found")

    if format.lower() == "csv":
        csv_data = pipeline_service.export_batch_csv(batch_id)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="face_quality_batch_{batch_id[:8]}.csv"'
            },
        )
    else:
        json_data = pipeline_service.export_batch_json(batch_id)
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="face_quality_batch_{batch_id[:8]}.json"'
            },
        )


# Mount pre-built production frontend (Single-port deployment support)
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

