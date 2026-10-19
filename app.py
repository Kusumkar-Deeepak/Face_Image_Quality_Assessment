"""Face Image Quality Assessment (FIQA) & Surveillance Dashboard
Hugging Face Spaces Entrypoint (Gradio SDK - 100% Free Tier Compatible)
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is None:
                return lambda f: f
            return fn

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.service import pipeline_service


# ---------------------------------------------------------------------------
# Helper: Annotate Detected Faces on Image
# ---------------------------------------------------------------------------
def annotate_faces_on_image(image_rgb: np.ndarray, faces: list[Any]) -> np.ndarray:
    """Draw bounding boxes and quality annotations on the detected faces."""
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]

    for idx, face in enumerate(faces):
        if not face.bbox or len(face.bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(round(coord)) for coord in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Color based on decision: Green for PASS, Amber for REVIEW, Red for REJECT, Blue default
        if face.effective_decision in ("PASS", "ACCEPT"):
            box_color = (34, 197, 94)  # Emerald / Green
        elif face.effective_decision == "REVIEW":
            box_color = (234, 179, 8)  # Amber
        elif face.effective_decision == "REJECT":
            box_color = (239, 68, 68)  # Rose / Red
        else:
            box_color = (59, 130, 246)  # Blue

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Draw label header
        score_text = f" #{idx+1}"
        if face.quality_score is not None:
            score_text += f" | Q:{face.quality_score:.2f}"
        if face.confidence is not None:
            score_text += f" ({face.confidence*100:.0f}%)"

        label_size, baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top_y = max(y1, label_size[1] + 6)
        cv2.rectangle(
            annotated,
            (x1, top_y - label_size[1] - 4),
            (x1 + label_size[0] + 6, top_y + 2),
            box_color,
            -1,
        )
        cv2.putText(
            annotated,
            score_text,
            (x1 + 3, top_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Single Image Analysis Callback
# ---------------------------------------------------------------------------
@spaces.GPU
def analyze_single_face(
    input_image: np.ndarray | None,
    detector: str,
    detection_mode: str,
    scrfd_size: int,
    confidence_thresh: float,
    head_pose: bool,
    gaze: bool,
    review_policy: str,
):
    if input_image is None:
        return None, None, "### ⚠️ Please upload or capture an image first.", None, None

    # Convert RGB to JPEG bytes
    pil_img = Image.fromarray(input_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    fallback_det = "retinaface" if detector == "scrfd" else "scrfd"

    results = pipeline_service.analyze_image_bytes(
        image_bytes=image_bytes,
        filename="input.jpg",
        detector=detector,
        fallback_detector=fallback_det,
        detection_mode=detection_mode,
        scrfd_input_size=int(scrfd_size),
        confidence_threshold=float(confidence_thresh),
        head_pose=head_pose,
        gaze=gaze,
        review_policy=review_policy,
    )

    if not results:
        return input_image, None, "### ❌ No result returned by pipeline.", None, None

    # Annotate image
    annotated = annotate_faces_on_image(input_image, results)

    # First face
    primary = results[0]
    crop_img = None
    if primary.face_crop_base64:
        try:
            b64_str = primary.face_crop_base64.split(",", 1)[-1]
            crop_bytes = base64.b64decode(b64_str)
            crop_img = Image.open(io.BytesIO(crop_bytes))
        except Exception:
            crop_img = None

    # Decision Badge Styling
    decision = primary.effective_decision
    color_map = {
        "PASS": ("#10b981", "#ecfdf5", "PASS / ACCEPT"),
        "REVIEW": ("#f59e0b", "#fffbeb", "REVIEW NEEDED"),
        "REJECT": ("#ef4444", "#fef2f2", "REJECTED"),
        "NO_FACE": ("#64748b", "#f8fafc", "NO FACE DETECTED"),
        "NOT_EVALUATED": ("#6366f1", "#eef2ff", "NOT EVALUATED (Uncalibrated)"),
        "ERROR": ("#dc2626", "#fef2f2", "PROCESSING ERROR"),
    }
    border_col, bg_col, label_text = color_map.get(decision, ("#64748b", "#f8fafc", decision))

    score_display = (
        f"{primary.quality_score:.4f}" if primary.quality_score is not None else "N/A"
    )
    conf_display = f"{primary.confidence * 100:.1f}%" if primary.confidence is not None else "N/A"

    badge_html = f"""
    <div style="padding: 16px; border-radius: 12px; border: 2px solid {border_col}; background-color: {bg_col}; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #475569; letter-spacing: 0.05em;">Gate Decision</span>
            <span style="background-color: {border_col}; color: white; padding: 4px 10px; border-radius: 20px; font-weight: 700; font-size: 12px;">{label_text}</span>
        </div>
        <div style="display: flex; gap: 16px; align-items: baseline;">
            <div>
                <span style="font-size: 11px; color: #64748b; display: block;">eDifFIQA Quality Score</span>
                <span style="font-size: 28px; font-weight: 800; font-family: monospace; color: #0f172a;">{score_display}</span>
            </div>
            <div>
                <span style="font-size: 11px; color: #64748b; display: block;">Detection Confidence</span>
                <span style="font-size: 20px; font-weight: 700; font-family: monospace; color: #334155;">{conf_display}</span>
            </div>
        </div>
        <div style="font-size: 12px; color: #334155; margin-top: 8px; border-top: 1px solid rgba(0,0,0,0.06); padding-top: 6px;">
            <strong>Decision Note:</strong> {primary.decision_reason or primary.error_reason or 'No evaluation reason recorded'}
        </div>
    </div>
    """

    # Pose & Diagnostics Markdown
    pose_str = (
        f"**Pitch**: {primary.pitch:.1f}° | **Yaw**: {primary.yaw:.1f}° | **Roll**: {primary.roll:.1f}°"
        if primary.pose_status == "OK" and primary.pitch is not None
        else f"Status: {primary.pose_status}"
    )

    gaze_str = (
        f"**Pitch (Vert)**: {primary.gaze_pitch:.1f}° | **Yaw (Horiz)**: {primary.gaze_yaw:.1f}°"
        if primary.gaze_status == "OK" and primary.gaze_pitch is not None
        else f"Status: {primary.gaze_status}"
    )

    sharpness_str = (
        f"{primary.sharpness_laplacian_var:.1f}"
        if primary.sharpness_laplacian_var is not None
        else "N/A"
    )
    mean_lum = f"{primary.mean_luminance:.1f} / 255" if primary.mean_luminance is not None else "N/A"
    under_pct = (
        f"{primary.underexposure_ratio*100:.1f}%"
        if primary.underexposure_ratio is not None
        else "N/A"
    )
    over_pct = (
        f"{primary.overexposure_ratio*100:.1f}%"
        if primary.overexposure_ratio is not None
        else "N/A"
    )
    asym = (
        f"{primary.illumination_asymmetry:.3f}"
        if primary.illumination_asymmetry is not None
        else "N/A"
    )

    diagnostics_md = f"""
    ### 🧭 Pose & Line-of-Sight Gaze
    - **Head Pose ({primary.pose_status})**: {pose_str}
    - **MobileGaze ({primary.gaze_status})**: {gaze_str}

    ### 🔬 Photometric Clarity & Exposure
    - **Laplacian Sharpness Variance**: `{sharpness_str}` *(Higher = crisper focus)*
    - **Mean Luminance**: `{mean_lum}`
    - **Exposure Extremes**: Underexposed: `{under_pct}` | Overexposed: `{over_pct}`
    - **Illumination Asymmetry**: `{asym}` *(Left vs Right face ratio)*
    - **Inference Latency**: `{primary.processing_time_ms:.1f} ms`
    """

    # Structured JSON
    json_summary = {
        "status": primary.status,
        "detector": primary.detector,
        "gate_decision": primary.gate_decision,
        "effective_decision": primary.effective_decision,
        "review_policy": primary.review_policy,
        "policy_override": primary.policy_override,
        "quality_score": primary.quality_score,
        "confidence": primary.confidence,
        "bbox": primary.bbox,
        "face_area_ratio": primary.face_area_ratio,
        "inter_eye_distance_px": primary.inter_eye_distance_px,
        "total_faces_detected": len(results),
        "timing_breakdown_ms": {
            "detection": primary.detection_time_ms,
            "quality": primary.quality_time_ms,
            "pose": primary.pose_time_ms,
            "gaze": primary.gaze_time_ms,
            "photometrics": primary.photometrics_time_ms,
            "total": primary.processing_time_ms,
        },
    }

    return annotated, crop_img, badge_html, diagnostics_md, json_summary


# ---------------------------------------------------------------------------
# Batch Analysis Callback
# ---------------------------------------------------------------------------
@spaces.GPU(duration=120)
def analyze_batch_faces(
    files: list[Any] | None,
    detector: str,
    detection_mode: str,
    scrfd_size: int,
    confidence_thresh: float,
    head_pose: bool,
    gaze: bool,
    review_policy: str,
    progress=gr.Progress(),
):
    if not files or len(files) == 0:
        return "### ⚠️ No files uploaded.", None, None, None

    if len(files) > 100:
        return (
            f"### ❌ Error: Batch of {len(files)} exceeds maximum limit of 100 images.",
            None,
            None,
            None,
        )

    all_faces = []
    rows = []
    total = len(files)

    for i, file_obj in enumerate(files):
        progress((i / total), desc=f"Analyzing image {i+1} of {total}...")
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        filename = Path(file_path).name

        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()

            res_list = pipeline_service.analyze_image_bytes(
                image_bytes=img_bytes,
                filename=filename,
                detector=detector,
                fallback_detector="retinaface" if detector == "scrfd" else "scrfd",
                detection_mode=detection_mode,
                scrfd_input_size=int(scrfd_size),
                confidence_threshold=float(confidence_thresh),
                head_pose=head_pose,
                gaze=gaze,
                review_policy=review_policy,
            )
            all_faces.extend(res_list)

            for f_item in res_list:
                rows.append(
                    {
                        "Filename": f_item.image_name,
                        "Face #": f_item.face_index or 1,
                        "Status": f_item.status,
                        "Gate Decision": f_item.gate_decision,
                        "Effective Decision": f_item.effective_decision,
                        "Quality Score": (
                            f"{f_item.quality_score:.4f}"
                            if f_item.quality_score is not None
                            else "-"
                        ),
                        "Confidence": (
                            f"{f_item.confidence:.3f}" if f_item.confidence is not None else "-"
                        ),
                        "Pitch": f"{f_item.pitch:.1f}°" if f_item.pitch is not None else "-",
                        "Yaw": f"{f_item.yaw:.1f}°" if f_item.yaw is not None else "-",
                        "Roll": f"{f_item.roll:.1f}°" if f_item.roll is not None else "-",
                        "Sharpness": (
                            f"{f_item.sharpness_laplacian_var:.1f}"
                            if f_item.sharpness_laplacian_var is not None
                            else "-"
                        ),
                        "Latency (ms)": f"{f_item.processing_time_ms:.0f}",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "Filename": filename,
                    "Face #": "-",
                    "Status": "READ_ERROR",
                    "Gate Decision": "ERROR",
                    "Effective Decision": "ERROR",
                    "Quality Score": "-",
                    "Confidence": "-",
                    "Pitch": "-",
                    "Yaw": "-",
                    "Roll": "-",
                    "Sharpness": "-",
                    "Latency (ms)": "-",
                }
            )

    progress(1.0, desc="Evaluation completed!")

    # Summary metrics
    pass_cnt = sum(1 for r in rows if r["Effective Decision"] in ("PASS", "ACCEPT"))
    rev_cnt = sum(1 for r in rows if r["Effective Decision"] == "REVIEW")
    rej_cnt = sum(1 for r in rows if r["Effective Decision"] == "REJECT")
    no_face_cnt = sum(1 for r in rows if r["Effective Decision"] == "NO_FACE")
    err_cnt = sum(1 for r in rows if r["Effective Decision"] == "ERROR")

    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px;">
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
            <span style="font-size: 11px; color: #64748b; font-weight: 600;">Images</span>
            <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{total}</div>
        </div>
        <div style="background: #ecfdf5; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; text-align: center;">
            <span style="font-size: 11px; color: #166534; font-weight: 600;">PASS</span>
            <div style="font-size: 22px; font-weight: 800; color: #15803d;">{pass_cnt}</div>
        </div>
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; text-align: center;">
            <span style="font-size: 11px; color: #92400e; font-weight: 600;">REVIEW</span>
            <div style="font-size: 22px; font-weight: 800; color: #b45309;">{rev_cnt}</div>
        </div>
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px; text-align: center;">
            <span style="font-size: 11px; color: #991b1b; font-weight: 600;">REJECT</span>
            <div style="font-size: 22px; font-weight: 800; color: #b91c1c;">{rej_cnt}</div>
        </div>
        <div style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; text-align: center;">
            <span style="font-size: 11px; color: #475569; font-weight: 600;">NO FACE / ERR</span>
            <div style="font-size: 22px; font-weight: 800; color: #334155;">{no_face_cnt + err_cnt}</div>
        </div>
    </div>
    """

    df = pd.DataFrame(rows)

    # Export Passed Faces Zip
    batch_id = f"batch_{len(all_faces)}"
    with pipeline_service.lock:
        from api.schemas import BatchAnalysisResult
        pipeline_service._batches[batch_id] = BatchAnalysisResult(
            batch_id=batch_id,
            created_at="now",
            total_images=total,
            processed_images=total,
            successful_evaluations=len(all_faces),
            failures=err_cnt,
            pass_count=pass_cnt,
            review_count=rev_cnt,
            reject_count=rej_cnt,
            no_face_count=no_face_cnt,
            error_count=err_cnt,
            total_faces=len(all_faces),
            review_policy=review_policy,
            results=all_faces,
        )

    zip_path, passed_count = pipeline_service.generate_passed_faces_zip(batch_id, review_policy)
    return summary_html, df, zip_path, f"Passed Faces Export ready ({passed_count} faces included)"


# ---------------------------------------------------------------------------
# Gradio Blocks Application Layout
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(title="Face Image Quality Assessment (FIQA)") as demo:
    gr.Markdown(
        """
        # 👁️ Face Image Quality Assessment & Surveillance Platform
        ### Powered by **eDifFIQA-S** (IResNet-18) & **UniFace** (SCRFD / RetinaFace Cascades)
        Evaluate whether detected face images satisfy downstream face recognition requirements using quality scoring, head-pose tolerances, eye-gaze tracking, and explainable photometric clarity.
        """
    )

    with gr.Tabs():
        # TAB 1: Single Image
        with gr.TabItem("🔍 Single Image Assessment"):
            with gr.Row():
                with gr.Column(scale=5):
                    single_input = gr.Image(type="numpy", label="Upload or Capture Face Image")

                    with gr.Accordion("⚙️ Detector & Model Parameters", open=True):
                        with gr.Row():
                            detector_drop = gr.Dropdown(
                                choices=["scrfd", "retinaface"],
                                value="scrfd",
                                label="Primary Detector",
                            )
                            mode_drop = gr.Dropdown(
                                choices=["fallback", "complementary"],
                                value="fallback",
                                label="Detection Mode",
                            )
                        with gr.Row():
                            scrfd_size = gr.Dropdown(
                                choices=[320, 640, 960, 1280],
                                value=640,
                                label="Resolution Scale (px)",
                            )
                            conf_slider = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.05,
                                label="Confidence Threshold",
                            )
                        with gr.Row():
                            pose_chk = gr.Checkbox(value=True, label="UniFace Head Pose")
                            gaze_chk = gr.Checkbox(value=True, label="MobileGaze Line-of-Sight")
                            review_pol = gr.Dropdown(
                                choices=["manual_review", "treat_as_pass", "treat_as_reject"],
                                value="manual_review",
                                label="Review Policy",
                            )

                    analyze_btn = gr.Button("⚡ Analyze Face Quality", variant="primary", size="lg")

                    # Preset test images
                    example_dir = ROOT_DIR / "assets" / "test_images"
                    if example_dir.exists():
                        examples = [str(p) for p in example_dir.glob("*.jpg")][:5]
                        if examples:
                            gr.Examples(examples=examples, inputs=single_input)

                with gr.Column(scale=7):
                    annotated_out = gr.Image(label="Detection Visualizer (Annotated Faces)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            crop_out = gr.Image(label="112×112 Aligned Crop", width=128, height=128)
                        with gr.Column(scale=3):
                            badge_out = gr.HTML(label="Quality Decision")

                    diagnostics_out = gr.Markdown(label="Diagnostics & Signals")
                    with gr.Accordion("📋 Raw JSON Output", open=False):
                        json_out = gr.JSON(label="Pipeline Diagnostics")

            analyze_btn.click(
                fn=analyze_single_face,
                inputs=[
                    single_input,
                    detector_drop,
                    mode_drop,
                    scrfd_size,
                    conf_slider,
                    pose_chk,
                    gaze_chk,
                    review_pol,
                ],
                outputs=[annotated_out, crop_out, badge_out, diagnostics_out, json_out],
            )

        # TAB 2: Batch Evaluation
        with gr.TabItem("📦 Batch Evaluation (Up to 100 Images)"):
            gr.Markdown(
                "Upload up to **100 images** to evaluate entire batches with partial failure resilience, review policies, and compliant face crop ZIP export."
            )
            with gr.Row():
                with gr.Column(scale=4):
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Image Batch (Max 100)",
                    )
                    batch_review_pol = gr.Dropdown(
                        choices=["manual_review", "treat_as_pass", "treat_as_reject"],
                        value="manual_review",
                        label="Review Policy",
                    )
                    batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                    batch_zip_out = gr.File(label="⬇️ Download Passed Faces (.zip)")

                with gr.Column(scale=8):
                    batch_summary = gr.HTML(label="Summary Metrics")
                    batch_table = gr.DataFrame(label="Batch Results Table")

            batch_btn.click(
                fn=analyze_batch_faces,
                inputs=[
                    batch_files,
                    detector_drop,
                    mode_drop,
                    scrfd_size,
                    conf_slider,
                    pose_chk,
                    gaze_chk,
                    batch_review_pol,
                ],
                outputs=[batch_summary, batch_table, batch_zip_out, gr.Textbox(visible=False)],
            )

        # TAB 3: Architecture & Model Documentation
        with gr.TabItem("📖 Architecture & NIST Benchmarks"):
            gr.Markdown(
                """
                ## 🏛️ System Architecture
                - **eDifFIQA-S (IResNet-18)**: Pre-trained quality estimation backbone predicting a single scalar score from aligned 112×112 faces.
                - **Dual Detection Cascade**: UniFace SCRFD primary detector paired with RetinaFace fallback & complementary ensemble to eliminate missed faces in crowded scenes.
                - **Deterministic 6-State Gate**: `PASS`, `REVIEW`, `REJECT`, `NO_FACE`, `NOT_EVALUATED`, and `ERROR`.
                - **Photometric Explainability**: Laplacian variance for blur/sharpness, luminance histograms for underexposure/overexposure, and bilateral symmetry.
                - **UniFace MobileGaze**: Distinct eye line-of-sight direction estimation separated from head pose.
                - **Export Safety**: Filtered ZIP archive containing only passed face crops, complete with `manifest.csv` and `manifest.json`.
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme)
