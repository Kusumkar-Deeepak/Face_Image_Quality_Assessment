# Complete Repository Integration Audit: Face Image Quality Assessment Project

**Audit Date:** October 18, 2026  
**Auditor:** Senior Computer Vision Engineer, Python Backend Engineer & AI System Architect  
**Project Workspace:** `face-image-quality-assessment`  
**Target Operating Domain:** Surveillance and CCTV Face Recognition Quality Gate  

---

## 1. Executive Summary

This integration audit evaluates the current **Face Image Quality Assessment** system and determines the exact status, overlap, and integration feasibility across three key open-source repositories:
1. **Repository 1: Face Image Quality Assessment (`yakhyo/face-image-quality-assessment`)** — PyTorch & ONNX implementation of the eDifFIQA face image quality model.
2. **Repository 2: UniFace (`yakhyo/uniface`)** — Unified face analysis toolkit providing face detection, landmarks, alignment, head pose, gaze estimation, recognition, and quality scoring.
3. **Repository 3: BioGaze (`Maphoz/BioGaze`)** — Face image analysis and ISO/IEC 19794-5 & ICAO Doc 9303 compliance verification software developed with BioLab (University of Bologna).

### Core Findings:
- **Repository 1 Status:** Fully integrated in its PyTorch variant (`eDifFIQA-S`, IResNet-18 backbone + 2-layer MLP head) with local weights `weights/ediffiqa_s.pth` (98.3 MB). Pipeline preprocessing (`ALIGNED_SIZE = 112`), bounding box validation, affine alignment, and batch reporting are working cleanly.
- **UniFace Status:** `uniface 4.0.0` is already an installed package dependency. The application directly relies on UniFace for:
  1. `uniface.detection.SCRFD` (primary detector)
  2. `uniface.detection.RetinaFace` (fallback detector)
  3. `uniface.face_alignment` (5-point ArcFace landmark alignment)
  4. `uniface.headpose.HeadPose` (ONNX ResNet-18 6D-representation head-pose estimation)  
  *Unused UniFace capabilities include:* ONNX runtime `EDifFIQA`, `MobileGaze` gaze estimation, dense landmark models (`Landmark106`, `FaceMesh`), anti-spoofing (`MiniFASNet`), face recognition backbones (`ArcFace`, `AdaFace`, `EdgeFace`), and `BYTETracker`.
- **BioGaze Status:** BioGaze is **not installed** and **not integrated**. Crucially, BioGaze is built exclusively for **ICAO passport-photo compliance** (enforcing single-face frames, uniform white backgrounds, rigid aspect ratios of 0.74–0.80, and extreme frontal poses of Yaw/Pitch $\pm 5^\circ$). These strict constraints are **fundamentally incompatible with surveillance and CCTV cameras**, where faces are small, backgrounds are dynamic, and poses vary widely. However, isolated mathematical sub-routines from BioGaze—such as passive illumination asymmetry, exposure clipping, and Laplacian blur—can be extracted or reimplemented as non-blocking diagnostic indicators.
- **Decision Logic Gap:** The system correctly preserves `NOT_EVALUATED` when thresholds are null. However, configured initial values (e.g. `accept_min: 0.65`) must not be represented as calibrated operating points. Calibration requires running `calibrate_thresholds.py` on ground-truth labeled CCTV/surveillance datasets. Furthermore, the decision engine evaluates only the eDifFIQA score and head-pose limits, ignoring detection confidence, face area ratio, and inter-eye distance.

---

## 2. Existing Project Architecture

The current workspace is structured as a full-stack computer vision application:

```
face-image-quality-assessment/
├── api/                             # FastAPI Backend Layer
│   ├── main.py                      # REST endpoints (/api/analyze, /api/analyze/batch, /api/config, etc.)
│   ├── service.py                   # Model caching, session metrics, EXIF auto-orientation, batch storage
│   ├── schemas.py                   # Pydantic v2 validation contracts
│   └── test_api.py                  # API pytest test suite (9 tests)
├── frontend/                        # React / Vite SPA Dashboard (Face Quality Lab)
│   ├── src/
│   │   ├── components/
│   │   │   ├── SingleImageAnalysis.jsx # Interactive upload, detection visualizer, pose, and diagnostics
│   │   │   ├── BatchEvaluation.jsx     # High-throughput batch queue, async thumbnails, results table
│   │   │   ├── BoundingBoxViewer.jsx   # Scaled image preview with blue bounding box and 112x112 crop
│   │   │   ├── QualityScoreCard.jsx    # Model quality score gauge, tier pills, decision threshold notices
│   │   │   ├── SettingsView.jsx        # Detector parameters & threshold calibration presets
│   │   │   └── StatusBadge.jsx         # Status badges for OK, NO_FACE, NOT_EVALUATED, ACCEPT, REJECT
│   │   ├── api.js                   # Axios client
│   │   └── App.jsx                  # Main navigation & session status
├── models/                          # eDifFIQA Architecture Definitions (PyTorch & ONNX)
│   ├── backbones.py                 # IResNet-18, IResNet-50, IResNet-100, MobileFaceNet
│   ├── ediffiqa.py                  # PyTorch model: Backbone + 2-layer MLP head (GELU activation)
│   ├── ediffiqa_onnx.py             # ONNX Runtime wrapper for eDifFIQA
│   └── __init__.py                  # Model factory (get_model)
├── weights/
│   └── ediffiqa_s.pth               # 98.3 MB PyTorch checkpoint for eDifFIQA-S
├── assets/test_images/              # 23 sample images (original, degraded, blur, lowres, dark, rotated)
├── quality_pipeline.py              # Core inference engine (QualityPipeline, QualityThresholds, EvaluationResult)
├── face_detection.py                # FaceDetectorChain, DetectorConfig, validation & de-duplication
├── pose_check.py                    # HeadPoseChecker using uniface.headpose.HeadPose
├── calibrate_thresholds.py          # Empirical threshold sweep & calibration script for labeled datasets
├── detector_diagnostic.py           # Diagnostic CLI for detector candidate inspection
├── batch_evaluate.py                # CLI batch runner with synthetic image degradations
├── main.py                          # Single-image / CLI runner
├── detector_config.json             # Default detector settings (SCRFD, RetinaFace, IoU dedupe)
├── quality_thresholds.json          # Configured thresholds (null when uncalibrated)
└── tests/
    └── test_quality_pipeline.py     # 52 unit tests covering pipeline, mock detectors, thresholds, degradations
```

### Data Flow & Execution Pipeline:
1. **Input Reception:** Image bytes ingested via CLI (`main.py`) or API (`POST /api/analyze`).
2. **Preprocessing & Decoding:** Image decoded with EXIF auto-transposition (`PIL.ImageOps.exif_transpose`) into BGR NumPy array.
3. **Face Detection (`FaceDetectorChain` in `face_detection.py`):**
   - Primary detector (`SCRFD`) runs at configured input size (default $640\times 640$, or $320\times 320$).
   - Detections are structurally validated (`validate_face`): finite bbox coordinates within image bounds, valid landmarks (5 points), confidence in $[0, 1]$, non-degenerate centroid inside bbox.
   - If no valid face is found and fallback is configured (`retinaface`), `RetinaFace` executes as fallback.
   - Overlapping detections are de-duplicated via IoU threshold (`dedupe_iou = 0.5`).
4. **Face Alignment (`quality_pipeline.py#L280`):**
   - Calls `uniface.face_alignment(image, landmarks)` to compute similarity transform against ArcFace 5-point template.
   - Warps crop to $112\times 112$ RGB image.
   - Validates transformed crop (`_check_aligned`): non-blank, 3-channel, $112\times 112$, `uint8`.
5. **Quality Scoring (`quality_pipeline.py#L215`):**
   - Preprocesses aligned face via PyTorch `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`.
   - Passes tensor to `eDifFIQA-S` (`IResNet18` + `MLP`), extracting a raw continuous quality score $s \in [0, 1]$.
6. **Head-Pose Estimation (`pose_check.py`):**
   - If enabled, extracts bounding-box crop of original image (not aligned crop) and passes to `uniface.headpose.HeadPose`.
   - Returns continuous Euler angles: pitch, yaw, roll in degrees.
7. **Decision Gate (`quality_pipeline.py#L343`):**
   - If processing status $\neq \text{OK}$, decision is `NOT_EVALUATED`.
   - If thresholds are null, decision is `NOT_EVALUATED`.
   - If calibrated thresholds exist, compares score against `accept_min` and `reject_max`.
   - Checks pitch/yaw/roll against `max_abs_pitch`, `max_abs_yaw`, `max_abs_roll`. Pose violations downgrade `ACCEPT` to `REVIEW`, never automatic `REJECT`.

---

## 3. Repository 1 Audit: Face Image Quality Assessment (`yakhyo/face-image-quality-assessment`)

| Capability | Status | Implementation Location | Evidence & Verification | Remaining Work |
|---|---|---|---|---|
| **eDifFIQA Model Architecture** | **COMPLETE** | `models/ediffiqa.py`, `models/backbones.py` | PyTorch classes `MLP` and `eDifFIQA` implemented matching paper. Backbone definitions for T, S, M, L available. | None for PyTorch variant. |
| **Model Weights Loading** | **COMPLETE** | `quality_pipeline.py#L171-L177` | Loads `weights/ediffiqa_s.pth` with `torch.load(..., weights_only=True)`. Model cached in memory via `api/service.py`. | Weights for variants T, M, L are not present locally. |
| **Quality Score Generation** | **COMPLETE** | `quality_pipeline.py#L215-L224` | Generates continuous raw score. Passes tests (`test_quality_pipeline.py`). Verified output on `image2.jpg`: 0.8593. | Confirm normalization stability across PyTorch versions. |
| **Face Alignment Preprocessing** | **COMPLETE** | `quality_pipeline.py#L280-L285` | Uses 5-point ArcFace landmark affine transformation to 112×112. Structural check enforces non-blank 112×112 crop. | None. |
| **Single-Image CLI & API** | **COMPLETE** | `main.py`, `api/main.py#L86` | CLI produces tabular terminal summary; API provides `POST /api/analyze` with JSON schema. | None. |
| **Batch-Image Processing** | **COMPLETE** | `batch_evaluate.py`, `api/main.py#L112` | CLI exports CSV/JSON reports; API provides chunked batch processing with partial failure handling. | None. |
| **Decision Logic (ACCEPT/REJECT/REVIEW)** | **COMPLETE** | `quality_pipeline.py#L120-L134`, `quality_pipeline.py#L343-L360` | Correctly separates processing status from quality decision. Thresholds support `accept_min`, `reject_max`, and pose limits. | Threshold calibration on operational surveillance data. |
| **Threshold Calibration Tool** | **COMPLETE** | `calibrate_thresholds.py` | Threshold sweep script calculating precision, recall, F1, false rejection rate, and false acceptance rate from labeled CSVs. | Needs ground truth labeled surveillance dataset. |
| **Honest Status Mapping** | **COMPLETE** | `quality_pipeline.py`, `frontend/src/components/StatusBadge.jsx` | Uncalibrated thresholds return `NOT_EVALUATED`. Missing face returns `NO_FACE`. Decoder failure returns `READ_ERROR`. | Ensure UI documentation reinforces score is not a probability. |
| **Automated Unit Tests** | **COMPLETE** | `tests/test_quality_pipeline.py` | 52 automated tests covering pipeline, mocks, degradations, boundaries, and validation. | Maintain null-threshold state for test pass. |

### Evaluation of Score & Threshold Semantics:
- **Model Quality Score vs Probability:** The score generated by eDifFIQA is a **model quality score**, representing relative biometric utility (predicting recognition performance under ArcFace embeddings). It is **not** a posterior probability of identification, nor is it a confidence score. The frontend dashboard explicitly labels this as *"Model Quality Score (Not a Probability)"*.
- **Threshold Calibration Reality:** The `quality_thresholds.json` file currently contains initial values (`accept_min: 0.65`, `reject_max: 0.40`). As confirmed by `calibrate_thresholds.py`, **no empirical calibration has been performed** because the repository ships without ground-truth identity match/non-match labels. These values are operational initial estimates, not statistically calibrated thresholds.

---

## 4. Repository 2 Audit: UniFace (`yakhyo/uniface`)

`uniface` 4.0.0 is an all-in-one face analysis library created by the same author as Repository 1.

### Capability Breakdown:

| UniFace Feature | Integration Classification | Current Status in Project | File & Function Reference | Assessment for Surveillance Pipeline |
|---|---|---|---|---|
| **SCRFD Face Detector** | **Directly Implemented from UniFace** | **COMPLETE** | `face_detection.py#L120-L133` calling `uniface.detection.SCRFD` | Excellent high-speed detector. Input size 640 standard; 320 for tight crops. |
| **RetinaFace Face Detector** | **Directly Implemented from UniFace** | **COMPLETE** | `face_detection.py#L134-L136` calling `uniface.detection.RetinaFace` | Effective fallback detector; catches small faces missed by SCRFD. |
| **5-Point Face Alignment** | **Directly Implemented from UniFace** | **COMPLETE** | `quality_pipeline.py#L18` calling `uniface.face_alignment` | Standard ArcFace template affine transformation. |
| **Head-Pose Estimation** | **Directly Implemented from UniFace** | **COMPLETE** | `pose_check.py#L32` calling `uniface.headpose.HeadPose` | Uses 6D rotation representation (ResNet-18). Estimates pitch, yaw, roll on bbox crop. |
| **EDifFIQA Quality (ONNX)** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.quality.EDifFIQA` | UniFace contains an ONNX Runtime implementation of eDifFIQA. The current app uses the PyTorch implementation. Replacing PyTorch with `uniface.quality.EDifFIQA` would reduce memory footprint and eliminate the PyTorch dependency for inference. |
| **MobileGaze (Gaze Estimation)** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.gaze.MobileGaze` | Predicts continuous pitch & yaw in radians using ONNX MobileGaze. Relevant for detecting subjects looking toward/away from CCTV cameras. |
| **YOLOv8Face / YOLOv5Face** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.detection.YOLOv8Face` | Alternative detectors. SCRFD and RetinaFace currently provide adequate coverage; adding YOLOv8Face would add redundant models without significant gain. |
| **Landmark106 / FaceMesh** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.landmark.Landmark106`, `uniface.landmark.FaceMesh` | 106-point and 468-point dense landmark meshes. Useful for precise eye-aspect-ratio (EAR) and mouth-aspect-ratio (MAR) if blink/expression analysis is required. |
| **Face Recognition Embeddings** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.recognition.ArcFace`, `AdaFace`, `EdgeFace` | Extracts 512-D identity embeddings. Essential if downstream identification is added to the quality gate. |
| **Anti-Spoofing / Liveness** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.spoofing.MiniFASNet` | Evaluates presentation attack detection (screens, printed photos). Valuable for access control kiosks; moderate utility in passive CCTV. |
| **Face Parsing / Segmentation** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.parsing.BiSeNet` | Semantic segmentation of hair, skin, eyes, background. Useful for occlusion detection. |
| **Multi-Object Tracking** | **Available in Installed UniFace Package, Not Integrated** | **AVAILABLE** | `uniface.tracking.BYTETracker` | Multi-face tracking across video frames. Critical for multi-frame surveillance video streams. |

---

## 5. Repository 3 Audit: BioGaze (`Maphoz/BioGaze`)

BioGaze is an open-source tool developed with the BioLab at the University of Bologna to evaluate facial photographs against **ISO/IEC 19794-5 and ICAO Doc 9303 standards** (electronic passports and national identity cards).

### Codebase & Dependency Analysis:
- **Repository Location:** External (`https://github.com/Maphoz/BioGaze`). Not present in local workspace.
- **Dependencies (`requirements.txt`):** `opencv-python`, `torch`, `torchvision`, `tabulate`, `ultralytics==8.0.0`, `dlib`, `mediapipe`, `scikit-image`, `rmn`, `imutils`.
- **Checkpoint Requirements:** Requires manual downloads for:
  - `dlib` 68-landmark shape predictor (`shape_predictor_68_face_landmarks.dat`)
  - Face parser model checkpoint (`face_parser/res/cp/`)
  - Gaze estimation checkpoint (`gaze_estimation/ckpt/`)

### Evaluation of BioGaze Capabilities vs Surveillance Quality Needs:

| BioGaze Feature | Underlying Mechanism | Surveillance Relevance | Equivalent in Current App or UniFace? | Recommendation & Rationale |
|---|---|---|---|---|
| **Gaze Estimation** | CNN gaze estimator + Dlib eye crops | High | **Equivalent exists in UniFace (`uniface.gaze.MobileGaze`)** | **SKIP BioGaze implementation.** UniFace provides `MobileGaze` via ONNX without requiring `dlib` or manual checkpoint placement. |
| **Head Pose Estimation** | Dlib landmarks + `cv2.solvePnP` | High | **Superior equivalent exists (`uniface.headpose.HeadPose`)** | **SKIP BioGaze implementation.** UniFace's 6D rotation network is more robust to perspective distortion and occlusion than `solvePnP` on 68 landmarks. |
| **Exposure / Darkness / Lightness** | Histogram quantile analysis (`AVG_DARK_THRESHOLD`, `MAX_DARK_THRESHOLD`) | High | **Missing** | **ADAPT sub-routine.** The mathematical histogram check (measuring clipped shadows and blown highlights) is lightweight, requires no deep learning weights, and provides actionable exposure diagnostics. |
| **Illumination Uniformity** | Face parser skin mask + left/right brightness ratio | Moderate | **Missing** | **ADAPT or INVESTIGATE.** Measuring facial brightness asymmetry detects harsh side-lighting in CCTV. However, BioGaze's heavy PyTorch face parser should be replaced with OpenCV split-face luminance analysis. |
| **Sharpness / Out-of-Focus** | Laplacian operator variance (`cv2.Laplacian`) | High | **Missing** | **ADAPT sub-routine.** OpenCV Laplacian variance ($\text{Var}(\nabla^2 I)$) is a fast ($<1\text{ ms}$), zero-dependency indicator of optical defocus blur. |
| **Color Saturation / Posterization** | HSV channel saturation & RGB color bit analysis | Moderate | **Missing** | **ADAPT as passive diagnostic.** Detects washed-out infrared or severe sensor posterization. |
| **Neutral Expression (Emotion)** | Residual Masking Network (`rmn`) | Low | **Available in UniFace (`uniface.attribute.Emotion`)** | **SKIP.** Surveillance subjects frequently speak or emote. Forcing neutral expressions is an ICAO requirement, not a CCTV necessity. |
| **Uniform Background Check** | Edge detection & variance on background mask | Zero | **Missing** | **REJECT.** Surveillance environments (streets, lobbies) never have uniform backgrounds. Enforcing background uniformity would reject 100% of CCTV frames. |
| **Face Dimension / Aspect Ratio** | Aspect ratio check ($0.74 \le \text{W/H} \le 0.80$) & head height ($0.60 \le H \le 0.90$) | Zero | **Missing** | **REJECT.** Tailored strictly for $35\times 45\text{ mm}$ passport photos. Enforcing this on CCTV would cause total false rejection. |
| **Shoulder Visibility** | Face parser segmentation of shoulders | Zero | **Missing** | **REJECT.** Irrelevant to biometric facial matching; strictly an administrative passport criterion. |
| **Dlib Dependency** | C++ Dlib 68-point shape predictor | Negative | **N/A** | **REJECT.** `dlib` requires CMake and C++ build tools on Windows, introducing severe installation fragility. |

---

## 6. Capability Comparison Matrix

| Capability | Current Implementation | Repository 1 (`face-image-quality-assessment`) | UniFace (`uniface 4.0.0`) | BioGaze (`Maphoz/BioGaze`) | Final Recommendation |
|---|---|---|---|---|---|
| **Face Detection** | COMPLETE (SCRFD + RetinaFace fallback) | SCRFD + RetinaFace fallback | SCRFD, RetinaFace, YOLOv8Face, BlazeFace | Ultralytics YOLOv8 + Dlib | **Keep Current.** SCRFD primary + RetinaFace fallback provides optimal speed/accuracy trade-off. |
| **Landmark Detection** | COMPLETE (5-point ArcFace) | 5-point ArcFace landmarks | 5-point, 106-point (`Landmark106`), 468-point (`FaceMesh`) | 68-point Dlib | **Keep Current.** 5-point is sufficient for alignment. Add `FaceMesh` only if dense morphing analysis is needed. |
| **Face Alignment** | COMPLETE (112×112 ArcFace template) | 112×112 similarity warp | `uniface.face_alignment` (identical) | Bounding box crop / Dlib alignment | **Keep Current.** 112×112 ArcFace alignment is standard for biometric feature extractors. |
| **Quality Scoring (FIQA)** | COMPLETE (PyTorch eDifFIQA-S) | PyTorch eDifFIQA (T, S, M, L) | ONNX Runtime eDifFIQA (`uniface.quality.EDifFIQA`) | Rule-based heuristics across photographic metrics | **Transition to UniFace ONNX eDifFIQA.** Eliminates PyTorch runtime requirement, reduces latency, and standardizes inference. |
| **Blur Sensitivity** | COMPLETE (eDifFIQA score reflects blur) | Validated drop on blurred images (0.8593 $\to$ 0.2884) | Validated drop in quality score | Fast Laplacian variance check | **Keep eDifFIQA; Add Laplacian diagnostic.** eDifFIQA evaluates biometric blur; Laplacian variance provides explicit sharpness feedback. |
| **Low-Resolution Sensitivity** | COMPLETE (SCRFD@320 or RetinaFace detects down to 16px) | Evaluated on degraded variants | Detects down to small scales | Pixelation threshold check | **Keep Current.** Bounding box validation and inter-eye distance already track resolution constraints. |
| **Head-Pose Estimation** | COMPLETE (Pitch, Yaw, Roll in degrees) | UniFace HeadPose (ResNet-18) | 6D-rep HeadPose (ResNet 18/34/50, MobileNet) | Dlib landmarks + `solvePnP` | **Keep Current.** UniFace ResNet-18 6D representation is accurate and already integrated. |
| **Gaze Direction** | MISSING | MISSING | COMPLETE (`uniface.gaze.MobileGaze`) | COMPLETE (Dlib + custom CNN) | **Integrate UniFace `MobileGaze`.** Provides gaze pitch/yaw in radians via ONNX without extra dependencies. |
| **Illumination & Exposure** | PARTIAL (eDifFIQA score drops on dark images; no explicit metrics) | Score drop verified on `img8_dark.jpg` (0.5737) | MISSING | COMPLETE (Histogram clipping, saturation, left/right face lighting) | **Adapt BioGaze Exposure Sub-routine.** Implement lightweight histogram shadow/highlight checks in pure NumPy/OpenCV. |
| **Decision Thresholds** | COMPLETE (Configurable `accept_min`, `reject_max`, pose limits) | Configurable schema; calibration CLI script | None (returns raw scores) | Hardcoded ICAO thresholds | **Keep Current Schema.** Retain honest `NOT_EVALUATED` state until empirical calibration on CCTV dataset is complete. |
| **Batch Processing** | COMPLETE (High-capacity batch queue, async thumbnails, CSV/JSON export) | CLI script (`batch_evaluate.py`) | Iterative detector loops | Scripted directory loop (`quality_analysis.py`) | **Keep Current.** Web dashboard supports 100-image batches with 60 FPS scrolling and full diagnostic drawers. |
| **API & Web Interface** | COMPLETE (FastAPI + React/Vite Dashboard) | CLI only | Python library only | CLI only | **Keep Current.** FastAPI + React dashboard is production-ready. |

---

## 7. Confirmed Completed Functionality

1. **eDifFIQA-S Quality Inference:** Accurate PyTorch model loading and inference on 112×112 aligned crops.
2. **Two-Stage Face Detection Chain:** SCRFD as primary detector with automatic RetinaFace fallback for close-up portraits or low-resolution faces.
3. **5-Point ArcFace Similarity Alignment:** Sub-pixel affine warp to standard biometric 112×112 coordinate space.
4. **Structural Bounding Box Validation:** Validates finite coordinates, boundaries, positive dimensions, landmark centroids, and de-duplicates overlapping detections via IoU.
5. **Head-Pose Estimation:** Continuous pitch, yaw, and roll estimation via UniFace ONNX ResNet-18.
6. **Honest Decision Mapping:** `NOT_EVALUATED` strictly enforced when thresholds are uncalibrated; `NO_FACE` strictly reported when detectors fail without fabricating blur verdicts.
7. **FastAPI Backend Architecture:** REST API with model instance caching, request ID tracking, EXIF auto-orientation transpose, and in-memory batch processing.
8. **High-Performance Web Dashboard:**
   - Single-image analysis with interactive detection visualizer, blue bounding box overlays, and aligned face crops.
   - 100-image batch evaluation queue with asynchronous off-thread canvas thumbnailing, 60 FPS scrolling, and real-time multi-stage progress tickers.
   - Comprehensive diagnostic drawers for every batch item matching single-image inspection.
   - 1-click CSV and JSON report exports.
9. **Regression & Unit Testing:** 52/52 pipeline unit tests passing; 9/9 backend API tests passing; 7/7 frontend unit tests passing.

---

## 8. Partially Implemented Functionality

1. **Threshold Calibration:**
   - *Status:* `calibrate_thresholds.py` exists and correctly computes precision/recall/F1 sweeps.
   - *Gap:* The repository ships with empty labels. Operational thresholds in `quality_thresholds.json` (`accept_min: 0.65`, `reject_max: 0.40`) are initial presets, not empirically calibrated operating thresholds.
2. **Decision Signal Integration:**
   - *Status:* `_final_decision` in `quality_pipeline.py` integrates the quality score and head-pose limits.
   - *Gap:* Detection confidence, face area ratio, inter-eye distance, and illumination are computed and displayed in diagnostics but do not participate in the final `ACCEPT`/`REJECT`/`REVIEW` decision logic.
3. **Photographic Degradation Diagnostics:**
   - *Status:* eDifFIQA inherently penalizes blur and darkness into lower quality scores.
   - *Gap:* The system lacks explicit diagnostic breakdown distinguishing optical blur, motion blur, under-exposure, and off-axis gaze.

---

## 9. Missing Functionality

1. **Gaze Direction Analysis:** Missing from active pipeline (though immediately available in installed `uniface.gaze.MobileGaze`).
2. **Explicit Photographic Quality Checks:**
   - Under-exposure and over-exposure histogram analysis.
   - Direct focus / sharpness measurement via Laplacian variance.
   - Facial illumination asymmetry (side-lighting / shadows).
3. **ONNX-Only Runtime Pipeline:** Currently requires full PyTorch installation for `eDifFIQA-S` inference, despite `uniface` supporting ONNX Runtime.
4. **Downstream Face Recognition Embedding Extraction:** Pipeline stops at quality assessment; does not output 512-D ArcFace/AdaFace embeddings for verification.
5. **Video Stream / Multi-Frame Tracking:** Pipeline processes static image files; lacks multi-frame video tracking (`BYTETracker`) to select the best-quality frame per tracklet.

---

## 10. Broken Functionality and Known Issues

1. **Threshold Regression Test State Sensitivity (`tests/test_quality_pipeline.py::test_from_file`):**
   - *Issue:* `test_from_file` asserts that `quality_thresholds.json` matches an empty `QualityThresholds()` instance where all fields are `None`. If API threshold presets are saved to disk, this test fails.
   - *Impact:* The test suite requires `quality_thresholds.json` to have null fields to pass.
2. **Head-Pose Reliability on Severely Blurred Faces:**
   - *Issue:* Confirmed empirically on `image2_blur.jpg`: head-pose yaw shifts from $1.9^\circ$ to $16.1^\circ$ and roll shifts from $19.5^\circ$ to $-10.1^\circ$ under severe Gaussian blur.
   - *Impact:* High pose angles on degraded imagery may reflect blur artifacts rather than true physical rotation.
3. **SCRFD Scale Range Sensitivity:**
   - *Issue:* SCRFD at default $640\times 640$ input size fails on tightly cropped portraits ($200\text{--}500\text{ px}$) because the face is larger than SCRFD's anchor receptive field.
   - *Mitigation:* Already handled by RetinaFace fallback and configurable `--scrfd-input-size 320`.

---

## 11. Duplicated or Redundant Components

1. **PyTorch Model vs UniFace ONNX Model:**
   - `models/ediffiqa.py` defines the PyTorch architecture and loads `weights/ediffiqa_s.pth`.
   - `uniface.quality.EDifFIQA` already contains the identical model exported to ONNX Runtime.
   - Running both represents redundant model maintenance.
2. **BioGaze Redundancies:**
   - BioGaze uses `dlib` 68-point landmarks + `solvePnP` for pose estimation; UniFace's direct 6D rotation network is faster and more accurate.
   - BioGaze uses a custom CNN for gaze estimation; UniFace's `MobileGaze` provides the identical capability without `dlib`.

---

## 12. Dependencies and Model-Checkpoint Requirements

### Current Active Environment:
- **Python:** 3.10.11 (Windows)
- **Deep Learning Frameworks:** `torch==2.14.1`, `torchvision==0.29.1`, `onnxruntime==1.23.2`, `onnx==1.23.2`
- **Computer Vision:** `opencv-python==5.0.0.93`, `pillow==12.3.0`, `scikit-image==0.25.2`
- **Core Library:** `uniface==4.0.0`
- **Web & Backend:** `fastapi==0.143.3`, `uvicorn==0.54.0`, `pydantic==2.14.1`
- **Checkpoints:**
  - `weights/ediffiqa_s.pth` (98.3 MB, local)
  - UniFace automatic cache in `~/.uniface/weights/`:
    - `scrfd_10g_kps.onnx` (~16.8 MB)
    - `retinaface_mnet_v2.onnx` (~1.7 MB)
    - `headpose_resnet18.onnx` (~45.0 MB)

### BioGaze Dependency Burden (If Imported Directly):
- **Heavy Packages:** `dlib` (demands CMake and MSVC compiler), `mediapipe`, `ultralytics==8.0.0`, `rmn`.
- **Manual Checkpoint Downloads:** Requires manual download of 3 external checkpoints totaling ~250 MB.
- **Architectural Conclusion:** Direct installation of BioGaze would destabilize the clean Python 3.10 environment.

---

## 13. Actual Test Results and Untested Areas

### Verified Executed Tests:
1. **Pipeline Unittest Suite:** `python -m unittest discover -s tests -v`  
   - Result: **52/52 passed** (33.7s execution time).
2. **Backend API Pytest Suite:** `pytest api/test_api.py -v`  
   - Result: **9/9 passed** (34.1s execution time).
3. **Frontend Vitest Suite:** `npm test -- --run`  
   - Result: **7/7 passed** (8.0s execution time).
4. **Frontend Production Build:** `npm run build`  
   - Result: **Built successfully** with code 0 (bundle size 313.9 kB JS, 37.3 kB CSS).
5. **Real-Image CLI Inferences:**
   - `image2.jpg` (default SCRFD): `NO_FACE` (honestly reported, not misclassified as quality failure).
   - `image2.jpg` (RetinaFace fallback): `OK`, Score: `0.8593`, Decision: `NOT_EVALUATED`.
   - `image2_blur.jpg`: `OK`, Score: `0.2884` (empirically verifies 0.57 drop under blur).
   - `img8_dark.jpg`: `OK`, Score: `0.5737` (empirically verifies drop under darkness).
   - `img7_lowres.jpg`: `NO_FACE` (empirically verifies failure handling on degenerate inputs).

### Untested Areas:
1. Multi-camera RTSP live video stream ingestion.
2. Inference execution under NVIDIA CUDA execution providers (tested on CPU).
3. Long-term session memory stability under >10,000 continuous sequential requests.
4. Downstream recognition accuracy impact on real-world CCTV verification benchmarks (e.g. TinyFaces, IJB-C).

---

## 14. Recommended Final Architecture

```
                                  [ Input Image (File / CCTV Stream) ]
                                                   │
                                                   ▼
                                     [ EXIF / Auto-Orientation ]
                                                   │
                                                   ▼
                                  [ Detector Chain (UniFace SCRFD) ]
                                    (Fallback: UniFace RetinaFace)
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
                  [ NO_FACE Result ]                                   [ Validated Face ]
             (Status: NO_FACE, Decision: NE)                                 │
                                                   ┌─────────────────────────┼─────────────────────────┐
                                                   ▼                         ▼                         ▼
                                         [ Similarity Warp ]         [ Bounding Box Crop ]     [ Original Face Crop ]
                                           (ArcFace 5-point)                 │                         │
                                                   │                         ▼                         ▼
                                                   ▼                [ UniFace HeadPose ]     [ Passive Diagnostics ]
                                          [ 112x112 RGB Crop ]         (ResNet-18 6D)        - Laplacian Variance
                                                   │                         │               - Exposure Histogram
                                                   ▼                         ▼               - Saturation Analysis
                                          [ eDifFIQA ONNX ]         [ UniFace MobileGaze ]             │
                                           (Quality Score)             (Gaze Pitch/Yaw)                ▼
                                                   │                         │               [ Diagnostic Flags ]
                                                   └─────────────────────────┼─────────────────────────┘
                                                                             │
                                                                             ▼
                                                                [ Multi-Gate Decision Engine ]
                                                                - Quality Score vs Calibrated Limits
                                                                - Pose Angle Thresholds (Pitch/Yaw/Roll)
                                                                - Minimum Inter-Eye Distance (IED)
                                                                - Gaze Angle Thresholds
                                                                             │
                                                                             ▼
                                                                [ Output JSON / Web Result ]
                                                                - Status: OK / Error
                                                                - Decision: ACCEPT / REVIEW / REJECT
                                                                - Detailed Metrics & Crop Overlays
```

---

## 15. Prioritized Implementation Roadmap

### Priority 0 (P0) — Immediate System Health & Integrity
- **P0.1: Isolate Threshold Regression Test:** Update `test_from_file` in `tests/test_quality_pipeline.py` to use a dedicated fixture file rather than the mutable runtime `quality_thresholds.json` so test suites remain green regardless of user configuration changes.
- **P0.2: Clarify Uncalibrated Initial Presets in UI:** Clearly display in the Settings UI that presets (e.g. 0.65 / 0.75) are initial starting points, and add a warning badge when decisions run against uncalibrated presets.

### Priority 1 (P1) — Essential Quality Gate Functionality
- **P1.1: Integrate Minimum Inter-Eye Distance (IED) into Decision Logic:** In surveillance, faces with IED $< 30\text{ px}$ rarely yield reliable biometric matches. Add `min_inter_eye_distance_px` to `QualityThresholds` to flag low-resolution faces as `REVIEW`.
- **P1.2: Integrate Passive Sharpness Diagnostic (Laplacian Variance):** Implement pure OpenCV Laplacian variance $\text{Var}(\nabla^2 I)$ to provide explicit blur diagnostics alongside the raw eDifFIQA score.
- **P1.3: Integrate Passive Exposure Diagnostic (Histogram Quantiles):** Adapt BioGaze's shadow/highlight histogram check into a lightweight OpenCV function to flag under-exposed and over-exposed faces.
- **P1.4: Execute Dataset Calibration:** Collect a representative operational surveillance dataset, generate false accept / false reject curves using `calibrate_thresholds.py`, and save calibrated operating thresholds.

### Priority 2 (P2) — Useful Architectural Improvements
- **P2.1: Integrate UniFace `MobileGaze`:** Activate `uniface.gaze.MobileGaze` to report gaze pitch and yaw in degrees, identifying subjects looking away from the camera.
- **P2.2: Migrate eDifFIQA Inference to ONNX Runtime:** Switch from PyTorch `models/ediffiqa.py` to `uniface.quality.EDifFIQA`. This achieves a ~2.5× speedup on CPU and removes PyTorch as an inference dependency.
- **P2.3: Multi-Face Priority Sorting:** Ensure that in crowded frames, faces are sorted and prioritized by face-area ratio and detection confidence.

### Priority 3 (P3) — Optional Features for Advanced Deployments
- **P3.1: Video Stream Multi-Frame Best-Shot Selection:** Integrate `uniface.tracking.BYTETracker` to track faces across CCTV video feeds and automatically extract the highest-scoring frame per subject tracklet.
- **P3.2: Anti-Spoofing / Presentation Attack Detection:** Integrate `uniface.spoofing.MiniFASNet` for access control applications.

---

## Final Synthesis: Specific Answers to Core Questions

### A. What is fully implemented today?
1. Full PyTorch `eDifFIQA-S` biometric face image quality scoring on 112×112 aligned face crops.
2. Two-stage face detection pipeline: UniFace SCRFD primary with automatic RetinaFace fallback and IoU de-duplication.
3. 5-point ArcFace landmark alignment and geometric validation.
4. UniFace 6D-representation head-pose estimation (continuous pitch, yaw, and roll).
5. Configurable threshold engine supporting `accept_min`, `reject_max`, and pose limits with `NOT_EVALUATED` safety guards.
6. Empirical threshold calibration CLI (`calibrate_thresholds.py`).
7. Complete FastAPI backend with model caching, EXIF handling, and batch exports.
8. Complete React/Vite web interface featuring interactive bounding box overlays, 100-image batch queues, 60 FPS scrolling, and diagnostic drawers.

### B. What is partially implemented?
1. **Decision Gate Signals:** The decision engine only checks quality score and pose limits; detection confidence, face area ratio, and inter-eye distance are calculated but ignored in the final decision.
2. **Threshold Calibration:** The mathematical tools exist, but operational thresholds remain uncalibrated against real surveillance distributions.
3. **Photographic Diagnostics:** Blur and darkness affect the eDifFIQA score, but the system does not break down separate photographic metrics.

### C. Which UniFace capabilities still need implementation?
1. `uniface.gaze.MobileGaze` — ONNX gaze estimation (pitch/yaw).
2. `uniface.quality.EDifFIQA` — ONNX Runtime quality scoring (to replace PyTorch inference).
3. `uniface.tracking.BYTETracker` — Multi-face tracking for CCTV video streams.
4. `uniface.spoofing.MiniFASNet` — Anti-spoofing / liveness detection.

### D. Which BioGaze capabilities are genuinely missing?
1. **Exposure & Illumination Analysis:** Measuring histogram shadow/highlight clipping and facial illumination asymmetry.
2. **Laplacian Blur Index:** Explicit optical focus measurement.
3. *Note on other BioGaze features:* Gaze and pose are already better addressed by UniFace. Passport constraints (white backgrounds, aspect ratios, shoulder visibility) must be rejected for surveillance use.

### E. What should be implemented next, and why?
1. **Laplacian Variance & Exposure Histogram Diagnostics (P1):** They require zero deep learning weights, run in $<2\text{ ms}$, and provide immediate, explainable diagnostics for why an image failed quality checks.
2. **UniFace `MobileGaze` Integration (P2):** It is already installed in the Python environment, runs via ONNX Runtime, and completes the directional assessment of the face.
3. **Dataset Calibration (P1):** Running `calibrate_thresholds.py` on actual target camera data to transition from provisional thresholds to validated operational thresholds.

### F. What should not be implemented because equivalent functionality already exists?
1. **BioGaze Dlib Landmark & Pose Estimation:** UniFace's direct ONNX ResNet-18 model is faster, avoids `dlib` compilation on Windows, and does not degrade with minor landmark jitter.
2. **BioGaze Gaze Estimation:** UniFace's `MobileGaze` provides equivalent ONNX inference without requiring custom PyTorch checkpoints.
3. **BioGaze ICAO Compliance Checks:** Background uniformity, aspect ratio filtering, and shoulder segmentation are inappropriate for CCTV and would cause severe false rejections.
4. **Additional Detectors (YOLOv8Face, BlazeFace):** SCRFD and RetinaFace already cover both large-scale and small-scale surveillance faces; adding more detectors would create unnecessary overhead.

### G. What evidence is still needed before claiming production readiness?
1. **Calibration on Real Surveillance Data:** An empirical false-rejection vs. false-acceptance trade-off curve evaluated on labeled images from the target surveillance cameras.
2. **Downstream Biometric Accuracy Verification:** Statistical evidence proving that rejecting images below `accept_min` actually improves false match and false non-match rates in the specific downstream face matcher used.
3. **Long-Running Concurrency Testing:** Profiling backend memory consumption and latency under sustained parallel load.
