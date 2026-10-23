---
title: Face Image Quality Assessment Lab
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Face Image Quality Assessment (FIQA) & Biometric Assurance Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CPU-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.15%2B-blue.svg)](https://onnxruntime.ai/)
[![UniFace](https://img.shields.io/badge/UniFace-4.0.0-purple.svg)](https://github.com/yakhyo/uniface)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-ready, end-to-end **Face Image Quality Assessment (FIQA)** system designed for biometric enrollment validation, access control, and crowded surveillance monitoring. 

The platform integrates **eDifFIQA-S** deep quality utility estimation, a two-stage hybrid face detection pipeline (**UniFace SCRFD + RetinaFace Fallback**), 5-point facial landmark alignment, 3D head-pose estimation (**6DRepNet**), gaze vector tracking (**MobileGaze**), explainable photographic diagnostics, a deterministic 6-state decision gate engine, and a modern **React/Vite dashboard** powered by a high-performance **FastAPI** backend.

---

## 🌐 Live Demonstrations & Deliverables

| Deliverable | Access Link / Location | Description |
| :--- | :--- | :--- |
| **Hugging Face Live Demo** | [deeepak08-face-image-quality.hf.space](https://deeepak08-face-image-quality.hf.space) | High-RAM (16 GB) live cloud demo running all models concurrently. |
| **Render Cloud Deployment** | [face-quality-recognization.onrender.com](https://face-quality-recognization.onrender.com) | Unified React + FastAPI production deployment on Render. |
| **Interactive API Documentation** | [localhost:8000/docs](http://localhost:8000/docs) (or on live URLs) | Interactive Swagger / OpenAPI documentation for all REST endpoints. |
| **Engineering Status Report** | [`docs/FIQA_Project_Status_Report.pdf`](docs/FIQA_Project_Status_Report.pdf) | Formal project report covering architecture, benchmarks, and challenges. |
| **Quick Windows Demo** | Double-click [`run_demo.bat`](run_demo.bat) | 1-click script to build the frontend and launch the server. |

---

## 🎯 Key Capabilities & Architecture

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │                   INPUT FACE IMAGE                     │
                                      └──────────────────────────┬─────────────────────────────┘
                                                                 │
                                                [Safe Pre-Scaling to 1920px Lanczos]
                                                                 │
                                                                 ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │             HYBRID TWO-STAGE DETECTION CHAIN           │
                                      │  Primary: UniFace SCRFD  ──► Fallback: RetinaFace     │
                                      │  Deduplication: IoU Threshold <= 0.5                   │
                                      └──────────────────────────┬─────────────────────────────┘
                                                                 │
                                                 [5-Point Facial Landmarks (Affine)]
                                                                 │
                                                                 ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │           NORMALIZED 112x112 FACE CROP & SCORING       │
                                      │  • eDifFIQA-S Biometric Utility Score (0.0 to 1.0)     │
                                      │  • 6DRepNet 3D Head Pose (Pitch, Yaw, Roll)            │
                                      │  • MobileGaze Vector (Gaze Pitch, Gaze Yaw)            │
                                      │  • Laplacian Variance (Sharpness & Blur)               │
                                      │  • Exposure Ratios (Underexposure / Overexposure %)   │
                                      │  • Illumination Asymmetry (Left vs Right Face Split)   │
                                      └──────────────────────────┬─────────────────────────────┘
                                                                 │
                                                                 ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │             6-STATE DETERMINISTIC GATE ENGINE          │
                                      │   [ PASS | REVIEW | REJECT | NO_FACE | NOT_EVAL | ERR ] │
                                      └──────────────────────────┬─────────────────────────────┘
                                                                 │
                                                                 ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │               REACT DASHBOARD & EXPORT                 │
                                      │   Interactive BBoxes • Live Batch Progress • ZIP / CSV │
                                      └────────────────────────────────────────────────────────┘
```

### 1. eDifFIQA-S Biometric Quality Scoring
* Standardized quality estimation predicting face utility for 1:1 verification and 1:N biometric identification.
* Generates continuous normalized scores ($[0.0, 1.0]$) where higher scores indicate superior biometric matchability.

### 2. High-Recall Crowd Detection Architecture
* Solves the common problem of missed faces in dense surveillance and tourist photographs (e.g., Gateway of India benchmark).
* **UniFace SCRFD** operates as the ultra-fast primary detector.
* **RetinaFace** operates as a high-recall fallback or complementary detector.
* Bounding boxes are filtered through an IoU deduplication engine ($\text{IoU} \le 0.5$), ensuring nearby individuals are kept while duplicate candidate boxes are removed.

### 3. 3D Head Pose & Gaze Vector Tracking
* **Head Pose (6DRepNet):** Estimates 3D Euler angles (Pitch, Yaw, Roll) to enforce frontal-face compliance for biometric enrollment.
* **Eye Gaze (MobileGaze):** Tracks line-of-sight pitch and yaw vectors to detect if the subject is looking directly into the camera.

### 4. Explainable Photographic Diagnostics
* **Sharpness / Blur:** OpenCV Laplacian Variance on the cropped face region.
* **Exposure Histograms:** Quantifies underexposure ($<15\%$ luminance) and overexposure ($>85\%$ luminance) percentages.
* **Illumination Asymmetry:** Evaluates directional harsh shadowing by comparing left-versus-right split-face mean luminance.

### 5. Production Decision Gate Engine
* Discretizes evaluation into 6 deterministic states:
  * `PASS`: Satisfies all validated operational criteria.
  * `REVIEW`: Borderline score, non-frontal pose angle, or extreme illumination asymmetry.
  * `REJECT`: Explicit operational rejection (e.g., severe blur, severe pose violation).
  * `NO_FACE`: Zero face candidates detected.
  * `NOT_EVALUATED`: Thresholds intentionally uncalibrated.
  * `ERROR`: File decode error, corrupt data, or alignment failure.
* **Dynamic Configuration:** Live threshold updates supported via REST API without server downtime.

### 6. Interactive React 18 / Vite Dashboard
* **Single Image Studio:** Drag-and-drop analysis, responsive bounding box rendering, landmark visualizer, 112×112 aligned crop previews, and full diagnostic breakdowns.
* **High-Throughput Batch Evaluator:** Process up to 100 images with parallel micro-chunk streaming, live progress bars, filterable result tables, and **1-click downloads for CSV reports, JSON data, and ZIP archives containing only passed faces**.

---

## 📂 Project Directory Structure

```
face-image-quality-assessment/
├── api/                           # FastAPI backend implementation
│   ├── __init__.py
│   ├── main.py                    # Server routing, endpoints, CORS & static mounting
│   ├── schemas.py                 # Pydantic v2 validation models & response schemas
│   └── service.py                 # Pipeline service singleton, caching & memory control
├── assets/                        # Sample evaluation photographs & benchmarks
│   ├── test_images/               # Portraits, degraded variants, and crowd scenes
│   └── results/                   # Benchmark visual output examples
├── configs/                       # Configuration schemas
│   ├── detector_config.json       # SCRFD / RetinaFace input sizes and thresholds
│   └── thresholds.json            # Calibrated gate acceptance and rejection limits
├── docs/                          # Project documentation & engineering logs
│   ├── DEPLOYMENT_GUIDE.md        # Comprehensive multi-cloud deployment guide
│   ├── FIQA_Project_Status_Report.pdf  # Formal executive status report (PDF)
│   └── REPOSITORY_INTEGRATION_AUDIT.md # Technical audit & benchmark report
├── frontend/                      # Modern React / Vite web dashboard
│   ├── dist/                      # Pre-compiled production web assets
│   ├── src/
│   │   ├── components/            # UI components (SingleImage, BatchEvaluation, BoundingBoxes)
│   │   ├── api.js                 # Axios API client wrapper
│   │   ├── App.jsx                # Core layout and tab navigation
│   │   └── main.jsx               # React entrypoint
│   ├── package.json               # Node.js dependencies
│   └── vite.config.js             # Vite build configuration
├── models/                        # Deep learning model architectures
│   ├── backbones.py               # Modified IResNet and MobileFaceNet backbones
│   ├── ediffiqa.py                # eDifFIQA PyTorch head and feature map wrapper
│   └── ediffiqa_onnx.py           # ONNX Runtime inference implementation
├── scripts/                       # Engineering utilities and reporting tools
│   └── generate_report_pdf.py     # Script to compile executive status report to PDF
├── tests/                         # Production verification & automated test suites
│   ├── test_crowd_detection.py    # Multi-face crowd detection benchmark tests
│   ├── test_gaze.py               # MobileGaze unit and boundary tests
│   ├── test_photometric.py        # Sharpness, exposure and asymmetry metric tests
│   ├── test_production_batch.py   # 100-image batch limits, partial failure, ZIP exports
│   └── test_quality_pipeline.py   # eDifFIQA scoring, landmark alignment, and gate logic
├── weights/                       # Pretrained model weights
│   ├── .gitkeep
│   ├── ediffiqa_s.onnx            # eDifFIQA-S ONNX weights (auto-downloadable)
│   └── ediffiqa_s.pth             # eDifFIQA-S PyTorch weights (auto-downloadable)
├── app.py                         # Hugging Face Spaces Gradio interface
├── calibrate_thresholds.py        # Empirical threshold calibration tool
├── detector_config.json           # Active detector configuration
├── Dockerfile                     # Multi-stage production container definition
├── docker-compose.yml             # Docker compose service definition
├── face_detection.py              # SCRFD + RetinaFace detector chain with IoU deduplication
├── gaze_check.py                  # UniFace MobileGaze angle extraction
├── main.py                        # Standalone CLI entrypoint
├── photometric_quality.py         # Explicit photometric quality measurement functions
├── pose_check.py                  # UniFace 6DRepNet head-pose extraction
├── quality_pipeline.py            # Main pipeline coordinator, singleton weights & scoring
├── quality_thresholds.json        # Active gate thresholds
├── render.yaml                    # Render blueprint deployment specification
├── requirements.txt               # Python production dependencies
└── run_demo.bat                   # 1-click Windows demo launcher
```

---

## 🚀 Quickstart & Setup Guide

### Option 1: 1-Click Windows Demo (Easiest)
If you are on Windows, simply double-click [`run_demo.bat`](run_demo.bat) in the root directory. It automatically verifies your environment, compiles the frontend if needed, and starts the unified application on `http://localhost:8000`.

---

### Option 2: Unified Production Mode (FastAPI + React)
FastAPI natively serves both the backend API and the compiled React dashboard on a single port.

#### 1. Clone & Set Up Python Virtual Environment:
```powershell
# Clone the repository
git clone https://github.com/Kusumkar-Deeepak/Face_Image_Quality_Assessment.git
cd Face_Image_Quality_Assessment

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Ensure Model Weights are Present:
The system automatically downloads weights on first use, or you can download them manually:
```powershell
# PowerShell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_s.pth -OutFile weights\ediffiqa_s.pth

# Linux / macOS
mkdir -p weights && curl -L https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_s.pth -o weights/ediffiqa_s.pth
```

#### 3. Build Frontend & Launch Server:
```powershell
# Build the React frontend
cd frontend
npm install
npm run build
cd ..

# Launch unified server on Port 8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
* **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **System Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Option 3: Development Mode (Hot-Reload)
For active development with Vite and FastAPI hot-reloading:

* **Terminal 1 (FastAPI Backend)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
  ```

* **Terminal 2 (React Frontend)**:
  ```powershell
  cd frontend
  npm install
  npm run dev
  ```
  Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### Option 4: Docker & Docker Compose
To run in an isolated Docker container:
```bash
docker compose up -d --build
```
Access the dashboard at [http://localhost:8000](http://localhost:8000).

---

### Option 5: Command-Line (CLI) Inference
You can also run standalone CLI evaluations without starting the web server:

```powershell
# Evaluate a single face image with head-pose and gaze tracking
python main.py assets/test_images/image5.jpg --variant s --head-pose --gaze

# Evaluate with RetinaFace fallback enabled
python main.py assets/test_images/image2.jpg --variant s --fallback-detector retinaface

# Run batch evaluation with synthetic degradations across a directory
python batch_evaluate.py assets/test_images --fallback-detector retinaface --head-pose --degrade blur:3 --degrade lowres:4
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive `pytest` verification suite covering batch processing, partial failure handling, biometric thresholds, and crowd detection:

```powershell
# Run production batch & decision engine tests
pytest tests/test_production_batch.py -v

# Run crowd detection & fallback tests
pytest tests/test_crowd_detection.py -v

# Run photometric quality diagnostics tests
pytest tests/test_photometric.py -v

# Run full test suite
pytest tests/ -v
```

---

## 📊 Threshold Calibration Guide

Biometric decision thresholds should always be calibrated to the specific camera hardware, illumination environment, and operational risk tolerance:

1. Label a set of face images as `usable` or `unusable` in a CSV file (`image,label`).
2. Run empirical threshold sweeping:
   ```powershell
   python batch_evaluate.py my_dataset --labels labels.csv --fallback-detector retinaface
   python calibrate_thresholds.py results/batch_evaluation.csv --target-precision 0.95 --max-false-rejection 0.02
   ```
3. Update `configs/thresholds.json` directly or use the live UI Settings tab to apply the calibrated limits.

---

## ⚙️ Cloud Resource Optimization & Architecture Notes

* **Multi-Model Footprint:** Running 5 deep learning models (`SCRFD` + `RetinaFace` + `eDifFIQA-S` + `6DRepNet` + `MobileGaze`) simultaneously on CPU requires approximately **~700–800 MB of RAM** during peak batch inference.
* **512 MB Container Hardening (Render Free Tier):**
  * *Singleton Model Caching:* `QualityPipeline` shares model weights and ONNX runtime detector sessions across instances, eliminating duplicate 200 MB allocations.
  * *Kernel Memory Trimming:* Injected `malloc_trim(0)` via `libc` to return unmapped heap chunks back to the Linux OS after every image.
  * *Safe Pre-Scaling:* Images $>1920\text{px}$ are downscaled via Lanczos resampling prior to tensor conversion, slashing raw memory by $>80\%$.
  * *Micro-Chunk Streaming:* Batch queues stream files in micro-chunks of 2, preventing memory spikes.
* **High-Capacity Deployment (Hugging Face Spaces):**
  * Deployed on Hugging Face Spaces (16 GB Free RAM) where all 5 models and large batch queues execute with zero hardware constraints.

---

## 📄 License & Acknowledgements

* **License:** [MIT License](LICENSE)
* **eDifFIQA:** Babnik et al., *"eDifFIQA: Efficient Diffusion-based Face Image Quality Assessment"*, IEEE Access 2024. [Paper Link](https://ieeexplore.ieee.org/document/10468647)
* **UniFace:** Yakhyo et al., *"UniFace: Deep Face Analysis Toolkit"*. [GitHub Link](https://github.com/yakhyo/uniface)
