---
title: Face Image Quality Assessment Lab
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Face Image Quality Assessment

[![Downloads](https://img.shields.io/github/downloads/yakhyo/face-image-quality-assessment/total.svg?label=Downloads)](https://github.com/yakhyo/face-image-quality-assessment/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> [!TIP]
> The models and functionality in this repository are **integrated into [UniFace](https://github.com/yakhyo/uniface)** — an all-in-one face analysis toolkit.<br>
> [![PyPI Version](https://img.shields.io/pypi/v/uniface.svg)](https://pypi.org/project/uniface/) [![GitHub Stars](https://img.shields.io/github/stars/yakhyo/uniface)](https://github.com/yakhyo/uniface/stargazers)

PyTorch inference, ONNX export, and ONNX Runtime inference for [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA) — a face image quality estimator that predicts a single quality score from an aligned 112×112 face. Uses [UniFace](https://github.com/yakhyo/uniface) for face detection and alignment.

## Models

| Variant    | Backbone      | Params | Input   | PyTorch (.pth)                                                                                           | ONNX                                                                                                      |
| ---------- | ------------- | ------ | ------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| eDifFIQA-T | MobileFaceNet | 1.7M   | 112x112 | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_t.pth) | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_t.onnx) |
| eDifFIQA-S | IResNet-18    | 24.6M  | 112x112 | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_s.pth) | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_s.onnx) |
| eDifFIQA-M | IResNet-50    | 44.1M  | 112x112 | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_m.pth) | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_m.onnx) |
| eDifFIQA-L | IResNet-100   | 65.7M  | 112x112 | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_l.pth) | [Link](https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_l.onnx) |

Higher score = better quality. Authors evaluate against 10 FIQA competitors on 7 static-image datasets + 1 video dataset using EDC curves; LFW and XQLFW are the commonly cited references for high-quality vs. low-quality regimes. eDifFIQA-L ranks first on the NIST [FATE-Quality](https://pages.nist.gov/frvt/html/frvt_quality.html) Kiosk-to-Entry track. See the [paper](https://ieeexplore.ieee.org/document/10468647) for per-dataset numbers.

> `.pth` weights are ported from the upstream authors' [OneDrive folder](https://unilj-my.sharepoint.com/:f:/g/personal/zb4290_student_uni-lj_si/EpxkJRo7artCud18dC2NAXcBAegU0uDaSMPwzG9ufcXkLg?e=zhsnho); `.onnx` files are exported from those weights and mirrored to this repo's release.

## Installation

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# eDifFIQA weights (Git Bash / WSL: bash download.sh). In PowerShell:
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights/ediffiqa_s.pth -OutFile weights\ediffiqa_s.pth
python -m unittest discover -s tests -v
```

UniFace 4.x does not install an ONNX runtime for you; `requirements.txt` lists `onnxruntime` (CPU). UniFace downloads
its detector / head-pose weights on first use.

### Linux / macOS

```bash
pip install -r requirements.txt && bash download.sh
```

## Web Dashboard: Face Quality Lab

A polished, responsive full-stack computer vision dashboard for single-image and batch face image quality assessment with eDifFIQA-S and UniFace.

### Features
- **Single-Image Analysis**: Drag-and-drop upload, interactive face bounding box overlay, aligned 112×112 crop preview, raw eDifFIQA model score visualization, and head-pose angle estimation.
- **Batch Evaluation**: Multi-image processing with live progress indication, filterable and sortable results table, partial failure recovery, and CSV / JSON report downloads.
- **Live System Metrics**: Real-time inference device indicators (CPU/CUDA) and session statistics (images processed, faces detected, detection failures, decisions).
- **Honest Status Mapping**: Uncalibrated thresholds strictly display `NOT_EVALUATED` and *"Score available — decision threshold not calibrated"* without inventing arbitrary thresholds.

### Unified Single-Command Showcase (Production Mode)

FastAPI automatically serves both the compiled React dashboard and REST APIs on a single port:

```powershell
# 1. Compile frontend (if not already built)
cd frontend && npm install && npm run build && cd ..

# 2. Launch unified application on Port 8000
.\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000
```
- **Web UI & Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Team LAN / Wi-Fi Access**: `http://<your-machine-ip>:8000`
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

> **Quick Windows Demo**: You can also double-click [`run_demo.bat`](file:///c:/Users/anjal/Desktop/Face_Image_Quality_Assessment/face-image-quality-assessment/run_demo.bat) in the root directory to launch everything with one click!

### Docker Deployment
```bash
docker compose up -d --build
```
Access at [http://localhost:8000](http://localhost:8000). See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for full cloud and team deployment steps.

### Development Mode (Hot Reload)

Open two separate PowerShell terminal windows:

#### Terminal 1 — FastAPI Backend:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2 — React/Vite Frontend:
```powershell
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### Running Automated Tests

#### Production Batch & Decision Engine Verification Suite:
```powershell
.\.venv\Scripts\pytest.exe tests/test_production_batch.py -v
```

#### Backend API Regression Suite:
```powershell
.\.venv\Scripts\pytest.exe api/test_api.py -v
```

#### Frontend Component Test Suite:
```powershell
cd frontend
npm test
```

#### Existing Pipeline Test Suite:
```powershell
python -m unittest discover -s tests -v
```


> The README demo table below is **upstream's** eDifFIQA-T output, not a result of the pipeline in this fork.

## Demo (upstream)

eDifFIQA-T quality scores on `assets/test_images/`. The same face, progressively degraded, scored lower in these examples:

|                  Original                   |                  Down-up sampled                   |                    Heavy blur                    |
| :-----------------------------------------: | :------------------------------------------------: | :----------------------------------------------: |
| ![](assets/results/image2.jpg)<br>**0.762** | ![](assets/results/image2_lowres.jpg)<br>**0.435** | ![](assets/results/image2_blur.jpg)<br>**0.144** |
| ![](assets/results/image3.jpg)<br>**0.737** | ![](assets/results/image3_lowres.jpg)<br>**0.537** | ![](assets/results/image3_blur.jpg)<br>**0.273** |

A lower score for a degraded variant is **not guaranteed** (see "Known limitations").

## Pipeline

```
detect (SCRFD, optional RetinaFace fallback) -> validate bbox/landmarks -> face_alignment (unchanged, 112x112)
  -> eDifFIQA score -> [optional head pose on the bbox crop] -> status + decision + reason
```

* **Status** (did processing work?): `OK`, `READ_ERROR`, `NO_FACE`, `ALIGNMENT_ERROR`, `MODEL_ERROR`.
* **Decision** (is the face good enough?): `ACCEPT`, `REJECT`, `REVIEW`, `NOT_EVALUATED`. Anything that is not
  status `OK` is `NOT_EVALUATED`. A detector miss is **never** reported as a quality verdict.
* The score is the raw eDifFIQA output. It is not a probability and is not calibrated by default.
* Thresholds ship as `null` (`quality_thresholds.json`), so decisions are `NOT_EVALUATED` until you calibrate on
  labeled data (`calibrate_thresholds.py`).

## Inference

```powershell
python main.py assets\test_images\image2.jpg --variant s
```

With the default SCRFD settings (640x640 input, confidence 0.5) the tight portraits in `assets/test_images` are
reported as `NO_FACE`. Options, in the order we recommend trying them:

```powershell
# 1) RetinaFace only when SCRFD yields no usable face
python main.py assets\test_images\image2.jpg --variant s --fallback-detector retinaface

# 2) Smaller SCRFD input for close-up / pre-cropped faces (no confidence change)
python main.py assets\test_images\image2.jpg --variant s --scrfd-input-size 320

# 3) Head pose (pitch/yaw/roll) as an extra report field
python main.py assets\test_images\image2.jpg --variant s --fallback-detector retinaface --head-pose
```

Why `NO_FACE` happens, measured on the 5 sample portraits (`detector_diagnostic.py`): UniFace letterboxes every image
to the detector input and *upscales* small images. The four portraits SCRFD misses (208-537 px images) have faces of
~460-515 px at SCRFD's 640 input (using RetinaFace's boxes), and SCRFD's best candidate scores only 0.20-0.34 (for
`image3.jpg` that candidate is a 16 px false positive, not the face, so lowering the threshold would return a wrong
box). The one portrait SCRFD finds (`image5.jpg`, 0.857) has a ~270 px face at input. With `--scrfd-input-size 320` the
same four faces are ~240-260 px at input and are found at 0.86 with the confidence threshold unchanged. This is
consistent with the faces being larger than SCRFD's comfortable scale range, but five images do not prove the
mechanism. `input_size=320` is **not** a safe default for wide surveillance frames with small faces; choose per source.

Detector settings live in `detector_config.json` (CLI flags override it). Defaults equal UniFace's defaults.
Bounding boxes and landmarks are validated (finite, positive-size box that overlaps the image, 5x2 non-degenerate
landmarks whose centroid lies in the box, confidence in [0, 1]); overlapping detections are de-duplicated.

### Diagnose a detector miss

```powershell
python detector_diagnostic.py assets\test_images\image2.jpg --scrfd-input-sizes 320 --json results\detector_diagnostic.json
```

Reports production detections, sub-threshold candidates (low floor used for diagnosis only), face-to-image ratio, the
face size at the detector input, alignment status, and an optional input-size sweep.

### Batch evaluation

```powershell
python batch_evaluate.py assets\test_images --fallback-detector retinaface --head-pose `
    --degrade blur:3 --degrade lowres:4 --degrade dark:0.3 --yaw-bins 15 30
```

Writes `results\batch_evaluation.csv/.json` (one row per face, or per image if nothing was scored) and
`results\batch_summary.json`: status/decision counts, detection success, fallback frequency, timing, score
distribution per category, original-vs-variant pairs (with exceptions listed), and score by |yaw| bin.
`--degrade` creates deterministic in-memory variants (`blur:SIGMA`, `lowres:FACTOR` down/up-sample, `dark:SCALE`
brightness scaling). Files named `*_blur` / `*_lowres` are categorised automatically. Low-resolution, dark and
side-facing categories beyond that need your own images, supplied via `--labels labels.csv` (`image[,category][,label]`).

### Threshold calibration (pending until you have labels)

```powershell
python batch_evaluate.py my_dataset --labels labels.csv --fallback-detector retinaface
python calibrate_thresholds.py results\batch_evaluation.csv --target-precision 0.95 --max-false-rejection 0.02
```

Labels are `usable` / `unusable` for recognition. The tool writes a threshold sweep (precision, recall, F1, false
rejection / acceptance rates) and a *provisional* proposal only when each class has at least `--min-per-class`
samples (default 50) and you supply the two operating targets. It never edits `quality_thresholds.json`. No labeled
dataset ships with this repository, so **no threshold has been calibrated**.

### Head pose

`--head-pose` uses `uniface.headpose.HeadPose` (ResNet-18) on the detector's **bbox crop** of the original image. It
estimates head orientation, not gaze. Optional limits (`max_abs_pitch/yaw/roll` in `quality_thresholds.json`) turn
violations into `REVIEW` (never an automatic `REJECT`) and are `null` by default; calibrate them against your
cameras before use. Setting limits without `--head-pose` is an error.

### Aligned crops

```powershell
python main.py --aligned aligned_face.png --variant s
```

The input must be exactly 112x112, 3-channel, uint8 and non-constant; anything else is `ALIGNMENT_ERROR`. The check
cannot verify that the crop was aligned with the 5-point ArcFace template; that is your responsibility.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests needing `weights\ediffiqa_s.pth` and `assets\test_images` are skipped when those are missing.

## Known limitations

* Scores depend on the detector's landmarks (the same face scored 0.7615 with SCRFD@320 and 0.7756 with
  RetinaFace on `image3.jpg`). Calibrate with the same detector configuration you deploy.
* A degraded variant does not always score lower (sample data: `image0.jpg` down/up-sampled x4 scored higher than the
  original). Check the `paired_summary.exceptions` in the batch summary.
* Head-pose estimates on blurred crops are unreliable (`image2_blur.jpg` -> yaw 50 deg vs 1.9 deg for the sharp image).
* Nothing here measures face-recognition accuracy; any benefit to recognition must be evaluated separately.

## ONNX Inference

```bash
python onnx_inference.py --model weights/ediffiqa_s.onnx assets/test_images/image2.jpg
```

```python
import cv2
from models import eDifFIQAOnnx
from uniface.detection import SCRFD

detector = SCRFD()
quality = eDifFIQAOnnx("weights/ediffiqa_s.onnx")

image = cv2.imread("image.jpg")
for face in detector.detect(image):
    score = quality.get_quality(image, face.landmarks)
    print(f"Quality: {score:.4f}")
```

## ONNX Export

```bash
python onnx_export.py -v s -w weights/ediffiqa_s.pth -o weights/ediffiqa_s.onnx --dynamic
```

## Reference

- [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA) — Original PyTorch implementation and paper
- [UniFace](https://github.com/yakhyo/uniface) — Face detection and alignment

## License

[MIT License](./LICENSE)
