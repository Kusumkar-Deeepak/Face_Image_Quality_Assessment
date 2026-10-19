# Face Image Quality Assessment System — Complete Deployment & Showcase Guide

This guide provides step-by-step instructions to deploy, run, and showcase the Face Image Quality Assessment (FIQA) & Surveillance application to your team, stakeholders, or clients.

---

## Architecture Overview

The system runs as a single unified service:
- **Backend**: FastAPI serving eDifFIQA-S quality assessment, UniFace SCRFD & RetinaFace detectors, head pose estimation, UniFace MobileGaze, and explainable photometric clarity diagnostics.
- **Frontend**: High-performance React + Tailwind CSS dashboard with responsive charts, multi-face visualizer, 100-image batch queue, review policy selector, lightbox zoom modal, and passed faces ZIP download.
- **Unified Delivery**: The compiled React production bundle is mounted directly by FastAPI. **One port (`8000`) serves both the entire interactive Web UI and the REST API.**

```
+-------------------------------------------------------------+
|                     User / Team Browser                     |
+-------------------------------------------------------------+
                               |
                        Port 8000 (HTTP)
                               |
                               v
+-------------------------------------------------------------+
|                     FastAPI Application                     |
|  +-----------------------+   +---------------------------+  |
|  |   Frontend Mount (/)  |   |    REST API (/api/...)    |  |
|  |   (Compiled React)    |   |  • /analyze               |  |
|  |                       |   |  • /analyze/batch (<=100) |  |
|  |                       |   |  • /results/{id}/passed-* |  |
|  +-----------------------+   +---------------------------+  |
+-------------------------------------------------------------+
                               |
+-------------------------------------------------------------+
|                   Computer Vision Engine                    |
|  • UniFace SCRFD (640/960/1280) + RetinaFace Fallback       |
|  • eDifFIQA-S Torch Inference Mode                          |
|  • UniFace Head Pose & MobileGaze Eye Tracking              |
|  • Photometric Diagnostics (Laplacian Sharpness, Exposure)  |
+-------------------------------------------------------------+
```

---

## Method 1: Instant Local / Team LAN Demo (Recommended for Quick Showcase)

Use this method when presenting from your machine, screen-sharing in a meeting, or letting colleagues on the same office or home Wi-Fi test the app directly from their phones or laptops.

### Quick Start (Windows One-Click)
Double-click the provided batch script in the root directory:
```powershell
run_demo.bat
```

### Manual Command Line Steps
1. **Ensure the React frontend is compiled**:
   ```powershell
   cd frontend
   npm run build
   cd ..
   ```
2. **Start the Unified Server**:
   ```powershell
   .\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Open the URLs**:
   - **Local Machine**: [http://localhost:8000](http://localhost:8000)
   - **Colleagues on LAN / Wi-Fi**: `http://<your-ip-address>:8000`  
     *(Find your IP by opening a terminal and typing `ipconfig`)*
   - **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **System Health Status**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## Method 2: Instant Public Demo URL in 30 Seconds (No Port Forwarding)

Use this when presenting remotely over Zoom/Teams and you want team members to click a link and test the app live on their own devices.

### Option A: Cloudflare Tunnel (Free, Fast, No Account Required)
1. Download `cloudflared` (or run if installed):
   ```powershell
   winget install Cloudflare.cloudflared
   ```
2. In a separate terminal while `uvicorn` is running on port 8000:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```
3. Cloudflare will output a temporary public HTTPS link (e.g. `https://random-subdomain.trycloudflare.com`). Share this link with your team!

### Option B: ngrok
1. Start ngrok:
   ```powershell
   ngrok http 8000
   ```
2. Copy the public forwarding URL (e.g. `https://xyz.ngrok-free.app`) and send it to your team.

---

## Method 3: Containerized Deployment (Docker & Docker Compose)

This is the standard enterprise method for deploying to any server with Docker installed.

### Using Docker Compose (Easiest)
From the repository root:
```bash
# Build and launch the container
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop the container
docker compose down
```

### Using Standard Docker CLI
```bash
# 1. Build the production image
docker build -t face-quality-assessment:latest .

# 2. Run the container
docker run -d \
  --name face_quality_app \
  -p 8000:8000 \
  --restart unless-stopped \
  face-quality-assessment:latest

# 3. Check container status
docker ps
```
The app will be available at `http://<server-ip>:8000`.

---

## Method 4: Production Linux Cloud VM (AWS EC2 / DigitalOcean / GCP / Azure)

Follow these steps to deploy on an Ubuntu 22.04 / 24.04 LTS server.

### 1. Provision System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl libgl1 libglib2.0-0 libgomp1 nginx
```

### 2. Clone Repository and Setup Virtualenv
```bash
git clone <your-repo-url> /opt/face-quality-assessment
cd /opt/face-quality-assessment

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build Frontend
```bash
# Install Node.js 20 LTS if not present
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

cd /opt/face-quality-assessment/frontend
npm ci
npm run build
cd /opt/face-quality-assessment
```

### 4. Create Systemd Service (Auto-restart & 24/7 Uptime)
Create service file `/etc/systemd/system/fiqa.service`:
```ini
[Unit]
Description=Face Image Quality Assessment API & Web UI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/face-quality-assessment
ExecStart=/opt/face-quality-assessment/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fiqa
sudo systemctl start fiqa
sudo systemctl status fiqa
```

### 5. Configure Nginx Reverse Proxy with SSL
Create `/etc/nginx/sites-available/fiqa`:
```nginx
server {
    listen 80;
    server_name fiqa.yourcompany.com;  # or server IP

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable and reload Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/fiqa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

*(Optional) Add free SSL with Let's Encrypt:*
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d fiqa.yourcompany.com
```

---

## Method 5: Demo Script & Talking Points for the Team

When presenting your work, use this recommended walkthrough structure:

### 1. The Problem & Business Context
- Explain that standard face recognition in security, surveillance, and automated identity verification fails when low-quality, blurry, off-angle, or poorly lit faces are passed to downstream recognition models.
- Traditional thresholding is either too brittle or lacks explainability.

### 2. Live Demo — Single Face Analysis
1. Open the UI at `http://localhost:8000`.
2. Upload a sample portrait from `assets/test_images/image2.jpg`.
3. Highlight:
   - **Detector**: SCRFD primary with RetinaFace automatic fallback.
   - **112×112 Aligned Face Crop**: Displayed with inter-eye distance and bounding box coordinates.
   - **Explainable Photometric Diagnostics**: Laplacian sharpness variance, exposure extremes (underexposed/overexposed ratios), and left/right facial illumination asymmetry.
   - **UniFace MobileGaze**: Eye line-of-sight tracking distinguished from head-pose angles.
   - **Click-to-Zoom Lightbox**: Click the face crop or "Zoom Viewer" button to demonstrate pan, zoom, 1:1 inspection, and overlay toggling.

### 3. Live Demo — Crowded Scene & Missed Faces Fix
1. Switch detector preset to **"Crowded Scene (High Recall)"**.
2. Explain how this activates the complementary ensemble (SCRFD at 960px + RetinaFace) to recover small, occluded, or background faces that single detectors miss.

### 4. Live Demo — 100-Image High-Capacity Batch Evaluation
1. Switch to the **Batch Evaluation** tab.
2. Select multiple images from `assets/test_images/`.
3. Highlight:
   - **Ingestion Safety**: Strictly enforces $\le 100$ images, protects against decompression bombs and partial file corruptions.
   - **Multi-Stage Progress Bar**: Live transitions from Uploading $\to$ Validating $\to$ Detecting Faces $\to$ Assessing Quality.
   - **Summary Metric Cards**: 6 discrete gate states (`PASS`, `REVIEW`, `REJECT`, `NO_FACE`, `NOT_EVALUATED`, `ERROR`).
   - **Configurable Review Policy**: Switch between `Manual Review`, `Treat as PASS`, and `Treat as REJECT`, demonstrating how the effective decision and live counts update in real-time.
   - **Download Passed Faces (.zip)**: Click the export button to download the archive containing only compliant 112×112 face crops, complete with `manifest.csv` and `manifest.json`.
