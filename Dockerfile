# Multi-stage Universal Dockerfile for Face Image Quality Assessment System
# Compatible with Hugging Face Spaces (Port 7860, non-root user), Render ($PORT), and Local Docker

# ==========================================
# Stage 1: Build React/Vite Production Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend Runtime
# ==========================================
FROM python:3.10-slim AS runtime

# Install system libraries for OpenCV and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces & security
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip & install CPU-only PyTorch first (drastically reduces image size from ~3GB to ~600MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=user:user . .

# Copy compiled frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder --chown=user:user /app/frontend/dist ./frontend/dist

# Create weights & temp directories and set permissions
RUN mkdir -p weights && chown -R user:user /app

USER user

# Default port 7860 (Hugging Face Spaces standard); Render overrides via $PORT
ENV PORT=7860
EXPOSE 7860 8000 10000

# Start unified FastAPI server dynamically binding to $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
