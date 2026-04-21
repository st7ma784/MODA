# FastMODA Production Dockerfile - Multi-stage (CPU/GPU)
# Build CPU: docker build -t st7ma784/moda-fastmoda:latest-cpu -f FastMODA/Dockerfile --target base .
# Build GPU: docker build -t st7ma784/moda-fastmoda:latest-gpu -f FastMODA/Dockerfile --target gpu --build-arg CUDA_VERSION=12.4 .

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.4

# ============================================================
# BASE STAGE - CPU optimized for Kubernetes/Rancher
# ============================================================
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastmoda/ ./fastmoda/
COPY templates/ ./templates/
COPY app.py ./
COPY example_usage.py ./

# Create directory for uploaded files
RUN mkdir -p /app/uploads

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=5)" || exit 1

# Run the application with gunicorn (production-grade WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

# ============================================================
# GPU STAGE - Extends base with CUDA support
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS gpu

ARG PYTHON_VERSION=3.11

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    gcc \
    g++ \
    libgomp1 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY fastmoda/ ./fastmoda/
COPY templates/ ./templates/
COPY app.py ./
COPY example_usage.py ./

# Create directory for uploaded files
RUN mkdir -p /app/uploads

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=false
ENV TORCH_HOME=/app/torch_cache
ENV USE_GPU=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=5)" || exit 1

# Run the application with gunicorn (production-grade WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
