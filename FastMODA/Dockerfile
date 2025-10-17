# Multi-stage build for FastMODA with optional GPU support
# Build with: docker build -t fastmoda:latest .
# For GPU: docker build -t fastmoda:gpu --build-arg CUDA_VERSION=11.8 .

ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.11

# Base stage - CPU only
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-gpu.txt ./

# Install Python dependencies (CPU version)
RUN pip install --no-cache-dir -r requirements.txt

# GPU stage - with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-runtime-ubuntu22.04 as gpu

ARG PYTHON_VERSION=3.10

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy requirements
COPY requirements.txt requirements-gpu.txt ./

# Install Python dependencies (GPU version)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-gpu.txt

# Final stage - choose base or gpu
FROM ${BUILD_TARGET:-base} as final

WORKDIR /app

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# Run the application
CMD ["python", "app.py"]
