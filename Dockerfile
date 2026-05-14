# MODA Server Dockerfile — Multi-stage
# Serves the MODA algorithm suite via a Python/scipy backend.
# The MATLAB GUIs in allguis/ require a licensed MATLAB install and are
# not included here; this image exposes the same algorithms via FastMODA.
#
# Build (CI):  docker build --target moda-prod -t moda-server .
# Build (dev): docker build --target moda-dev  -t moda-server:dev .

ARG PYTHON_VERSION=3.11
ARG MATLAB_VERSION=r2024b

# ── base: shared dependencies ────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY FastMODA/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── moda-dev: development image with test tooling ────────────────────────────
FROM base AS moda-dev

RUN pip install --no-cache-dir pytest pytest-cov

COPY FastMODA/ ./FastMODA/
COPY allguis/  ./allguis/

WORKDIR /app/FastMODA
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pytest", "test.py", "-v"]

# ── moda-prod: lean production image ─────────────────────────────────────────
FROM base AS moda-prod

COPY FastMODA/fastmoda/ ./fastmoda/
COPY FastMODA/templates/ ./templates/
COPY FastMODA/app.py     ./app.py

RUN mkdir -p /app/uploads

EXPOSE 5000

ENV FLASK_APP=app.py \
    PYTHONUNBUFFERED=1 \
    MATLAB_VERSION=${MATLAB_VERSION}

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", \
     "--timeout", "300", "--keep-alive", "5", "app:app"]
