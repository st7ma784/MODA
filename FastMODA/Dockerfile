# FastMODA Production Dockerfile - CPU optimized for Kubernetes/Rancher
# Build with: docker build -t st7ma784/moda-fastmoda:latest-cpu -f FastMODA/Dockerfile .

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY FastMODA/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY FastMODA/fastmoda/ ./fastmoda/
COPY FastMODA/templates/ ./templates/
COPY FastMODA/app.py ./
COPY FastMODA/example_usage.py ./

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
