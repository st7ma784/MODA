# Building the Containerized Emulator

## Overview

This guide explains how to build the Docker images for the new containerized emulator, customize them, and deploy to different environments.

---

## Quick Build

### Build All Emulator Images

```bash
cd /home/user/MODA

# Build all three images
docker-compose build fastmoda-api flutter-emulator moda-signal-mock-server

# Or with no-cache to rebuild from scratch
docker-compose build --no-cache fastmoda-api flutter-emulator moda-signal-mock-server
```

### Build Individual Images

```bash
# FastMODA (standard Python API)
docker-compose build fastmoda-api

# Flutter Web App
docker-compose build flutter-emulator

# Signal Mock Server
docker-compose build moda-signal-mock-server
```

---

## Customization

### Flutter Web App

#### Change FastMODA URL at Build Time

```bash
docker-compose build \
  --build-arg FASTMODA_URL=https://moda.example.com \
  flutter-emulator
```

Or directly with Docker:

```bash
docker build \
  --build-arg FASTMODA_URL=http://192.168.1.100:5000 \
  -f APP/Dockerfile.web \
  -t moda-flutter-web:custom \
  APP/
```

#### Build for Production

```bash
# Production build (already enabled in Dockerfile.web)
docker build \
  --build-arg FASTMODA_URL=https://api.moda-prod.com \
  -f APP/Dockerfile.web \
  --label version=1.0 \
  --label env=production \
  -t moda-flutter-web:1.0-prod \
  APP/
```

#### Use Different Nginx Configuration

If you need custom Nginx settings, create `APP/nginx.prod.conf`:

```nginx
# Production-grade configuration
server {
    listen 80;
    server_name moda-app.example.com;
    
    # ... custom settings ...
}
```

Then update `Dockerfile.web`:

```dockerfile
# Copy custom config
COPY nginx.prod.conf /etc/nginx/conf.d/default.conf
```

### Signal Mock Server

#### Customize Sample Rate

Edit `emulator_refactored.py`:

```python
class SignalGenerator:
    def __init__(self, fs: float = 512.0):  # Change from 256.0
        self.fs = fs
```

Rebuild:

```bash
docker-compose build --no-cache moda-signal-mock-server
```

#### Add Custom Presets

Edit `emulator_refactored.py`:

```python
PRESETS = {
    # ... existing presets ...
    "custom_seizure": dict(
        alpha=0.05, theta=0.05, beta=0.05, delta=0.05, gamma=2.0, noise=0.1
    ),
}
```

### FastMODA

Usually no rebuild needed — uses existing `FastMODA/Dockerfile`. To customize:

```bash
# Rebuild with custom environment
docker-compose build \
  --build-arg PYTHON_VERSION=3.12 \
  fastmoda-api
```

---

## Multi-Architecture Builds

Build for ARM64 (Apple Silicon, Raspberry Pi, etc.):

### Option 1: BuildKit (Recommended)

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build for ARM64
docker build \
  --platform linux/arm64 \
  -f APP/Dockerfile.web \
  -t moda-flutter-web:arm64 \
  APP/
```

### Option 2: Docker Buildx (Multi-Platform)

```bash
# Create builder
docker buildx create --name moda-builder
docker buildx use moda-builder

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f APP/Dockerfile.web \
  -t moda-flutter-web:latest \
  APP/

# Push to registry (requires --push)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t registry.example.com/moda-flutter-web:latest \
  APP/
```

---

## Registry Deployment

### Push to Docker Hub

```bash
# Tag images
docker tag moda-flutter-web:latest myusername/moda-flutter-web:latest
docker tag moda-signal-mock:latest myusername/moda-signal-mock:latest

# Login
docker login

# Push
docker push myusername/moda-flutter-web:latest
docker push myusername/moda-signal-mock:latest
```

### Pull from Registry

Update `docker-compose.yml`:

```yaml
flutter-emulator:
  image: myusername/moda-flutter-web:latest  # Instead of building locally
  depends_on:
    fastmoda-api:
      condition: service_healthy
  # ... rest of config
```

Then:

```bash
docker-compose pull
docker-compose up -d flutter-emulator
```

### Private Registry (e.g., AWS ECR)

```bash
# Create ECR repository
aws ecr create-repository --repository-name moda-flutter-web

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag moda-flutter-web:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/moda-flutter-web:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/moda-flutter-web:latest
```

---

## Layer Optimization

### View Image Size

```bash
docker images | grep moda

# Detailed breakdown
docker history moda-flutter-web:latest
```

### Reduce Flutter App Size

In `APP/Dockerfile.web`, the build is already optimized:

```dockerfile
RUN flutter build web \
  --release \
  --no-tree-shake-icons  # Only if keeping tree-shaking disabled
```

### Use Multi-Stage for Smaller Images

Already implemented in `Dockerfile.web`:

```dockerfile
FROM ghcr.io/cirruslabs/flutter:latest AS builder
# Large build image with all tools

FROM nginx:alpine
# Small runtime image (~80 MB)
COPY --from=builder /app/build/web /usr/share/nginx/html
```

Typical sizes:
- **flutter-emulator**: 80-120 MB
- **moda-signal-mock**: 150-200 MB (Python 3.11)
- **fastmoda-api**: 400-600 MB (PyTorch)

---

## Local Development Build

For faster iteration during development:

### Using Flutter Dev Server (instead of Nginx)

Create `APP/Dockerfile.dev`:

```dockerfile
FROM ghcr.io/cirruslabs/flutter:latest

WORKDIR /app
COPY . .

RUN flutter config --enable-web && flutter pub get

EXPOSE 8000

CMD ["flutter", "run", "-d", "web-server", "--web-port", "8000"]
```

Then in `docker-compose.dev.yml`:

```yaml
version: '3.8'
services:
  flutter-emulator:
    build:
      context: ./APP
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./APP/lib:/app/lib  # Hot reload
    depends_on:
      - fastmoda-api
    networks:
      - moda-net
```

Run:

```bash
docker-compose -f docker-compose.dev.yml up flutter-emulator
```

### Using Local Flutter Dev Server

For fastest iteration, run Flutter locally:

```bash
cd APP

# Configure FastMODA URL
export FASTMODA_URL=http://localhost:5000

# Run web dev server
flutter run -d web-server --dart-define=FASTMODA_URL=$FASTMODA_URL

# App opens at http://localhost:59000 (or similar)
```

Still run FastMODA in Docker:

```bash
docker-compose up -d fastmoda-api moda-signal-mock-server
```

---

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/build-emulator.yml`:

```yaml
name: Build Emulator Images

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build Flutter Emulator
        uses: docker/build-push-action@v4
        with:
          context: ./APP
          file: ./APP/Dockerfile.web
          push: false
          tags: moda-flutter-web:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            FASTMODA_URL=http://localhost:5000
      
      - name: Build Signal Mock Server
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.signal-mock
          push: false
          tags: moda-signal-mock:${{ github.sha }}
      
      - name: Run Smoke Tests
        run: |
          docker-compose build
          docker-compose up -d
          sleep 30
          python3 tests/emulator_smoke_tests.py
```

### GitLab CI Example

Create `.gitlab-ci.yml`:

```yaml
stages:
  - build
  - test

build-flutter:
  stage: build
  image: docker:latest
  script:
    - docker build --build-arg FASTMODA_URL=http://fastmoda-api:5000 -f APP/Dockerfile.web -t moda-flutter-web:$CI_COMMIT_SHA APP/

build-signal-mock:
  stage: build
  image: docker:latest
  script:
    - docker build -f Dockerfile.signal-mock -t moda-signal-mock:$CI_COMMIT_SHA .

integration-test:
  stage: test
  image: docker-compose
  script:
    - docker-compose up -d
    - sleep 30
    - python3 tests/emulator_smoke_tests.py
```

---

## Troubleshooting Builds

### Build Hangs on Flutter Compilation

```bash
# Increase timeout
docker-compose build --build-arg BUILDKIT_STEP_LOG_MAX_SIZE=10000000 flutter-emulator
```

### Out of Disk Space

```bash
# Clean up old images
docker system prune -a --volumes

# Then rebuild
docker-compose build --no-cache flutter-emulator
```

### Python Pip Errors

```bash
# View detailed logs
DOCKER_BUILDKIT=0 docker build -f Dockerfile.signal-mock -t test .
```

### Nginx Config Syntax Error

```bash
# Validate config inside container
docker run --rm -v $(pwd)/APP/nginx.conf:/etc/nginx/conf.d/default.conf:ro nginx:alpine nginx -t
```

---

## Build Caching Strategy

Docker caching works best with logical layer ordering:

```dockerfile
# Good: Put stable layer first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only changes to code trigger rebuild
COPY emulator_refactored.py .

# Bad: Code changes trigger dependency reinstall
COPY emulator_refactored.py .
RUN pip install -r requirements.txt
```

---

## References

- [Docker Build Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Flutter Web Deployment](https://docs.flutter.dev/deployment/web)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Build](https://docs.docker.com/compose/compose-file/#build)

---

**Last updated**: May 2, 2026
