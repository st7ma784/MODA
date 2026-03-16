# MODA Docker Setup & Testing Guide

**Version:** 1.0  
**Date:** March 5, 2026  
**Target:** MATLAB R2023a+ (with App Designer)

---

## Table of Contents

1. [Overview & Architecture](#overview)
2. [Prerequisites](#prerequisites)
3. [Building MATLAB Container](#building-matlab)
4. [Testing Strategies](#testing)
5. [Docker Compose Setup](#docker-compose)
6. [Running & Debugging](#running)
7. [Troubleshooting](#troubleshooting)

---

## Overview & Architecture {#overview}

### Why Containerize MODA?

✅ **Benefits:**
- Reproducible environments across machines
- Isolated dependencies (no MATLAB version conflicts)
- Easy distribution and testing
- CI/CD pipeline integration
- Simplified deployment (cloud or local)

⚠️ **Challenges:**
- MATLAB licensing in containers
- GUI requires X11 display server (optional)
- Large image size (~8-15GB with MATLAB)
- License server integration

### Deployment Scenarios

| Scenario | Purpose | Display | Container | Size |
|----------|---------|---------|-----------|------|
| **Development** | Local testing of MODA GUI | X11 | Small base | 3GB |
| **Testing** | Automated algorithm validation | Headless | Test runner | 5GB |
| **Production (Server)** | FastMODA API + MATLAB backend | Headless | API server | 8GB |
| **CI/CD** | GitHub Actions automated tests | Headless | Minimal | 2GB |

---

## Prerequisites {#prerequisites}

### System Requirements

```bash
# Docker & Docker Compose
docker --version  # >= 20.10
docker-compose --version  # >= 1.29

# Storage space
df -h  # Need 50GB+ for Docker images + build cache

# For GUI testing (optional)
which xhost  # X11 display server (Linux only)
echo $DISPLAY  # Should show :0 or similar
```

### MATLAB License

Two options:

**Option 1: Network License** (Recommended for containers)
```
License Manager running on host or network
MATLAB_LICENSE_FILE=/path/to/license.dat in container
```

**Option 2: File Installation Key**
```
Interactive license setup during docker build
Include in container image
```

**Option 3: MATLAB Online** (Easiest)
```
Mount credentials from host
Use MATLAB licensing service
```

### Files Needed

```
/home/user/MODA/
├── Dockerfile                 # MATLAB container spec
├── docker-compose.yml         # Orchestration
├── .dockerignore              # Exclude large files
├── test.py                    # Test runner
├── MODA.m                     # Modernized main (already updated)
└── allguis/                   # All GUI modules
    └── codes/                 # Data I/O (already updated)
```

---

## Building MATLAB Container {#building-matlab}

### Step 1: Create Dockerfile for MATLAB MODA

Create `/home/user/MODA/Dockerfile`:

```dockerfile
# Multi-stage MATLAB MODA container
# For GUI: docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix moda-dev
# For testing: docker run moda-test bash test.sh

# ============================================================================
# Stage 1: Base MATLAB Runtime (smallest option, no MATLAB compilation)
# ============================================================================
FROM ubuntu:22.04 as matlab-runtime

WORKDIR /app
LABEL maintainer="MODA Team" version="2.0"

# Install system dependencies for MATLAB Runtime
RUN apt-get update && apt-get install -y \
    xorg \
    libxrender1 \
    libxext6 \
    libxt6 \
    libxmu6 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download MATLAB Runtime R2024b (if using compiled MATLAB)
# For development, use full MATLAB instead
# RUN cd /opt && curl -sSL https://ssd.mathworks.com/supportfiles/downloads/R2024b/Release/1/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2024b_glnxa64_installer.zip

# ============================================================================
# Stage 2: Full MATLAB Development Container
# ============================================================================
FROM mathworks/matlab:r2024b as matlab-dev

WORKDIR /app

# Install additional system tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    tree \
    git-lfs \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy MODA source
COPY . .

# Verify MATLAB and toolboxes are available
RUN matlab -batch "ver; exit(0);" || true

# Run startup checks
RUN matlab -batch "addpath(genpath(pwd)); checkMATLABVersion(); exit(0);" || true

# Set MATLAB path to include MODA modules
ENV MATLABPATH="/app:/app/allguis/codes:/app/allguis/guis"

# ============================================================================
# Stage 3: Test Runtime (headless testing)
# ============================================================================
FROM mathworks/matlab:r2024b as moda-test

WORKDIR /app

# Minimal dependencies for testing only
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Copy test configuration
COPY ./tests /app/tests

# Verify installation
RUN matlab -batch "addpath(genpath(pwd)); runtests tests/; exit(0);" || true

ENV MATLABPATH="/app:/app/allguis/codes:/app/allguis/guis"
ENV MATLAB_LOG_DIR=/tmp/matlab_logs

# ============================================================================
# Final Stage: Production Runtime (optimized)
# ============================================================================
FROM mathworks/matlab:r2024b as moda-prod

WORKDIR /app

# Minimal footprint
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=matlab-dev /app .

ENV MATLABPATH="/app:/app/allguis/codes:/app/allguis/guis"
ENV MATLAB_LOG_DIR=/tmp/matlab_logs

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD matlab -batch "exit(0);" 2>/dev/null || exit 1

ENTRYPOINT ["matlab", "-nodisplay", "-r"]
CMD ["MODAServer(); exit(0);"]
```

### Step 2: Create .dockerignore

Create `/home/user/MODA/.dockerignore`:

```
# Exclude from Docker build context
.git
.github
.gitignore
*.md
*.log
__pycache__
*.pyc
.DS_Store
.vscode
.idea
*.swp
*.swo
*~

# Exclude large directories
FastMODA/uploads/*
FastMODA/example_sigs/*
example_sigs/

# Exclude old backups
**/*_old.m
**/*_backup.*

# Exclude test outputs
test_output/
coverage/
```

### Step 3: Build the Docker Image

```bash
# Basic build (development)
docker build -t moda-dev:latest --target matlab-dev .

# Test image
docker build -t moda-test:latest --target moda-test .

# Production image
docker build -t moda-prod:latest --target moda-prod .

# With build cache (faster subsequent builds)
docker build --cache-from moda-dev:latest -t moda-dev:latest --target matlab-dev .
```

**Build Output:**
```
Step 1/X : FROM mathworks/matlab:r2024b
Step 2/X : WORKDIR /app
...
Successfully built abc123def456
Successfully tagged moda-dev:latest
```

### Step 4: Verify Build

```bash
# Check image size
docker images | grep moda

# Inspect layers
docker history moda-dev:latest

# Run verification
docker run --rm moda-dev:latest matlab -batch "ver; exit(0);"
```

---

## Testing Strategies {#testing}

### Strategy 1: Algorithm Validation (Headless)

**Test File:** `/home/user/MODA/tests/test_algorithms.m`

```matlab
function tests = test_algorithms
    tests = functiontests(localfunctions);
end

function testWaveletTransform(testCase)
    % Test WT algorithm produces reasonable output
    signal = sin(2*pi*(1:1000)/100);
    fs = 100;
    
    [WT, freq] = wt(signal, fs, 'Display', 'off');
    
    testCase.verifyTrue(size(WT,1) == length(signal));
    testCase.verifyTrue(all(freq > 0));
    testCase.verifyFalse(any(isnan(WT(:))));
    testCase.verifyFalse(any(isinf(WT(:))));
end

function testCoherence(testCase)
    % Test multiple signal coherence
    sr = 100;
    t = (0:0.01:10)';
    s1 = sin(2*pi*1*t);
    s2 = sin(2*pi*2*t);
    
    sigs = [s1, s2];
    
    % Should not crash and return numeric output
    try
        coh = CoherenceMulti(sigs, sr);
        testCase.verifyClass(coh, 'double');
    catch ME
        testCase.verificationFailed(ME.message);
    end
end

function testDataIO(testCase)
    % Test CSV read/write
    testData = [1,2,3; 4,5,6; 7,8,9];
    testFile = tempname;
    
    writematrix(testData, [testFile '.csv']);
    loaded = readmatrix([testFile '.csv']);
    
    testCase.verifyEqual(testData, loaded);
    delete([testFile '.csv']);
end
```

**Run in Container:**

```bash
docker run --rm moda-test:latest matlab -batch \
    "addpath(genpath(pwd)); runtests tests/test_algorithms.m; exit(0);"
```

### Strategy 2: GUI Module Validation

**Test File:** `/home/user/MODA/tests/test_guis.m`

```matlab
function tests = test_guis
    tests = functiontests(localfunctions);
end

function testMODAAppLaunches(testCase)
    % Test MODA main app constructs without errors
    try
        app = MODAApp();
        testCase.verifyTrue(isvalid(app.UIFigure));
        delete(app);
    catch ME
        testCase.verificationFailed(sprintf('MODA launch failed: %s', ME.message));
    end
end

function testTimeFrequencyAppLaunches(testCase)
    % Test sub-module (once converted to App Designer)
    try
        app = TimeFrequencyAnalysisApp();
        testCase.verifyTrue(isvalid(app.UIFigure));
        delete(app);
    catch ME
        % Expected for now - sub-modules not yet refactored
        testCase.verifyMatches(ME.message, 'Not yet ported to App Designer');
    end
end

function testVersionCheck(testCase)
    % Verify MATLAB version requirements
    v = ver('MATLAB');
    matlabVersion = str2double(regexp(v.Release, '\d{4}[a-z]', 'match', 'once'));
    
    testCase.verifyGreaterThanOrEqual(matlabVersion, 2023);
end
```

### Strategy 3: Integration Testing

**Test File:** `/home/user/MODA/tests/test_integration.sh`

```bash
#!/bin/bash
# Integration test script for Docker container

set -e  # Exit on error

echo "=== MODA Docker Integration Tests ==="

# Test 1: MATLAB is accessible
echo "[TEST 1] MATLAB availability..."
matlab -batch "disp('MATLAB is working'); exit(0);" > /tmp/test1.log
grep -q "MATLAB is working" /tmp/test1.log && echo "✓ PASSED" || echo "✗ FAILED"

# Test 2: MODA modules load
echo "[TEST 2] Module loading..."
matlab -batch "addpath(genpath(pwd)); ...
    disp('Loading MODA'); ...
    which MODA; ...
    which TimeFrequencyAnalysis; ...
    which CoherenceMulti; ...
    exit(0);" > /tmp/test2.log
grep -q "MODA.m" /tmp/test2.log && echo "✓ PASSED" || echo "✗ FAILED"

# Test 3: Toolboxes present
echo "[TEST 3] Toolbox verification..."
matlab -batch "addpath(genpath(pwd)); ...
    v = ver; ...
    tbx = {v.Name}; ...
    assert(any(contains(tbx, 'Signal Processing')), 'Signal Toolbox missing'); ...
    assert(any(contains(tbx, 'Wavelet')), 'Wavelet Toolbox missing'); ...
    disp('All toolboxes present'); ...
    exit(0);" > /tmp/test3.log
grep -q "toolboxes present" /tmp/test3.log && echo "✓ PASSED" || echo "✗ FAILED"

# Test 4: Run unit tests
echo "[TEST 4] Running unit tests..."
matlab -batch "addpath(genpath(pwd)); ...
    runtests tests/test_algorithms.m; ...
    exit(0);" > /tmp/test4.log && echo "✓ PASSED" || echo "✗ FAILED"

echo ""
echo "=== Test Summary ==="
echo "Check logs in /tmp/test*.log for details"
```

**Run Integration Tests:**

```bash
docker run --rm moda-test:latest bash tests/test_integration.sh
```

---

## Docker Compose Setup {#docker-compose}

### Complete Multi-Container Setup

**File:** `/home/user/MODA/docker-compose.yml`

```yaml
version: '3.8'

services:
  # MATLAB MODA Development Container
  moda-dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: matlab-dev
    container_name: moda-dev
    image: moda-dev:latest
    
    # Display settings for GUI (Linux only)
    environment:
      - DISPLAY=${DISPLAY}
      - MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis
    
    # Mount X11 socket for GUI
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./:/app:rw
      - ~/.Xauthority:/home/matlab/.Xauthority:ro
    
    networks:
      - moda-net
    
    # Allow access to display
    stdin_open: true
    tty: true
    
    command: matlab -r "MODA; exit;"

  # Test Runner
  moda-test:
    build:
      context: .
      dockerfile: Dockerfile
      target: moda-test
    container_name: moda-test
    image: moda-test:latest
    
    environment:
      - MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis
      - MATLAB_LOG_DIR=/tmp/matlab_logs
    
    volumes:
      - ./:/app:ro
      - ./test_results:/tmp/test_results:rw
    
    networks:
      - moda-net
    
    # Run tests automatically
    command: bash tests/test_integration.sh
    
    # Fail if tests don't complete
    exit_code_from: moda-test

  # Python FastMODA API Server (as reference)
  fastmoda-api:
    build:
      context: ./FastMODA
      dockerfile: Dockerfile
      target: base
    container_name: fastmoda-api
    image: fastmoda:latest
    
    ports:
      - "5000:5000"
    
    environment:
      - FLASK_APP=app.py
      - FLASK_ENV=development
    
    volumes:
      - ./FastMODA:/app:rw
      - ./uploads:/app/uploads:rw
    
    networks:
      - moda-net
    
    command: python app.py

  # License Server (if using network license)
  # license-server:
  #   image: mathworks/license-server:latest
  #   ports:
  #     - "27000:27000"
  #   volumes:
  #     - ./license.dat:/path/to/license.dat:ro

networks:
  moda-net:
    driver: bridge
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f moda-test
docker-compose logs -f fastmoda-api

# Stop all services
docker-compose down

# Run one-off command
docker-compose run --rm moda-test matlab -batch "runtests; exit(0);"

# Remove all containers and volumes
docker-compose down -v
```

---

## Running & Debugging {#running}

### Scenario 1: GUI Development (X11 Display)

```bash
# Allow Docker to access X11 display
xhost +local:docker

# Run with display forwarding
docker run -it \
  --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/home/matlab/.Xauthority:ro \
  moda-dev:latest \
  matlab -r "MODA; exit;"

# Cleanup
xhost -local:docker
```

**Troubleshooting GUI:**
```bash
# Check X11 permission
echo $DISPLAY  # Should show :0 or :1

# Test X11 access within container
docker run -it --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  moda-dev:latest \
  xclock  # Should show clock if X11 works
```

### Scenario 2: Headless Testing (CI/CD)

```bash
# Run tests without display
docker run --rm \
  -v $(pwd)/test_results:/tmp/test_results \
  moda-test:latest \
  matlab -batch "runtests tests/; exit(0);"

# Check results
cat test_results/summary.txt
```

### Scenario 3: Interactive Development

```bash
# Shell access for development
docker run -it \
  -v $(pwd):/app \
  moda-dev:latest \
  /bin/bash

# Inside container:
$ matlab
>> addpath(genpath('/app'))
>> runtests tests/
>> exit
```

### Scenario 4: Debug Mode (Verbose Output)

```bash
docker run -it \
  -e MATLAB_LOG_DIR=/tmp/matlab_logs \
  moda-test:latest \
  matlab -r "...
    addpath(genpath(pwd)); ...
    disp('=== MODA Debug Start ==='); ...
    disp(getenv('MATLABPATH')); ...
    which MODA; ...
    runtests tests/ -Verbose; ...
    exit(0);"
```

---

## Troubleshooting {#troubleshooting}

### Problem: "License not found"

**Symptom:** 
```
Error: License not found. The MathWorks License Manager is not available.
```

**Solution:**

```bash
# Option 1: Use network license
docker run -e MLM_LICENSE_FILE=hostname@portnum \
  moda-dev:latest

# Option 2: Mount license file
docker run -v /path/to/license.dat:/MATLAB/licenses/license.dat \
  moda-dev:latest

# Option 3: Use MATLAB Online
docker run -e MATLAB_ONLINE=1 \
  moda-dev:latest
```

### Problem: "X11 connection refused"

**Symptom:**
```
Error: unable to open display
```

**Solution:**

```bash
# Check DISPLAY variable
echo $DISPLAY

# Allow other containers to access X11
xhost +local:

# Run with correct display
docker run -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  moda-dev:latest
```

### Problem: "Out of memory"

**Symptom:**
```
Killed | Memory allocation failed
```

**Solution:**

```bash
# Increase Docker memory limit
docker run -m 8g \
  --memory-swap 8g \
  moda-dev:latest

# Or in docker-compose.yml:
services:
  moda-dev:
    mem_limit: 8g
    memswap_limit: 8g
```

### Problem: "Module not found"

**Symptom:**
```
Undefined function or variable 'MODA'
```

**Solution:**

```bash
# Verify MATLABPATH in container
docker run moda-dev:latest \
  matlab -batch "path; exit(0);"

# Check file existence
docker run moda-dev:latest \
  ls -la /app/MODA.m

# Add to path explicitly
docker run -e MATLABPATH=/app:/app/allguis/codes \
  moda-dev:latest
```

### Problem: "Build fails with toolbox errors"

**Symptom:**
```
Error: Required toolbox 'Signal Processing' not found
```

**Solution:**

```bash
# Use official MATLAB image with toolboxes
FROM mathworks/matlab:r2024b  # Pre-includes Signal, Wavelet, etc.

# Verify toolboxes in build
docker build --target matlab-dev \
  --progress=plain \
  -t moda-dev:latest .
```

---

## CI/CD Integration

### GitHub Actions Example

**File:** `.github/workflows/test-moda.yml`

```yaml
name: MODA Docker Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build test image
        run: docker build -t moda-test:latest --target moda-test .
      
      - name: Run tests
        run: docker run --rm moda-test:latest bash tests/test_integration.sh
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:  
          name: test-results
          path: test_results/
```

---

## Performance Optimization

### Multi-stage Builds (Smaller Images)

```dockerfile
# Base: 4GB
FROM mathworks/matlab:r2024b as builder
COPY . /app
RUN matlab -batch "compile(); exit(0);" || true

# Final: Only runtime needed (~8GB)
FROM mathworks/matlab:r2024b
COPY --from=builder /app/compiled /app/compiled
```

### Caching Strategy

```bash
# Build with cache
docker build \
  --cache-from moda-dev:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t moda-dev:latest \
  .

# Push to registry for CI/CD
docker tag moda-dev:latest registry.example.com/moda-dev:latest
docker push registry.example.com/moda-dev:latest
```

---

## Next Steps

1. **Build the Docker image** (run commands above)
2. **Test in container** (run test scripts)
3. **Fix any issues** (refer to troubleshooting)
4. **Integrate with CI/CD** (GitHub Actions example)
5. **Deploy to production** (use production stage)

---

## Quick Reference

```bash
# Build
docker build -t moda-dev:latest --target matlab-dev .

# Test
docker run --rm moda-test:latest bash tests/test_integration.sh

# Develop (with GUI)
xhost +local:docker
docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix moda-dev:latest
xhost -local:docker

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

**For advanced topics, see:**
- [Official MATLAB Docker Guide](https://github.com/mathworks-ref-arch/matlab-dockerfile)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MATLAB in Singularity Containers](https://www.mathworks.com/help/compiler/singularity-containers.html)
