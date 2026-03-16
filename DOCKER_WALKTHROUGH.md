# Docker & Testing Walkthrough for MODA

**Date:** March 5, 2026  
**Audience:** Developers getting MODA into Docker containers

This guide walks you through dockerizing and testing MODA step-by-step, from building your first image to running a complete test suite.

---

## What We're Building

By the end of this walkthrough, you'll have:

✅ **Development Container** - MATLAB GUI with your code  
✅ **Test Container** - Automated headless testing  
✅ **Docker Compose Stack** - Multiple services working together  
✅ **CI/CD Ready** - Automated tests on every code change

---

## Part 1: Prerequisites (15 minutes)

### Step 1.1: Install Docker

Check if Docker is installed:
```bash
docker --version
```

If not: [Install Docker Desktop](https://docs.docker.com/get-docker/)

### Step 1.2: Install Docker Compose

```bash
docker-compose --version
```

If not installed:
```bash
pip install docker-compose
# or (on macOS with Homebrew)
brew install docker-compose
```

### Step 1.3: Check Disk Space

MATLAB Docker images are large (~15GB):

```bash
df -h  # Need ~50GB free
```

If low on space, consider:
- Cleaning old Docker images: `docker image prune`
- Using external storage
- Building smaller test-only images

### Step 1.4: Quick Verification

```bash
bash docker_quickstart.sh check
```

Expected output:
```
=== MODA Docker Quick Start ===
✓ Docker found (Docker version 24.0.0, build build123)
✓ Docker Compose found  
✓ Disk space available: 250G
```

---

## Part 2: Build Your First Image (20 minutes)

### Step 2.1: Review the Dockerfile

The Dockerfile has 3 stages:

```dockerfile
FROM mathworks/matlab:r2024b as matlab-dev     # Stage 1: Development
FROM mathworks/matlab:r2024b as moda-test      # Stage 2: Testing  
FROM mathworks/matlab:r2024b as moda-prod      # Stage 3: Production
```

View it:
```bash
cat Dockerfile
```

### Step 2.2: Build the Development Image

```bash
docker build -t moda-dev:latest --target matlab-dev .
```

**What's happening:**
1. Pulls MATLAB R2024b image from MathWorks
2. Installs git, curl, vim
3. Copies MODA source code into `/app`
4. Configures MATLAB path
5. Verifies MATLAB loads correctly

**Expected output:**
```
Step 1/15 : FROM mathworks/matlab:r2024b as matlab-dev
...
Step 15/15 : ENV MATLABPATH=/app:/app/allguis/codes:/app/allguis/guis
---> Successfully built abc123def456
---> Successfully tagged moda-dev:latest
```

**First build takes 10-15 minutes** (subsequent builds use cache)

### Step 2.3: Verify the Image

```bash
docker images | grep moda-dev
```

Output:
```
REPOSITORY   TAG      IMAGE ID      SIZE
moda-dev     latest   abc123def456  15.2GB
```

### Step 2.4: Try Running the Container

Quick test (no GUI yet):
```bash
docker run --rm moda-dev:latest matlab -batch "disp('Hello from Docker'); exit(0);"
```

Expected output:
```
Hello from Docker
```

✅ **Success!** You've built and run your first MODA container.

---

## Part 3: Interactive Development (optional - GUI on Linux only)

### Step 3.1: Check for X11 Display

```bash
echo $DISPLAY
```

If empty:
```bash
export DISPLAY=:0
```

### Step 3.2: Allow Docker X11 Access

```bash
xhost +local:docker
```

### Step 3.3: Run with GUI

```bash
docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/home/matlab/.Xauthority:ro \
  moda-dev:latest \
  matlab -r "MODA; exit;"
```

**What you see:**
- MATLAB starts
- MODA GUI loads
- You can interact with the application

**When done in MATLAB:**
```
>> exit
```

Then cleanup:
```bash
xhost -local:docker
```

⚠️ **GUI only works on Linux with X11.** Skip this section on Windows/Mac - move to testing instead.

---

## Part 4: Build Test Image (15 minutes)

### Step 4.1: Build

```bash
docker build -t moda-test:latest --target moda-test .
```

This creates a minimal test-only image (headless, no display).

### Step 4.2: Check

```bash
docker images | grep moda-test
```

Should be similar size to `moda-dev` (same MATLAB base).

---

## Part 5: Create Test Files (15 minutes)

### Step 5.1: Create Tests Directory

```bash
mkdir -p tests
ls tests/
```

You should see:
- `test_algorithms.m` - Example test file (we created this)

### Step 5.2: Understand the Test Structure

The test file contains 10 test functions:

```matlab
test 1: MATLAB Version Check       ✓ Verify R2023a+
test 2: Toolbox Availability       ✓ Signal, Wavelet, Stats
test 3: CSV Read/Write             ✓ readmatrix/writematrix
test 4: MAT File Read/Write        ✓ save/load
test 5: Wavelet Transform          ✓ Core algorithm
test 6: String Functions           ✓ contains vs strfind
test 7: File Operations            ✓ All files present
test 8: MODA App Structure         ✓ App Designer class
test 9: Deprecated Functions       ✓ No csvread/csvwrite
test 10: Path and Loading          ✓ Modules loadable
```

### Step 5.3: Run Tests Locally First (Optional)

```bash
cd /home/user/MODA
matlab -batch "addpath(genpath('.')); runtests tests/test_algorithms.m; exit(0);"
```

This runs before Docker to verify tests work.

---

## Part 6: Run Tests in Docker (20 minutes)

### Step 6.1: Simple Test Run

```bash
docker run --rm \
  -v $(pwd):/app \
  moda-test:latest \
  matlab -batch "addpath(genpath('.')); runtests tests/test_algorithms.m; exit(0);"
```

**What happens:**
1. Container starts
2. MATLAB loads
3. MODA code found in path
4. Tests run
5. Results printed
6. Container exits

**Expected output:**
```
Running test_algorithms
============================
Testing MATLAB Version... PASSED ✓
Testing Toolboxes... PASSED ✓
Testing CSV I/O... PASSED ✓
Testing MAT I/O... PASSED ✓
Testing Wavelet Transform... PASSED ✓
...
============================
Totals: 10 passed, 0 failed
```

### Step 6.2: Save Test Results

```bash
mkdir -p test_results

docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/test_results:/tmp/test_results \
  moda-test:latest \
  matlab -batch "addpath(genpath('.')); runtests tests/test_algorithms.m -ToFile /tmp/test_results/results.txt; exit(0);"

cat test_results/results.txt
```

### Step 6.3: Debug Failed Tests

If a test fails, run with verbose output:

```bash
docker run -it \
  -v $(pwd):/app \
  moda-test:latest \
  matlab -batch "addpath(genpath('.')); runtests tests/test_algorithms.m -Verbose; exit(0);"
```

The `-it` flags make it interactive so you can see detailed error messages.

---

## Part 7: Docker Compose (Full Stack)

### Step 7.1: Understand docker-compose.yml

The file defines 3 services:

```yaml
services:
  moda-dev:           # Development container
    image: moda-dev:latest
    
  moda-test:         # Test container
    image: moda-test:latest
    
  fastmoda-api:      # Python API (optional)
    image: fastmoda:latest
```

### Step 7.2: Start the Stack

```bash
docker-compose up -d
```

Check running services:
```bash
docker-compose ps
```

Expected output:
```
CONTAINER ID   IMAGE                COMMAND
abc123...      moda-test:latest     "bash -c ..."
def456...      moda-dev:latest      "matlab ..."
ghi789...      fastmoda:latest      "python app.py"
```

### Step 7.3: View Test Logs

```bash
docker-compose logs -f moda-test
```

Exit with `Ctrl+C`.

### Step 7.4: Access Services

**MODA Dev Container:**
```bash
docker-compose exec moda-dev matlab
```

**FastMODA API:**
```bash
curl http://localhost:5000/health
```

**Run one-off test:**
```bash
docker-compose run --rm moda-test matlab -batch "runtests; exit(0);"
```

### Step 7.5: Stop Everything

```bash
docker-compose down
```

This stops and removes all containers but keeps images.

---

## Part 8: Automated Testing Workflow

### Step 8.1: Create Integration Test Script

**File:** `tests/test_integration.sh`

```bash
#!/bin/bash
set -e

echo "=== MODA Integration Tests ==="

# Test 1: MATLAB accessibility
echo "[1/3] Testing MATLAB..."
matlab -batch "disp('MATLAB OK'); exit(0);"

# Test 2: Module loading
echo "[2/3] Testing module loading..."
matlab -batch "which MODA; which TimeFrequencyAnalysis; exit(0);"

# Test 3: Unit tests
echo "[3/3] Running unit tests..."
matlab -batch "runtests tests/test_algorithms.m; exit(0);"

echo "✓ All tests passed"
```

Make executable:
```bash
chmod +x tests/test_integration.sh
```

### Step 8.2: Run Integration Tests

In Docker:
```bash
docker run --rm \
  -v $(pwd):/app \
  moda-test:latest \
  bash tests/test_integration.sh
```

Output shows progress:
```
=== MODA Integration Tests ===
[1/3] Testing MATLAB...
MATLAB OK
[2/3] Testing module loading...
moda-dev:/app/MODA.m
...
[3/3] Running unit tests...
Totals: 10 passed, 0 failed
✓ All tests passed
```

---

## Part 9: Using the Quick Start Script

The `docker_quickstart.sh` script automates everything.

### Interactive Menu

```bash
bash docker_quickstart.sh
```

Choose from options:
```
1) Check prerequisites
2) Build dev image
3) Build test image
4) Run development
5) Run tests
6) Start with Docker Compose
7) Clean up
```

### Direct Commands

```bash
bash docker_quickstart.sh check       # Verify setup
bash docker_quickstart.sh dev         # Build & run dev
bash docker_quickstart.sh test        # Build & run tests
bash docker_quickstart.sh compose     # Start stack
bash docker_quickstart.sh clean       # Remove images
```

---

## Part 10: Troubleshooting

### Problem: Build fails with "License not found"

**Solution:** MathWorks containers need a license. Options:

**Option A: Network License**
```bash
docker run -e MLM_LICENSE_FILE=hostname@port moda-dev
```

**Option B: File-based License**
```bash
docker run -v /path/to/license.dat:/MATLAB/licenses/license.dat moda-dev
```

**Option C: MATLAB Online** (if available)
```bash
docker run -e MATLAB_ONLINE=1 moda-dev
```

### Problem: "Out of memory" error

**Solution:** Increase container memory:

```bash
docker run -m 8g --memory-swap 8g moda-test:latest matlab ...
```

Or in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### Problem: "Module not found" in tests

**Solution:** Verify MATLABPATH:

```bash
docker run moda-test:latest matlab -batch "path; exit(0);" | grep /app
```

### Problem: X11 "connection refused"

**Ensure:** 
```bash
echo $DISPLAY  # Should show :0 or :1
xhost +local:docker  # Grant permission
```

### Problem: Slow builds

**Use cache:**
```bash
docker build --cache-from moda-dev:latest -t moda-dev:latest .
```

---

## Summary Checklist

- [ ] Docker and Docker Compose installed
- [ ] Built `moda-dev:latest` image
- [ ] Built `moda-test:latest` image
- [ ] Created `tests/test_algorithms.m`
- [ ] Tests pass in Docker container
- [ ] Docker Compose stack runs
- [ ] Can view test logs
- [ ] Quick start script works
- [ ] Ready to integrate with CI/CD

---

## Next Steps

1. **Add more tests** for specific algorithms (Bayes, filtering, etc.)
2. **Set up CI/CD** (GitHub Actions template provided in `.github/workflows/`)
3. **Deploy to production** using the `moda-prod` stage
4. **Monitor with health checks** defined in docker-compose.yml
5. **Scale with Kubernetes** for larger deployments

---

## Quick Reference Commands

```bash
# Build
docker build -t moda-dev:latest --target matlab-dev .
docker build -t moda-test:latest --target moda-test .

# Run
docker run --rm moda-test:latest matlab -batch "runtests; exit(0);"

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down

# Quick Start
bash docker_quickstart.sh [check|dev|test|compose|clean]
```

---

## Resources

📖 **Comprehensive Guide:** `docs/DOCKER_SETUP_GUIDE.md`  
📖 **Quick Reference:** `DOCKER_QUICKREF.md`  
🐳 **Official MATLAB Docker:** https://github.com/mathworks-ref-arch/matlab-dockerfile  
🐳 **Docker Docs:** https://docs.docker.com/  

---

**You're all set!** MODA is now containerized, tested, and ready for deployment. 🚀
