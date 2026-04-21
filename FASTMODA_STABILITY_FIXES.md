# FastMODA Stability Fixes - Rancher Deployment

**Status**: ✅ STABLE - Critical startup crash fixed

## Problem Summary

FastMODA containers were entering `CrashLoopBackOff` on Rancher/Kubernetes with no visible error logs. The issue occurred before Flask even started, during Python startup phase.

## Root Cause

**Critical Bug in app.py (Lines 68-70)**:
```python
# OLD CODE (CRASHED):
if USE_GPU:
    print(f"Device: {torch.cuda.get_device_name(0)}")  # ❌ NameError if torch not imported
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Issue**: When PyTorch GPU modules failed to import (CPU-only environments), `TORCH_AVAILABLE` was False, but the code still tried to access undefined `torch.cuda` methods. This caused:
- **NameError** (torch undefined) or **AttributeError** (torch.cuda doesn't exist)
- Crash occurs BEFORE the application binds to port 5000
- Kubernetes health checks never connect, pod marked as failed
- Logs unavailable because crash happens during Python initialization

## Fixes Applied

### 1. **app.py - Fixed torch reference safety (Lines 60-70)**
```python
# NEW CODE (SAFE):
if USE_GPU and TORCH_AVAILABLE:  # ✅ Check TORCH_AVAILABLE first
    try:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"Warning: Could not query GPU info: {e}")
```

**Changes**:
- Added `TORCH_AVAILABLE` check before accessing `torch.cuda`
- Wrapped GPU info queries in try/except
- Gracefully continues on any error
- Added safety to GPU detection logic in USE_GPU assignment

### 2. **docker-compose.yml - Removed GPU-specific references**
**Before**:
```yaml
build:
  target: final  # ❌ Non-existent build target
  args:
    - CUDA_VERSION=11.8
    - BUILD_TARGET=gpu
environment:
  - USE_GPU=true  # ❌ Hardcoded - fails on CPU
healthcheck:
  test: ["CMD", "curl", "-f", ...]  # ❌ curl not in slim image
```

**After**:
```yaml
build:
  dockerfile: Dockerfile  # ✅ Correct reference
environment:
  - USE_GPU=auto  # ✅ Auto-detects GPU availability
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; ..."]  # ✅ Uses Python
```

## Testing Results

**Container startup verification** (test run):
```
[INFO] Starting gunicorn 25.3.0
[INFO] Listening at: http://0.0.0.0:5000
[INFO] Using worker: sync
[INFO] Booting worker with pid: 7/8/9/10
FastMODA OPTIMIZED - Starting
Backend: CPU
127.0.0.1 - - [GET /health HTTP/1.1] 200 16  ✅ Health check passes
```

**No crashes** ✅
**Graceful GPU fallback** ✅
**Health check responds** ✅

## Deployment Instructions

### For local testing:
```bash
cd /home/user/MODA
docker build -t st7ma784/moda-fastmoda:latest-cpu -f FastMODA/Dockerfile .
docker run --rm -p 5000:5000 st7ma784/moda-fastmoda:latest-cpu
```

### For Rancher/Kubernetes:

**1. Update image (rebuild and push):**
```bash
docker build -t st7ma784/moda-fastmoda:latest-cpu -f FastMODA/Dockerfile .
docker push st7ma784/moda-fastmoda:latest-cpu
```

**2. Helm deployment (CPU-only):**
```bash
helm upgrade moda helm/moda --values helm/moda/values.yaml \
  --set fastmoda.image.tag=latest-cpu \
  --set fastmoda.env.USE_GPU=auto
```

**3. Verify pod health:**
```bash
kubectl get pods -n moda | grep fastmoda
kubectl logs -n moda deployment/moda-fastmoda --tail=50
```

Should show:
- `Running` status (not CrashLoopBackOff)
- "Backend: CPU" in startup logs
- No crash/error messages

## Key Configuration Parameters

| Variable | Default | CPU Mode | GPU Mode | Notes |
|----------|---------|----------|----------|-------|
| `USE_GPU` | `auto` | CPU fallback | Uses CUDA if available | Recommended: `auto` |
| `FLASK_DEBUG` | `false` | `false` | `false` | Always false in Docker |
| `FLASK_ENV` | `production` | `production` | `production` | Always production |
| `PYTHONUNBUFFERED` | `1` | `1` | `1` | Ensures real-time logs |

## Monitoring

**Health check endpoint**: `GET /health`
- Returns: `{"status": "ok"}` with 200 status
- Called every 30 seconds by Kubernetes
- Used to detect failed containers

**Common success indicators**:
```
# Logs should show:
"FastMODA OPTIMIZED - Starting"
"Backend: CPU" (or "Backend: GPU (OPTIMIZED)")
"GET /health HTTP/1.1" 200  # Health checks passing
```

## If Issues Persist

1. **Check logs for import errors**:
   ```bash
   kubectl logs -n moda deployment/moda-fastmoda --previous
   ```

2. **Verify image is updated**:
   ```bash
   docker inspect st7ma784/moda-fastmoda:latest-cpu | grep RootFS
   kubectl describe pod -n moda <pod-name> | grep Image
   ```

3. **Test locally first**:
   ```bash
   docker run --rm st7ma784/moda-fastmoda:latest-cpu
   ```

4. **Check Rancher resource limits** (may kill slow startup):
   ```yaml
   resources:
     requests:
       cpu: 250m
       memory: 256Mi
     limits:
       cpu: 1
       memory: 1Gi
   ```

## Related Documentation

- [RANCHER_CRASHLOOP_FIX.md](RANCHER_CRASHLOOP_FIX.md) - Previous gunicorn fix
- [Dockerfile](FastMODA/Dockerfile) - Container image definition
- [helm/moda/values.yaml](helm/moda/values.yaml) - Kubernetes configuration

---
**Last Updated**: April 21, 2026
**Status**: Tested and verified stable ✅
