# FastMODA Rancher Debugging Quick Reference

**Problem**: Container crashes with `CrashLoopBackOff` before logs appear

## Quick Fixes

### 1. Check for Startup Crashes (silent failures)
```bash
# Run container locally to see actual error
docker run --rm st7ma784/moda-fastmoda:latest-cpu

# Expected: Should start without errors
# If crashes silently: you've found the issue
```

### 2. Common Cause: Unsafe Torch/GPU References
**Symptom**: Crashes during Python startup, before port binds
```python
# ❌ WRONG - crashes if torch not imported:
if USE_GPU:
    print(torch.cuda.get_device_name(0))

# ✅ CORRECT - safe fallback:
if USE_GPU and TORCH_AVAILABLE:
    try:
        print(torch.cuda.get_device_name(0))
    except:
        pass
```

### 3. Verify Health Check Works
```bash
# Health check should pass immediately:
curl http://localhost:5000/health
# Response: {"status": "ok"}

# If health check fails → container crashes
```

### 4. Check Startup Logs
```bash
# See what happens during startup
docker run --rm st7ma784/moda-fastmoda:latest-cpu 2>&1 | head -20

# Should show:
# "FastMODA OPTIMIZED - Starting"
# "Backend: CPU" or "Backend: GPU"
# "Listening at: http://0.0.0.0:5000"
```

## Kubernetes Troubleshooting

### Pod stuck in CrashLoopBackOff
```bash
# 1. Check pod status
kubectl get pods -n moda

# 2. Check logs (may be empty if crash is immediate)
kubectl logs -n moda deployment/moda-fastmoda --tail=100

# 3. Check previous container logs
kubectl logs -n moda deployment/moda-fastmoda --previous

# 4. Check pod events
kubectl describe pod -n moda <pod-name>

# 5. Check if image is correct
kubectl describe deployment -n moda moda-fastmoda | grep Image
```

### Health Check Failing
```bash
# Test manually
kubectl exec -it -n moda pod/moda-fastmoda-xxx -- \
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Should return 200 status code
```

### Resource Limits Too Tight
```yaml
# Current settings (CPU-only):
resources:
  requests:
    cpu: 250m      # May be too low for startup
    memory: 256Mi   # May be too low
  limits:
    cpu: 1000m     # 1 CPU
    memory: 1Gi    # 1 GB

# If container killed during startup:
# Increase requests.cpu to 500m or 1000m
```

## Version Check

### Verify FastMODA version has fix
```bash
# Look for this in app.py (line ~70):
if USE_GPU and TORCH_AVAILABLE:
    try:
        print(f"Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: Could not query GPU info: {e}")
```

### Verify Dockerfile uses gunicorn
```bash
# Last line should be:
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", ...]
```

## Prevention Checklist

Before deploying:
- [ ] Local test: `docker run --rm <image>` starts without crashing
- [ ] Health check returns 200: `curl http://localhost:5000/health`
- [ ] App logs show "Backend: CPU" or "Backend: GPU (OPTIMIZED)"
- [ ] No torch/CUDA references without TORCH_AVAILABLE check
- [ ] docker-compose.yml uses python healthcheck, not curl
- [ ] helm values.yaml has USE_GPU=auto

## Key Files

| File | Purpose | Last Updated |
|------|---------|--------------|
| [FastMODA/app.py](FastMODA/app.py) | Main app - torch safety fix at line 70 | Apr 21, 2026 |
| [FastMODA/Dockerfile](FastMODA/Dockerfile) | Uses gunicorn CMD | Mar 2026 |
| [FastMODA/docker-compose.yml](FastMODA/docker-compose.yml) | Dev setup - python healthcheck | Apr 21, 2026 |
| [helm/moda/values.yaml](helm/moda/values.yaml) | K8s deployment - USE_GPU=auto | Apr 21, 2026 |
| [FASTMODA_STABILITY_FIXES.md](FASTMODA_STABILITY_FIXES.md) | Detailed explanation | Apr 21, 2026 |

---
**Related Issues**: Immediate internal server error on login → See RANCHER_CRASHLOOP_FIX.md
