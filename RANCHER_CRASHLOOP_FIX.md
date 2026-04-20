# FastMODA Rancher CrashLoopBackOff - Fix Summary

## Problem Diagnosed
The FastMODA container was entering a `CrashLoopBackOff` state with `Terminated: Completed` status, indicating the process was exiting cleanly (exit code 0) rather than crashing with an error.

**Root Cause**: Flask running with `debug=True` in Docker/Kubernetes causes the reloader to fork processes. The parent process exits while the child handles requests, causing Kubernetes to detect the container as failed and restart it.

## Changes Made

### 1. **FastMODA/app.py** (Line 1617)
Updated the Flask application startup to conditionally disable debug mode:

```python
if __name__ == '__main__':
    # In Kubernetes/Docker, debug=True causes the reloader to fork and parent exits
    # Disable debug mode in production environments
    import sys
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    if debug_mode:
        print("Running in DEBUG mode (dev environment)")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    else:
        print("Running in PRODUCTION mode (Kubernetes/Docker environment)")
        # Use app.run without debug for production, or preferably use gunicorn
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

### 2. **FastMODA/requirements.txt**
Added `gunicorn` for production-grade WSGI serving:
```
gunicorn
```

### 3. **FastMODA/Dockerfile** (CMD instruction)
Updated to use gunicorn instead of Flask's dev server:

**Before:**
```dockerfile
CMD ["python", "-u", "app.py"]
```

**After:**
```dockerfile
ENV FLASK_DEBUG=false
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
```

### 4. **helm/moda/values.yaml**
Added explicit FLASK_DEBUG environment variable:

```yaml
env:
  FLASK_ENV: production
  FLASK_DEBUG: "false"
  PYTHONUNBUFFERED: "1"
```

## How It Works Now

1. **Production (Rancher/Kubernetes):**
   - `FLASK_DEBUG=false` (default)
   - Gunicorn WSGI server runs with 4 workers
   - Process stays alive and responds to health checks
   - Each worker can handle multiple requests concurrently

2. **Development (Local):**
   - Set `FLASK_DEBUG=true` to enable hot reload
   - Flask dev server runs with debug enabled
   - Reloader is disabled to prevent process forking issues

## Deployment Steps

1. Rebuild the Docker image:
   ```bash
   docker build -t st7ma784/moda-fastmoda:latest-cpu -f FastMODA/Dockerfile .
   ```

2. Push to registry:
   ```bash
   docker push st7ma784/moda-fastmoda:latest-cpu
   ```

3. Re-deploy to Rancher:
   ```bash
   helm upgrade moda helm/moda --values helm/moda/values.yaml
   ```

## Verification

After deployment, verify the pod is running:

```bash
kubectl get pods -n moda | grep fastmoda
```

Should show status `Running` (not `CrashLoopBackOff`).

Check logs for gunicorn startup message:
```bash
kubectl logs -n moda deployment/moda-fastmoda
```

Test health endpoint:
```bash
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000
curl http://localhost:5000/health
```

Should return: `{"status":"ok"}`

## Notes

- **Timeout**: Set to 120s to accommodate long-running signal analysis
- **Workers**: 4 workers should handle moderate traffic; adjust via environment variable if needed
- **Logging**: Both access and error logs are output to stdout for Kubernetes log aggregation
- **Health Checks**: Helm probes configured to check `/health` endpoint - this endpoint is now properly served by gunicorn
