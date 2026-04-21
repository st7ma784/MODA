# FastMODA Health Checks & Ingress Verification

**Date:** 2026-04-21  
**Configuration:** Nginx ingress at `/moda` path with health checks  
**Status:** ✅ Configured

---

## Configuration Summary

### Health Checks (Kubernetes Probes)

**Liveness Probe** — Detects crashed containers
```yaml
httpGet:
  path: /health
  port: http          # Named port (5000)
initialDelaySeconds: 30   # Wait 30s before first check
periodSeconds: 15         # Check every 15s
timeoutSeconds: 5         # Wait 5s for response
```

**Readiness Probe** — Detects unhealthy containers
```yaml
httpGet:
  path: /health
  port: http
initialDelaySeconds: 15   # Wait 15s before first check
periodSeconds: 5          # Check every 5s (faster feedback)
timeoutSeconds: 5         # Wait 5s for response
```

### Ingress Configuration

**Path:** `/moda(/|$)(.*)`  
**Backend Service:** `moda-fastmoda` on port 5000  
**Ingress Class:** nginx  
**Rewrite Rule:** `/$2` (strips `/moda` prefix before forwarding)

**Nginx Annotations:**
```yaml
nginx.ingress.kubernetes.io/rewrite-target: /$2      # Rewrite /moda/* to /*
nginx.ingress.kubernetes.io/use-regex: "true"        # Enable regex paths
nginx.ingress.kubernetes.io/proxy-body-size: "50m"   # 50MB upload limit
nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"  # Long timeouts
nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
```

---

## Verification Steps

### 1. Check Health Endpoint (Direct)

**Via port-forward:**
```bash
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl -v http://localhost:5000/health
```

**Expected Response:**
```json
{"status":"ok"}
```

### 2. Check Health Probes in Kubernetes

**View probe status:**
```bash
kubectl get pod -n moda -o wide
kubectl describe pod -n moda <pod-name>
```

**Look for:**
```
Liveness probe: http-get on http://10.x.x.x:5000/health delay=30s timeout=5s
Readiness probe: http-get on http://10.x.x.x:5000/health delay=15s timeout=5s
```

**Probe should be Green ✅:**
```
State:          Running (started 5m ago)
Ready:          True
Restart Count:  0
```

### 3. Test Ingress Routing

**Check ingress resource:**
```bash
kubectl get ingress -n moda
kubectl describe ingress -n moda moda
```

**Should show:**
```
Rules:
  Host  Path  Backends
  ----  ----  --------
        /moda(/|$)(.*) -> moda-fastmoda:5000 (10.x.x.x:5000)
```

### 4. Test Ingress Access (In-Cluster)

If cluster has ingress controller:
```bash
# Find ingress IP/domain
INGRESS_IP=$(kubectl get ingress -n moda moda -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test health via ingress
curl -v http://$INGRESS_IP/moda/health
```

**Expected:** Path is rewritten from `/moda/health` → `/health` before reaching app

### 5. Test API Endpoints via Ingress

```bash
# Health check
curl http://$INGRESS_IP/moda/health

# GPU info
curl http://$INGRESS_IP/moda/api/gpu-info

# Home page
curl http://$INGRESS_IP/moda/
```

---

## Debugging

### Issue: Health check failing

**Check probe logs:**
```bash
kubectl logs -n moda <pod-name> | tail -20
```

**Check directly:**
```bash
kubectl exec -n moda <pod-name> -- \
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
```

### Issue: Ingress not routing to /moda

**Check ingress configuration:**
```bash
kubectl get ingress -n moda moda -o yaml | grep -A 10 "rules:"
```

**Verify rewrite rule:**
```bash
# The path must be: /moda(/|$)(.*) 
# The rewrite target must be: /$2
```

### Issue: Port mismatch

**Verify service port:**
```bash
kubectl get svc -n moda moda-fastmoda
# Should show: 5000/TCP
```

**Verify deployment port:**
```bash
kubectl get pod -n moda <pod-name> -o yaml | grep -A 3 "ports:"
# Should show: containerPort: 5000, name: http
```

---

## API Endpoints

After deployment, access via ingress at:

```
http://your-ingress-host/moda/health         GET   - Health check
http://your-ingress-host/moda/api/gpu-info   GET   - GPU status
http://your-ingress-host/moda/                GET   - Web UI
http://your-ingress-host/moda/analyze        POST  - Process signal
http://your-ingress-host/moda/modwt          GET   - MODWT page
```

All paths are automatically rewritten from `/moda/*` → `/*` by nginx.

---

## Health Check Behavior

### Lifecycle

```
Pod Created
    ↓
Wait 15s (readinessProbe initialDelaySeconds)
    ↓
Is /health responding? → No → Pod "Not Ready" ❌
                    ↓
                    Yes → Pod "Ready" ✅
    ↓
Wait 30s (livenessProbe initialDelaySeconds)
    ↓
Is /health responding? → No → Restart Pod 🔄
                    ↓
                    Yes → Keep Pod Running ✅
```

### Recovery

If health check fails:
1. **Readiness Probe Fails** → Pod removed from service endpoints (no traffic)
2. **Liveness Probe Fails** → kubelet restarts the Pod
3. **Repeated Failures** → Pod in CrashLoopBackOff (Rancher shows error)

---

## Test Commands

### Complete Test Suite

```bash
# 1. Check pod is running
kubectl get pods -n moda

# 2. Check service is ready
kubectl get svc -n moda moda-fastmoda

# 3. Check ingress is configured
kubectl get ingress -n moda

# 4. Port-forward and test health
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
sleep 2
curl http://localhost:5000/health
kill %1

# 5. Check probe status
kubectl describe pod -n moda $(kubectl get pods -n moda -o name | head -1)

# 6. View logs for errors
kubectl logs -n moda $(kubectl get pods -n moda -o name | head -1)
```

---

## Success Indicators ✅

**Healthy deployment shows:**

1. **Pod Status:**
   ```
   NAME    READY   STATUS    RESTARTS   AGE
   moda... 1/1     Running   0          5m
   ```

2. **Service Endpoints:**
   ```
   kubectl get endpoints -n moda moda-fastmoda
   # Should show IP:5000
   ```

3. **Ingress Status:**
   ```
   kubectl get ingress -n moda -o wide
   # Should show IP or LoadBalancer address
   ```

4. **Health Check Response:**
   ```bash
   curl http://localhost:5000/health
   {"status":"ok"}
   ```

5. **Pod Probe Status (describe):**
   ```
   Liveness:  http-get http://10.x.x.x:5000/health delay=30s timeout=5s period=15s
   Readiness: http-get http://10.x.x.x:5000/health delay=15s timeout=5s period=5s
   ```

---

## Helm Deployment

### Deploy with Health Checks & Ingress

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace
```

### Verify Deployment

```bash
# Watch pod startup (should become Ready within 60s)
kubectl get pods -n moda -w

# Check ingress is configured
kubectl get ingress -n moda

# Test health endpoint
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl http://localhost:5000/health
```

---

## Configuration Files

**Values:** `helm/moda/values.yaml`
- Health probe configuration
- Ingress path: `/moda(/|$)(.*)`
- Service port: 5000

**Deployment:** `helm/moda/templates/fastmoda-deployment.yaml`
- Port named: `http`
- Probes configured

**Ingress:** `helm/moda/templates/ingress.yaml`
- Routing rules
- Backend service reference

**App:** `FastMODA/app.py`
- `@app.route('/health')` → returns `{"status":"ok"}`

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Health check failed (CrashLoopBackOff) | App not responding | Check logs: `kubectl logs -n moda <pod>` |
| Pod stuck "Not Ready" | Readiness probe failing | Wait 15s, then check: `curl localhost:5000/health` |
| Ingress not routing | Wrong path pattern | Verify path is `/moda(/\|$)(.*)` with regex enabled |
| 404 from ingress | Rewrite rule not working | Check: `nginx.ingress.kubernetes.io/rewrite-target: /$2` |
| Port mismatch | Named port vs number | Probes must use `port: http` (named), not `port: 5000` |

---

**Status:** ✅ Ready for Rancher Deployment  
**Next:** Deploy with `helm upgrade --install moda ./helm/moda -n moda --create-namespace`
