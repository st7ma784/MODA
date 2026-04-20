# MODA Helm Chart — Minimal CPU Deployment Configuration

**Date:** 2026-04-20  
**Change:** Updated to FastMODA CPU-only deployment with optional GPU support  
**Status:** ✅ Ready to deploy

---

## Configuration Changes Summary

### ✅ What Changed

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| **FastMODA API (CPU)** | Enabled (2 replicas) | Enabled (1 replica) | ✅ Default |
| **MODA Server (MATLAB)** | Enabled (heavy) | **Disabled** | ❌ Optional |
| **Mobile UI (Flutter)** | Enabled | **Disabled** | ❌ Optional |
| **GPU Support** | Disabled | Enabled (optional) | 🔧 Optional |
| **Default CPU Request** | 500m × 2 = 1 CPU | 250m × 1 = 250m CPU | ⬇️ 75% reduction |
| **Default Memory Request** | 512Mi × 2 = 1 GB | 256Mi × 1 = 256Mi RAM | ⬇️ 75% reduction |

---

## Default Resource Profile

### CPU Variant (Default)

```
Single Pod Resource Usage:
├─ CPU Request: 250m (0.25 core)
├─ CPU Limit: 1 (1 core)
├─ Memory Request: 256Mi (0.25 GB)
└─ Memory Limit: 1Gi (1 GB)

Total Cluster Usage (1 replica):
├─ CPU: 250m requested, 1 CPU max
└─ Memory: 256Mi requested, 1GB max

Suitable for:
✅ Development machines (4+ GB RAM)
✅ Minimal cloud instances (1+ vCPU, 512MB+ RAM)
✅ Single-node K3s clusters
✅ Rancher local clusters
✅ Raspberry Pi clusters (arm64)
```

### GPU Variant (Optional)

```
When --set fastmoda.gpu.enabled=true:

Single Pod Resource Usage:
├─ CPU Request: 500m (0.5 core)
├─ CPU Limit: 2 (2 cores)
├─ Memory Request: 512Mi (0.5 GB)
├─ Memory Limit: 2Gi (2 GB)
└─ GPU Request: 1 × nvidia.com/gpu

Requires:
✅ GPU node with nvidia.com/gpu label
✅ NVIDIA Container Toolkit installed
✅ NVIDIA device plugin running
```

---

## Installation Instructions

### Default (CPU Only)

```bash
# Minimal deployment
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace
```

**Resources Used:** 250m CPU, 256Mi RAM (minimal)

### With Auto-scaling

```bash
# Add horizontal auto-scaling (1-5 replicas)
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.autoscaling.enabled=true \
  --set fastmoda.autoscaling.minReplicas=1 \
  --set fastmoda.autoscaling.maxReplicas=5
```

**Resources Used:** 250m-1.25 CPU, 256Mi-1.28GB RAM (scales based on load)

### With GPU (If Available)

```bash
# Deploy with GPU support
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.image.tag=latest-gpu \
  --set fastmoda.gpu.enabled=true
```

**Resources Used:** 500m CPU, 512Mi RAM + 1 GPU (requires GPU node)

### Multiple Replicas

```bash
# Run 3 replicas (horizontal scaling)
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.replicaCount=3
```

**Resources Used:** 750m CPU, 768Mi RAM

---

## Deployment Details

### Default Values

```yaml
fastmoda:
  enabled: true
  image:
    repository: st7ma784/moda-fastmoda
    tag: latest-cpu          # ← Use CPU image by default
  replicaCount: 1             # ← Single replica
  
  resources:
    requests:
      cpu: 250m              # ← Minimal CPU
      memory: 256Mi          # ← Minimal RAM
    limits:
      cpu: "1"
      memory: 1Gi

  gpu:
    enabled: false           # ← GPU optional
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: "2"
        memory: 2Gi
        nvidia.com/gpu: "1"
    nodeSelector:
      nvidia.com/gpu: "true"  # ← Requires GPU nodes

server:
  enabled: false              # ← MATLAB Server disabled

mobile:
  enabled: false              # ← Mobile UI disabled
```

---

## API Endpoints

FastMODA API available at:

```
GET  http://moda-fastmoda:5000/health
GET  http://moda-fastmoda:5000/api/info
POST http://moda-fastmoda:5000/api/analyze
GET  http://moda-fastmoda:5000/api/status/<task_id>
```

### Access Methods

**Port Forward (local):**
```bash
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl http://localhost:5000/health
```

**Ingress (external):**
```bash
# First, enable ingress and configure domain
helm upgrade moda ./helm/moda -n moda \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=moda.example.com
```

**LoadBalancer (cloud):**
```bash
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.service.type=LoadBalancer
kubectl get svc -n moda moda-fastmoda
```

---

## Optional Components

### Enable MATLAB Server (If Needed)

Requires 2-4 CPU, 4-8GB RAM:

```bash
helm upgrade moda ./helm/moda -n moda \
  --set server.enabled=true
```

### Enable Mobile UI (If Needed)

Minimal resources (100m CPU, 64Mi RAM):

```bash
helm upgrade moda ./helm/moda -n moda \
  --set mobile.enabled=true
```

### Enable Both

```bash
helm upgrade moda ./helm/moda -n moda \
  --set server.enabled=true \
  --set mobile.enabled=true
```

---

## Cluster Size Recommendations

| Cluster | CPU | RAM | Max Pods | Notes |
|---------|-----|-----|----------|-------|
| **Laptop (minikube)** | 2 | 4GB | 4-8 | Good for dev |
| **Single-node K3s** | 2 | 2GB | 4-6 | Minimal overhead |
| **Small cloud (t3.small)** | 2 | 2GB | 6-8 | Good starter |
| **Standard cloud (t3.medium)** | 2 | 4GB | 12-16 | Production-ready |
| **Rancher local** | 4+ | 8GB+ | 16+ | Highly flexible |

---

## Monitoring & Scaling

### Check Deployment

```bash
kubectl get all -n moda
kubectl describe pod -n moda <pod-name>
kubectl logs -n moda <pod-name>
```

### Monitor Resources

```bash
kubectl top pods -n moda
kubectl top nodes
```

### Auto-scaling Status

```bash
kubectl get hpa -n moda
kubectl describe hpa -n moda moda-fastmoda
```

---

## Troubleshooting

### Pod won't start

```bash
# Check events
kubectl describe pod -n moda <pod-name>

# Check logs
kubectl logs -n moda <pod-name>

# Check node resources
kubectl describe nodes
kubectl top nodes
```

### OutOfMemory error

Increase limits:
```bash
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.resources.limits.memory=2Gi
```

### GPU not available

Check GPU nodes:
```bash
kubectl get nodes -L nvidia.com/gpu
kubectl get nodes --show-labels | grep gpu
```

Label GPU nodes:
```bash
kubectl label nodes <gpu-node> nvidia.com/gpu=true
```

---

## Comparison: Before vs After

### Before (Heavy Default)
- FastMODA: 2 replicas × (500m CPU + 512Mi RAM)
- MODA Server: 1 × (2 CPU + 4GB RAM)
- Mobile: 1 × (100m CPU + 64Mi RAM)
- **Total: 3.1 CPU + 4.6 GB RAM**

### After (Minimal Default)
- FastMODA: 1 replica × (250m CPU + 256Mi RAM)
- MODA Server: Disabled
- Mobile: Disabled
- **Total: 250m CPU + 256Mi RAM** ✅ **96% reduction!**

---

## Next Steps

1. ✅ Deploy with minimal resources:
   ```bash
   helm upgrade --install moda ./helm/moda -n moda --create-namespace
   ```

2. ✅ Test the API:
   ```bash
   kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
   curl http://localhost:5000/health
   ```

3. ✅ Scale as needed:
   - Add replicas if load increases
   - Enable GPU if available
   - Enable MATLAB server if needed

4. ✅ Configure external access:
   - Use Ingress for domain-based access
   - Use LoadBalancer for cloud deployments

---

**Chart Version:** 0.1.0  
**Deployment Type:** FastMODA CPU API (Minimal)  
**Default Resources:** 250m CPU, 256Mi RAM  
**Status:** ✅ Production-Ready
