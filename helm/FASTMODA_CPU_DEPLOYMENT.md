# FastMODA CPU API — Minimal Deployment Guide

## Default Configuration

The MODA Helm chart is now optimized for **minimal resource usage**, deploying only the **FastMODA API (CPU variant)**.

### What Gets Deployed by Default

✅ **FastMODA API** (CPU)
- Image: `st7ma784/moda-fastmoda:latest-cpu`
- Replicas: 1
- CPU Request: 250m
- Memory Request: 256Mi
- CPU Limit: 1
- Memory Limit: 1Gi
- Port: 5000/TCP

❌ **MODA Server** (MATLAB) — Disabled
- Requires 2-4 CPU and 4-8GB RAM
- Enable if you need backend MATLAB computation

❌ **Mobile UI** (Flutter web) — Disabled
- Optional web dashboard
- Can be enabled separately if needed

## Quick Install

```bash
# Minimal deployment (CPU only)
helm upgrade --install moda ./helm/moda -n moda --create-namespace
```

Verify deployment:

```bash
kubectl get all -n moda
kubectl get svc -n moda moda-fastmoda
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl http://localhost:5000/health
```

## Resource Usage

### CPU Variant (Default)
```
Total CPU Request: 250m    (0.25 CPU cores)
Total Memory Request: 256Mi  (0.25 GB)
```

This fits comfortably on:
- Single-node development clusters
- Rancher local clusters
- Raspberry Pi clusters (with arm64 support)
- Minimal cloud instances (1+ CPU, 512MB+ RAM)

### Example Cluster Sizes

| Cluster Type | CPU Available | RAM Available | Max Replicas |
|--------------|---------------|---------------|--------------|
| Laptop (minikube) | 2 | 4GB | 4 |
| Small cloud (t2.micro) | 1 | 1GB | 2 |
| Standard cloud (t3.small) | 2 | 2GB | 8 |
| Rancher local | 4+ | 8GB+ | 16+ |

## Enabling GPU (Optional)

If you have GPU nodes available:

```bash
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.image.tag=latest-gpu \
  --set fastmoda.gpu.enabled=true
```

### GPU Configuration

```yaml
fastmoda:
  image:
    tag: latest-gpu  # Switch from latest-cpu to latest-gpu
  gpu:
    enabled: true    # Enable GPU resource allocation
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: "2"
        memory: 2Gi
        nvidia.com/gpu: "1"  # Request 1 GPU
    nodeSelector:
      nvidia.com/gpu: "true"  # Require GPU nodes
```

GPU pods will:
- Request 1 NVIDIA GPU
- Be scheduled only on nodes with GPU labels
- Use up to 2 CPU and 2GB RAM
- Use GPU for tensor operations

## Scaling Configuration

### Add Replicas

```bash
# Run 3 replicas instead of 1
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.replicaCount=3
```

Total resources: 750m CPU, 768Mi RAM (3 × 250m, 256Mi)

### Enable Horizontal Pod Autoscaling

```bash
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.autoscaling.enabled=true \
  --set fastmoda.autoscaling.minReplicas=1 \
  --set fastmoda.autoscaling.maxReplicas=5 \
  --set fastmoda.autoscaling.targetCPUUtilizationPercentage=70
```

Auto-scales between 1-5 replicas based on CPU usage (70% threshold).

## API Endpoints

FastMODA exposes these endpoints:

```
GET  /health                - Health check
GET  /api/info              - API information
POST /api/analyze           - Run signal analysis
GET  /api/status/<task_id>  - Check task status
```

## Custom Resource Limits

Need different limits? Create a custom values file:

```yaml
# custom-values.yaml
fastmoda:
  replicaCount: 2
  image:
    tag: latest-cpu
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi
```

Deploy:

```bash
helm upgrade --install moda ./helm/moda -n moda -f custom-values.yaml
```

## Production Checklist

✅ CPU variant on minimal spec
✅ Single replica by default (can scale)
✅ Health checks configured
✅ Resource limits set
✅ Optional GPU support

For production:
- Enable autoscaling
- Set replica count based on load
- Use LoadBalancer or Ingress
- Configure TLS
- Monitor resource usage

## Troubleshooting

### Pod won't start

```bash
# Check pod status
kubectl describe pod -n moda <pod-name>

# Check logs
kubectl logs -n moda <pod-name>

# Check resource availability
kubectl describe nodes
```

### Out of memory

Increase limits:

```bash
helm upgrade moda ./helm/moda -n moda \
  --set fastmoda.resources.limits.memory=2Gi
```

### GPU not available

```bash
# Check if GPU nodes exist
kubectl get nodes -L nvidia.com/gpu

# If GPU nodes exist, ensure they're labeled:
kubectl label nodes <gpu-node> nvidia.com/gpu=true
```

## Next Steps

- Deploy with: `helm upgrade --install moda ./helm/moda -n moda --create-namespace`
- Access at: `localhost:5000` (with port-forward)
- Or expose via Ingress/LoadBalancer (see `RANCHER_DEPLOYMENT.md`)
- Monitor with: `kubectl get pods -n moda -w`

---

**Default Chart Configuration:**
- FastMODA API: ✅ Enabled
- MODA Server: ❌ Disabled (can be enabled if needed)
- Mobile UI: ❌ Disabled (can be enabled if needed)
- GPU Support: Optional (disabled by default)
- Minimum Cluster: 500m CPU, 512Mi RAM
- Recommended Cluster: 2+ CPU, 2GB+ RAM
