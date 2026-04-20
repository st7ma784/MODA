# MODA Helm Chart — Quick Install Reference

## One-Liner Deployment Commands

### 🏠 Rancher Local Cluster (Default)
```bash
helm upgrade --install moda ./helm/moda -n moda --create-namespace
```

### ☁️ AWS EKS (Network Load Balancer)
```bash
helm upgrade --install moda ./helm/moda \
  -n moda --create-namespace \
  --set fastmoda.service.type=LoadBalancer \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb" \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-cross-zone-load-balancing-enabled"="true"
```

Get LoadBalancer endpoint:
```bash
kubectl get svc -n moda moda-fastmoda -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

### 🔵 Azure AKS
```bash
helm upgrade --install moda ./helm/moda \
  -n moda --create-namespace \
  --set fastmoda.service.type=LoadBalancer
```

Get LoadBalancer IP:
```bash
kubectl get svc -n moda moda-fastmoda -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### 🔵 Google Cloud GKE
```bash
helm upgrade --install moda ./helm/moda \
  -n moda --create-namespace \
  --set ingress.className=gce \
  --set ingress.annotations."kubernetes\.io/ingress\.global-static-ip-name"="moda-ip"
```

### 🏗️ On-Premises (MetalLB)
```bash
# First, install MetalLB (if not already installed)
helm install metallb metallb/metallb -n metallb-system --create-namespace

# Then install MODA
helm upgrade --install moda ./helm/moda \
  -n moda --create-namespace \
  --set fastmoda.service.type=LoadBalancer \
  --set mobile.service.type=LoadBalancer
```

Check allocated IPs:
```bash
kubectl get svc -n moda
```

## Configuration Options

### Port Customization
```bash
# Change FastMODA API port (default: 5000)
--set fastmoda.service.port=8080

# Change Mobile UI port (default: 80)
--set mobile.service.port=8000

# Change Server port (default: 6789)
--set server.service.port=9999
```

### Scaling
```bash
# Set replicas for FastMODA
--set fastmoda.replicaCount=3

# Enable auto-scaling
--set fastmoda.autoscaling.enabled=true \
--set fastmoda.autoscaling.minReplicas=2 \
--set fastmoda.autoscaling.maxReplicas=10
```

### Custom Domain & TLS
```bash
# Configure domain
--set ingress.hosts[0].host=moda.mydomain.com \

# Enable TLS with cert-manager
--set ingress.annotations."cert-manager\.io/cluster-issuer"="letsencrypt-prod" \
--set ingress.tls[0].secretName="moda-tls" \
--set ingress.tls[0].hosts[0]="moda.mydomain.com"
```

### GPU Support
```bash
# Use GPU image
--set fastmoda.image.tag=latest-gpu \
--set fastmoda.gpu.enabled=true \
--set fastmoda.gpu.resources.limits."nvidia\.com/gpu"="1"
```

## Complete Multi-Argument Example

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.replicaCount=3 \
  --set fastmoda.service.type=LoadBalancer \
  --set fastmoda.autoscaling.enabled=true \
  --set fastmoda.autoscaling.minReplicas=2 \
  --set fastmoda.autoscaling.maxReplicas=8 \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=moda.example.com \
  --set ingress.annotations."cert-manager\.io/cluster-issuer"="letsencrypt-prod" \
  --set ingress.tls[0].secretName="moda-tls" \
  --set ingress.tls[0].hosts[0]="moda.example.com"
```

## Using Custom Values File

```bash
# Create custom-values.yaml with your settings
helm upgrade --install moda ./helm/moda \
  -n moda \
  --create-namespace \
  -f custom-values.yaml

# Or combine with CLI overrides
helm upgrade --install moda ./helm/moda \
  -n moda \
  --create-namespace \
  -f custom-values.yaml \
  --set fastmoda.replicaCount=5
```

## Verification Commands

```bash
# Check deployment status
kubectl get all -n moda

# Check services and endpoints
kubectl get svc,endpoints -n moda

# Check ingress
kubectl get ingress -n moda
kubectl describe ingress -n moda moda

# Check pod logs
kubectl logs -n moda -l app.kubernetes.io/component=fastmoda --tail=100

# Port forward for local access
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000

# Test health endpoint
curl http://localhost:5000/health
```

## Troubleshooting

### Check if services are healthy
```bash
kubectl get svc -n moda
# All should show ClusterIP or EXTERNAL-IP
```

### Check if pods are running
```bash
kubectl get pods -n moda
# All should show RUNNING/READY
```

### View pod logs
```bash
kubectl logs -n moda <pod-name>
```

### Verify port forwarding
```bash
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl -I http://localhost:5000/health
```

### Check service endpoints
```bash
kubectl get endpoints -n moda
# Should show IP:port for each service
```

## Uninstall

```bash
# Remove MODA release
helm uninstall moda -n moda

# Delete namespace
kubectl delete namespace moda

# Keep namespace (optional)
helm uninstall moda -n moda --keep-history
```

## Upgrade

```bash
# Update to latest version
helm upgrade moda ./helm/moda -n moda

# Upgrade with new values
helm upgrade moda ./helm/moda \
  -n moda \
  -f new-values.yaml
```

## Port Reference

| Service | Default | Type | Notes |
|---------|---------|------|-------|
| FastMODA API | 5000 | HTTP | REST API, WebSockets |
| MODA Server | 6789 | gRPC | MATLAB computation |
| Mobile UI | 80 | HTTP | Web frontend |
| Ingress | 80/443 | HTTP(S) | External gateway |

All ports are configurable via Helm values.

---

For detailed documentation, see:
- `RANCHER_DEPLOYMENT.md` - Comprehensive setup guide
- `PORT_CONFIGURATION.md` - Port mapping details
- `values-rancher-examples.yaml` - Configuration examples
