# MODA Helm Chart — Rancher Deployment Guide

## Overview

This guide explains how to deploy MODA on Rancher with proper ingress configuration and LoadBalancer support across different cloud providers and on-premises environments.

## Rancher Compliance Features

The MODA Helm chart now includes:

✅ **Rancher UI Integration**
- Proper labels for Rancher dashboard visibility
- Rancher ingress annotations
- Namespace metadata support
- Managed-by labels

✅ **Ingress Configuration**
- Port number specification (5000 for API, 6789 for server, 80 for mobile UI)
- Multiple ingress class support (nginx, traefik, local-path)
- TLS/HTTPS configuration
- Path-based routing with proper port mapping

✅ **LoadBalancer Support**
- Service type configuration (ClusterIP, LoadBalancer)
- Cloud provider-specific annotations
- External access control
- Proper health checks and probes

✅ **Kubernetes Best Practices**
- Proper namespace usage
- Component labels
- Version tracking
- Health checks configured
- Resource limits and requests

## Quick Start: Rancher Local Cluster

```bash
# 1. Add Helm repository (if needed)
helm repo add moda https://github.com/st7ma784/MODA

# 2. Install with default Rancher nginx-ingress
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --values ./helm/moda/values.yaml

# 3. Verify deployment
kubectl get all -n moda
kubectl get ingress -n moda

# 4. Access via Rancher UI
# Navigate to: moda namespace → Services → moda-fastmoda
```

## Service Configuration

### Default Service Types

```yaml
Services default to ClusterIP (internal access only):
- FastMODA API:    5000/TCP
- MODA Server:     6789/TCP  
- Mobile UI:       80/TCP
```

### Enable LoadBalancer for External Access

Edit `values.yaml`:

```yaml
fastmoda:
  service:
    type: LoadBalancer  # Changed from ClusterIP
    port: 5000

mobile:
  service:
    type: LoadBalancer  # Changed from ClusterIP
    port: 80
```

Then upgrade:

```bash
helm upgrade moda ./helm/moda \
  --namespace moda \
  -f values.yaml
```

Check external IP:

```bash
kubectl get svc -n moda
# Output: moda-fastmoda should show EXTERNAL-IP
```

## Ingress Configuration

### Example 1: Basic Rancher Ingress

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    kubernetes.io/ingress.class: nginx
    rancher.io/ingress: "true"
  hosts:
    - host: moda.example.com
      paths:
        - path: /api
          pathType: Prefix
          backend: fastmoda
        - path: /
          pathType: Prefix
          backend: mobile
```

**Port mapping:**
- `/api` → fastmoda:5000
- `/` → mobile:80

### Example 2: With TLS/HTTPS

```yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: moda.example.com
      paths:
        - path: /api
          backend: fastmoda
        - path: /
          backend: mobile
  tls:
    - secretName: moda-tls
      hosts:
        - moda.example.com
```

### Example 3: With Long Timeouts (for MATLAB processing)

```yaml
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
```

## Cloud Provider Setup

### AWS EKS with Network Load Balancer

**Install:**

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb" \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-cross-zone-load-balancing-enabled"="true"
```

**Get LoadBalancer DNS:**

```bash
kubectl get svc -n moda moda-fastmoda
# EXTERNAL-IP: moda-fastmoda-<id>.elb.amazonaws.com
```

### Azure AKS

**Install:**

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer
```

**Get Public IP:**

```bash
kubectl get svc -n moda
# EXTERNAL-IP: <public-ip>
```

### Google Cloud GKE

**Install:**

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --values ./helm/moda/values.yaml
```

**Configure Cloud Load Balancer:**

```bash
gcloud compute addresses create moda-ip --global
gcloud compute backend-services create moda-backend --global
```

### On-Premises with MetalLB

**Prerequisites:**

```bash
# Install MetalLB
helm install metallb metallb/metallb \
  --namespace metallb-system \
  --create-namespace
```

**Configure IP pool:**

```yaml
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: default
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.240-192.168.1.250
```

**Install MODA:**

```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer
```

## Port Reference

| Service | Default Port | Type | Purpose |
|---------|-------------|------|---------|
| FastMODA API | 5000 | HTTP | REST API, WebSocket |
| MODA Server | 6789 | gRPC/TCP | MATLAB computation |
| Mobile UI | 80 | HTTP | Web frontend |
| Ingress | 80/443 | HTTP/HTTPS | External gateway |

All ports are customizable via `values.yaml`.

## Verification

### Check Deployment Status

```bash
# Services
kubectl get svc -n moda
kubectl describe svc moda-fastmoda -n moda

# Ingress
kubectl get ingress -n moda
kubectl describe ingress moda -n moda

# Pods
kubectl get pods -n moda
kubectl logs -n moda deployment/moda-fastmoda
```

### Test Endpoints

```bash
# Via kubectl port-forward
kubectl port-forward -n moda svc/moda-fastmoda 5000:5000 &
curl http://localhost:5000/health

# Via LoadBalancer (if configured)
EXTERNAL_IP=$(kubectl get svc -n moda moda-fastmoda -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP:5000/health

# Via Ingress (requires DNS configured)
curl https://moda.example.com/api/health
```

## Troubleshooting

### Ingress not getting IP

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress status
kubectl describe ingress moda -n moda
```

### LoadBalancer stuck in Pending

```bash
# Check if provider supports LoadBalancer
kubectl get nodes -o wide

# For MetalLB, ensure IP pool is configured
kubectl get ipaddresspools -n metallb-system
```

### Service unreachable

```bash
# Check endpoints
kubectl get endpoints -n moda

# Check network policies
kubectl get networkpolicies -n moda

# Check pod logs
kubectl logs -n moda $(kubectl get pods -n moda -l app.kubernetes.io/component=fastmoda -o jsonpath='{.items[0].metadata.name}')
```

## Advanced: Custom Values File

Create `custom-values.yaml`:

```yaml
fastmoda:
  enabled: true
  image:
    repository: st7ma784/moda-fastmoda
    tag: latest-cpu
  replicaCount: 3
  service:
    type: LoadBalancer
    port: 5000
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: moda.company.com
      paths:
        - path: /api
          backend: fastmoda
        - path: /
          backend: mobile
  tls:
    - secretName: moda-tls
      hosts:
        - moda.company.com
```

Deploy:

```bash
helm upgrade --install moda ./helm/moda -f custom-values.yaml
```

## Cleanup

```bash
# Remove MODA deployment
helm uninstall moda -n moda

# Remove namespace
kubectl delete namespace moda
```

## Support

For issues or questions:
- Check logs: `kubectl logs -n moda <pod-name>`
- Rancher UI: Navigate to moda namespace for status
- GitHub Issues: https://github.com/st7ma784/MODA/issues
