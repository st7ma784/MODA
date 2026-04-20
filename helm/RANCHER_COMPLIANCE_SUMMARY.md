# MODA Helm Chart — Rancher Compliance Update Summary

**Date:** 2026-04-20  
**Version:** 0.1.0  
**Status:** ✅ Rancher-Ready

## Changes Made

### 1. **Chart Metadata** (`Chart.yaml`)

✅ **Added:**
- `kubeVersion: ">=1.20.0"` - Kubernetes compatibility requirement
- `icon:` - Chart icon URL for Rancher UI
- Email contact for maintainers
- Category annotations for discovery
- License information

**Benefits:**
- Rancher Dashboard shows proper chart information
- Better visibility in Rancher Marketplace
- Kubernetes version compatibility enforcement

---

### 2. **Default Values** (`values.yaml`)

✅ **Added Global Rancher Labels:**
```yaml
labels:
  app.kubernetes.io/managed-by: Helm
  chart: moda
  managed: "true"
```

✅ **Updated Service Configurations** (FastMODA, Server, Mobile):
- Added `annotations: {}` field for cloud provider annotations
- Added comments showing AWS/Azure annotation examples
- All services support both `ClusterIP` and `LoadBalancer` types

✅ **Enhanced Ingress Configuration:**
- Added Rancher-specific annotations:
  - `rancher.io/ingress: "true"`
  - `kubernetes.io/ingress.class: nginx`
- Added nginx-specific proxy timeout settings (600s for MATLAB processing)
- Added explicit path routing for `/api`, `/api/v1`, `/health`
- Improved TLS/HTTPS configuration comments
- Added certificate-manager integration examples

**Port Mapping (Explicit):**
| Path | Service | Port | Protocol |
|------|---------|------|----------|
| `/api` | fastmoda | 5000 | HTTP |
| `/api/v1` | fastmoda | 5000 | HTTP |
| `/health` | fastmoda | 5000 | HTTP |
| `/` | mobile | 80 | HTTP |

---

### 3. **Ingress Template** (`templates/ingress.yaml`)

✅ **Added:**
- `namespace: {{ .Release.Namespace }}` - Proper namespace support
- Component labels for better Rancher visibility:
  - `app.kubernetes.io/component: ingress`
  - `app.kubernetes.io/version: {{ .Chart.AppVersion }}`

✅ **Benefits:**
- Rancher Dashboard can track ingress by component
- Better filtering and organization in UI
- Clear version tracking

---

### 4. **Service Templates** (`templates/services.yaml`)

✅ **Updated All Three Services (FastMODA, Server, Mobile):**

**Added to each service:**
```yaml
namespace: {{ .Release.Namespace }}
labels:
  app.kubernetes.io/component: <service-name>
  app.kubernetes.io/port: {{ .Values.<service>.service.port }}
annotations:
  {{- with .Values.<service>.service.annotations }}
  {{- toYaml . | nindent 4 }}
  {{- end }}
```

✅ **Benefits:**
- Cloud provider annotations now supported (AWS, Azure, GCP, MetalLB)
- Port tracking in labels for Rancher UI
- Proper namespace scoping
- Explicit component identification

**Port Information (Now Labeled):**
- FastMODA: 5000/TCP (HTTP)
- Server: 6789/TCP (gRPC/MATLAB)
- Mobile: 80/TCP (HTTP)

---

## New Documentation Files

### 1. **`helm/RANCHER_DEPLOYMENT.md`** (Comprehensive Guide)
Covers:
- ✅ Rancher compliance features overview
- ✅ Quick start for Rancher local clusters
- ✅ Service type configuration (ClusterIP vs LoadBalancer)
- ✅ Ingress setup with examples
- ✅ Cloud provider integration (AWS EKS, Azure AKS, GKE, MetalLB)
- ✅ Port reference table
- ✅ Verification and troubleshooting steps
- ✅ Advanced configurations and custom values

### 2. **`helm/PORT_CONFIGURATION.md`** (Technical Reference)
Detailed documentation of:
- ✅ Port mapping architecture diagram
- ✅ Service-to-pod port forwarding
- ✅ ClusterIP vs LoadBalancer configuration
- ✅ Ingress rules and path-based routing
- ✅ TLS/HTTPS setup options
- ✅ Rancher-specific annotations
- ✅ Troubleshooting guide for port issues
- ✅ Best practices

### 3. **`helm/moda/values-rancher-examples.yaml`** (Ready-to-Use Examples)
Contains configurations for:
- ✅ Rancher Local Cluster with nginx-ingress
- ✅ AWS EKS with Network Load Balancer
- ✅ Azure AKS with Azure Load Balancer
- ✅ Google GKE with Cloud Load Balancer
- ✅ On-Premises with MetalLB
- ✅ Rancher Downstream with Traefik
- ✅ Quick reference for port numbers

---

## Rancher Compliance Checklist

| Feature | Status | Details |
|---------|--------|---------|
| Kubernetes Compatibility | ✅ | kubeVersion >= 1.20.0 |
| Chart Metadata | ✅ | Icon, description, maintainers, category |
| Namespace Support | ✅ | Proper namespace in all manifests |
| Component Labels | ✅ | app.kubernetes.io/component on all resources |
| Version Tracking | ✅ | app.kubernetes.io/version in labels |
| Rancher Annotations | ✅ | rancher.io/ingress: "true" |
| Ingress Support | ✅ | Multiple ingress classes (nginx, traefik, local-path) |
| LoadBalancer Support | ✅ | Type: LoadBalancer with cloud annotations |
| Port Configuration | ✅ | Explicit port numbers, all configurable |
| TLS/HTTPS | ✅ | cert-manager integration, manual certs |
| Service Types | ✅ | ClusterIP and LoadBalancer options |
| Health Checks | ✅ | Liveness and readiness probes configured |
| Resource Limits | ✅ | CPU/memory requests and limits set |
| Security Context | ✅ | Pod security context support |

---

## Quick Start Commands

### Rancher Local Cluster
```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace
```

### AWS EKS with NLB
```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb"
```

### On-Premises with MetalLB
```bash
helm upgrade --install moda ./helm/moda \
  --namespace moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer
```

---

## Migration Guide

If upgrading from previous chart version:

```bash
# Backup current deployment
helm get values moda -n moda > moda-values-backup.yaml

# Upgrade to new version
helm upgrade moda ./helm/moda \
  --namespace moda \
  --values moda-values-backup.yaml

# Verify upgrade
kubectl get all -n moda
kubectl describe ingress -n moda
```

No breaking changes - all new features are backward compatible.

---

## Testing

### Verify Ingress
```bash
kubectl describe ingress -n moda moda
# Should show backends: moda-fastmoda:5000, moda-mobile:80
```

### Verify Port Mapping
```bash
kubectl get svc -n moda -o wide
# FastMODA: 5000/TCP
# Server: 6789/TCP
# Mobile: 80/TCP
```

### Check Rancher Labels
```bash
kubectl get all -n moda --show-labels
# Should include: app.kubernetes.io/component, app.kubernetes.io/managed-by
```

---

## Support & Documentation

| Document | Purpose |
|----------|---------|
| `RANCHER_DEPLOYMENT.md` | Comprehensive Rancher setup guide |
| `PORT_CONFIGURATION.md` | Detailed port/ingress technical reference |
| `values-rancher-examples.yaml` | Ready-to-use configuration examples |
| `README.md` (main) | General MODA information |
| `API.md` (FastMODA) | FastMODA REST API reference |

---

## Next Steps

1. ✅ Review `RANCHER_DEPLOYMENT.md` for your environment
2. ✅ Select appropriate `values-rancher-examples.yaml` configuration
3. ✅ Deploy using `helm upgrade --install` command
4. ✅ Verify using troubleshooting steps in documentation
5. ✅ Configure custom domain and TLS (optional)

---

## Backward Compatibility

✅ All changes are **backward compatible**
- Existing deployments continue to work
- No required value changes
- All new features are optional

---

**Chart Version:** 0.1.0  
**App Version:** 2.0.0  
**Last Updated:** 2026-04-20  
**Rancher Ready:** ✅ Yes
