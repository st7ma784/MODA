# MODA Helm Chart — Ingress & Port Configuration Reference

## Port Mapping Architecture

```
External Request (Internet/LoadBalancer)
        ↓
   Ingress Controller
   (nginx-ingress / traefik / rancher-local)
        ↓
   Routing Rules (Path-based)
   ├── /api → ClusterIP Service: moda-fastmoda:5000
   ├── /api/v1 → ClusterIP Service: moda-fastmoda:5000
   ├── /health → ClusterIP Service: moda-fastmoda:5000
   └── / → ClusterIP Service: moda-mobile:80
        ↓
   Pod Service Endpoint
   ├── moda-fastmoda pods (port 5000)
   ├── moda-server pods (port 6789)
   └── moda-mobile pods (port 80)
```

## Ingress Service Port Mapping

### Current Configuration (values.yaml)

| Backend | Service Port | Pod Port | Path | Protocol |
|---------|-------------|----------|------|----------|
| FastMODA | 5000 | http (5000) | `/api*` | HTTP/WebSocket |
| Mobile | 80 | http (80) | `/` | HTTP |
| Server | 6789 | matlab (6789) | N/A* | TCP/gRPC |

*Server is internal only (not exposed via ingress)

## Service Type Configuration

### Option 1: ClusterIP (Default - Internal Only)

```yaml
fastmoda:
  service:
    type: ClusterIP
    port: 5000  # Internal access only
```

**Access methods:**
- Within cluster: `moda-fastmoda:5000`
- Via kubectl: `kubectl port-forward svc/moda-fastmoda 5000:5000`
- Via Rancher UI proxy: `https://<rancher>/api/v1/namespaces/moda/...`
- Via Ingress: Only if Ingress is enabled and configured

### Option 2: LoadBalancer (External Direct Access)

```yaml
fastmoda:
  service:
    type: LoadBalancer
    port: 5000
```

**Access methods:**
- External IP: `<EXTERNAL-IP>:5000`
- DNS: `moda-fastmoda.example.com:5000` (if DNS is configured)
- Via Ingress: If Ingress is also enabled

**Cloud-specific annotations:**

```yaml
# AWS Network Load Balancer
annotations:
  service.beta.kubernetes.io/aws-load-balancer-type: "nlb"

# Azure Load Balancer
annotations:
  service.beta.kubernetes.io/azure-load-balancer-internal: "false"

# Google Cloud LB
annotations:
  cloud.google.com/load-balancer-type: "External"

# MetalLB (On-prem)
annotations:
  metallb.universe.tf/address-pool: "default"
```

## Ingress Configuration

### Minimal Setup (ClusterIP Services + Ingress)

```yaml
fastmoda:
  service:
    type: ClusterIP  # Internal only
    port: 5000

mobile:
  service:
    type: ClusterIP
    port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: moda.example.com
      paths:
        - path: /api
          pathType: Prefix
          backend: fastmoda      # Routes to service:5000
        - path: /
          pathType: Prefix
          backend: mobile        # Routes to service:80
```

**Result:**
- `https://moda.example.com/api` → fastmoda service:5000
- `https://moda.example.com/` → mobile service:80
- Services are only accessible via ingress (not directly)

### Full External Access (LoadBalancer + Ingress)

```yaml
fastmoda:
  service:
    type: LoadBalancer  # Direct external access
    port: 5000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: moda.example.com
      paths:
        - path: /api
          backend: fastmoda
```

**Result:**
- Direct: `<EXTERNAL-IP>:5000` → fastmoda
- DNS: `moda-fastmoda.example.com:5000` → fastmoda
- Via Ingress: `moda.example.com/api` → fastmoda

## Ingress Rules (Detailed)

### Path-based Routing

```yaml
rules:
  - host: moda.example.com
    http:
      paths:
        # API endpoints
        - path: /api
          pathType: Prefix
          backend:
            service:
              name: moda-fastmoda
              port:
                number: 5000
        
        # API v1 endpoint
        - path: /api/v1
          pathType: Prefix
          backend:
            service:
              name: moda-fastmoda
              port:
                number: 5000
        
        # Health checks
        - path: /health
          pathType: Prefix
          backend:
            service:
              name: moda-fastmoda
              port:
                number: 5000
        
        # Web UI (fallback)
        - path: /
          pathType: Prefix
          backend:
            service:
              name: moda-mobile
              port:
                number: 80
```

### TLS/HTTPS Configuration

```yaml
ingress:
  tls:
    - secretName: moda-tls-cert
      hosts:
        - moda.example.com
  hosts:
    - host: moda.example.com
      paths:
        - path: /api
          backend: fastmoda
```

**Certificate sources:**
1. cert-manager automatic issuance:
   ```yaml
   annotations:
     cert-manager.io/cluster-issuer: letsencrypt-prod
   ```

2. Manual certificate:
   ```bash
   kubectl create secret tls moda-tls \
     --cert=path/to/cert.crt \
     --key=path/to/key.key \
     -n moda
   ```

## Rancher-Specific Annotations

```yaml
ingress:
  annotations:
    # Rancher UI integration
    rancher.io/ingress: "true"
    
    # Kubernetes standard
    kubernetes.io/ingress.class: nginx
    
    # nginx-ingress specific
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/force-ssl-redirect: "false"
    
    # Long timeouts for MATLAB processing (10 minutes)
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    
    # SSL/TLS
    cert-manager.io/cluster-issuer: letsencrypt-prod
```

## Port Troubleshooting

### Issue: "No endpoints available for service"

```bash
# Check if pods are running
kubectl get pods -n moda

# Verify pod is exposing correct port
kubectl describe pod -n moda $(kubectl get pods -n moda -l app.kubernetes.io/component=fastmoda -o jsonpath='{.items[0].metadata.name}')

# Check if port is correct in deployment
kubectl get deployment -n moda moda-fastmoda -o yaml | grep ports: -A 3
```

### Issue: Ingress shows no backend

```bash
# Check ingress status
kubectl describe ingress -n moda moda

# Verify service endpoints
kubectl get endpoints -n moda moda-fastmoda

# Check selector labels
kubectl get pods -n moda -L app.kubernetes.io/component
```

### Issue: LoadBalancer stuck on Pending

```bash
# Check if provider supports LoadBalancer
kubectl get nodes

# For MetalLB:
kubectl get ipaddresspools -n metallb-system

# Check service for errors
kubectl describe svc -n moda moda-fastmoda
```

## Best Practices

1. **Use Ingress for HTTP(S)** - Centralizes external access, supports multiple apps
2. **Use LoadBalancer for TCP** - Use for non-HTTP protocols (MATLAB server: 6789)
3. **Always set port.number explicitly** - Don't rely on port names
4. **Configure health checks** - Ensure Ingress knows if backends are ready
5. **Use TLS** - Secure external connections with cert-manager or manual certs
6. **Set timeouts** - MATLAB processing may take time, use proxy timeouts
7. **Label components** - Use `app.kubernetes.io/component` for better visibility

## Examples

### Quick Helm Install Commands

```bash
# Rancher local cluster (ingress only)
helm install moda ./helm/moda -n moda --create-namespace

# AWS EKS (LoadBalancer NLB)
helm install moda ./helm/moda \
  -n moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer \
  --set fastmoda.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb"

# On-prem with MetalLB
helm install moda ./helm/moda \
  -n moda \
  --create-namespace \
  --set fastmoda.service.type=LoadBalancer

# Custom values file
helm install moda ./helm/moda -f values-custom.yaml
```

## References

- [Kubernetes Ingress Documentation](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [nginx-ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [Rancher Documentation](https://rancher.com/docs/)
- [MetalLB Documentation](https://metallb.universe.tf/)
- [cert-manager Documentation](https://cert-manager.io/)
