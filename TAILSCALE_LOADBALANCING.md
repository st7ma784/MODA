# Tailscale Filters & Load Balancing for FastMODA

This guide explains how to use Tailscale to expose FastMODA backends securely to mobile clients and how to scale horizontally with multiple replicas.

---

## Why Tailscale?

The FastMODA server runs inside Kubernetes and is currently exposed via an nginx ingress on a public hostname. Tailscale lets you:

- **Lock down the ingress** — remove the public hostname entirely; only Tailscale nodes can reach port 5000.
- **Skip TLS certificates** — Tailscale's WireGuard tunnel is already encrypted end-to-end.
- **Scale safely** — add or remove backend replicas without changing DNS or firewall rules.
- **Control access per-device** — your phone reaches the backend; a colleague's laptop does not.

---

## 1. Install Tailscale on K8s Nodes

### Option A — Tailscale Kubernetes Operator (recommended)

The operator runs as a Deployment in the cluster and advertises services as Tailscale hostnames.

```bash
helm repo add tailscale https://pkgs.tailscale.com/helmcharts
helm repo update

helm install tailscale-operator tailscale/tailscale-operator \
  --namespace tailscale \
  --create-namespace \
  --set oauth.clientId=<YOUR_OAUTH_CLIENT_ID> \
  --set oauth.clientSecret=<YOUR_OAUTH_CLIENT_SECRET>
```

Generate the OAuth credentials at https://login.tailscale.com/admin/settings/oauth (scope: `devices:write`).

### Option B — Subnet Router DaemonSet

Runs a Tailscale node per worker that routes the pod CIDR over Tailscale. Simpler but less fine-grained.

---

## 2. Expose the FastMODA Service via Tailscale

Annotate the existing Kubernetes Service (or add a new one) so the operator creates a Tailscale endpoint for it:

```yaml
# In helm/moda/templates/fastmoda-service.yaml (or patch the existing Service)
apiVersion: v1
kind: Service
metadata:
  name: fastmoda-ts
  annotations:
    tailscale.com/expose: "true"
    tailscale.com/hostname: "fastmoda"   # Tailscale DNS: fastmoda.<tailnet>.ts.net
spec:
  selector:
    app: fastmoda
  ports:
    - port: 5000
      targetPort: 5000
```

After applying, the operator provisions a Tailscale node called `fastmoda`. Any device on your tailnet that has the ACL permission (see §3) can reach it at `http://fastmoda:5000`.

---

## 3. ACL Policy (Filters)

In the Tailscale admin console → **Access Controls**, add a rule that allows only your mobile device(s) to reach the FastMODA port:

```json
{
  "tagOwners": {
    "tag:moda-server": ["autogroup:admin"],
    "tag:moda-client": ["autogroup:admin"]
  },

  "acls": [
    {
      "action": "accept",
      "src": ["tag:moda-client"],
      "dst": ["tag:moda-server:5000"]
    }
  ],

  "nodeAttrs": [
    {
      "target": ["fastmoda"],
      "attr": ["tag:moda-server"]
    }
  ]
}
```

Tag your phone as `tag:moda-client` from the admin console or via:

```bash
tailscale set --advertise-tags=tag:moda-client
```

**Result:** only devices tagged `moda-client` can hit port 5000. All other tailnet members — including colleagues — are blocked.

---

## 4. Load Balancing Multiple Replicas

### Option A — Kubernetes-native (preferred)

Scale the FastMODA Deployment and let Kubernetes kube-proxy do round-robin across pods. The single Tailscale `fastmoda` Service endpoint load-balances automatically.

```bash
kubectl scale deployment fastmoda --replicas=3
```

Or enable HPA in `helm/moda/values.yaml`:

```yaml
fastmoda:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 8
    targetCPUUtilizationPercentage: 60
```

The Tailscale operator's Service endpoint remains `fastmoda:5000` regardless of replica count — no client-side changes required.

### Option B — Multiple Tailscale Nodes with a VIP

If you want client-visible load balancing (e.g., for geographic distribution across clusters):

1. Tag each cluster's FastMODA operator with a shared hostname suffix (e.g., `fastmoda-eu`, `fastmoda-us`).
2. Use Tailscale's `multipath` feature (Business plan) or a simple DNS round-robin:

```bash
# Add both IPs to a custom MagicDNS override
# Tailscale admin → DNS → Add nameserver override for fastmoda.ts.net
```

3. In the Flutter app, configure failover logic in `FastModaClient`:

```dart
final _endpoints = [
  'http://fastmoda-eu:5000',
  'http://fastmoda-us:5000',
];

Future<Map<String, dynamic>> checkHealth() async {
  for (final url in _endpoints) {
    try {
      final res = await Dio(BaseOptions(baseUrl: url)).get('/health');
      if (res.statusCode == 200) {
        setBaseUrl(url); // lock onto healthy endpoint
        return res.data;
      }
    } catch (_) {}
  }
  throw Exception('All FastMODA endpoints unreachable');
}
```

---

## 5. Flutter App Configuration

Change the server URL in the MODA app Settings tab to the Tailscale hostname:

```
http://fastmoda:5000
```

or the MagicDNS FQDN:

```
http://fastmoda.<your-tailnet>.ts.net:5000
```

The app stores this in secure storage and uses it for all API calls (including the `X-API-Key` header, which remains unchanged).

> **Note:** When using Tailscale, you can drop the nginx ingress and the public hostname entirely. Set `fastmoda.service.type: ClusterIP` in `helm/moda/values.yaml` — the pod is only reachable via Tailscale.

---

## 6. Removing the Public Ingress

Once Tailscale is working, lock down the cluster:

```yaml
# helm/moda/values.yaml
ingress:
  enabled: false

fastmoda:
  service:
    type: ClusterIP
```

```bash
helm upgrade moda ./helm/moda -f helm/moda/values.yaml
```

Verify: `curl https://moda.example.com/health` should time out; `curl http://fastmoda:5000/health` (from a Tailscale node) should return `{"status":"ok"}`.

---

## 7. Session Persistence for Long-Running Jobs

MODWT and bispectrum analyses can run for minutes. If a pod is replaced mid-job (HPA scale-down or rollout), the in-flight task is lost. Two strategies:

**a) Sticky sessions** — add `sessionAffinity: ClientIP` to the Kubernetes Service. The same pod handles all requests from a given client IP. Simple but imperfect (same IP can change on mobile).

**b) Shared task queue** — expose a Redis or Celery broker that all FastMODA pods share. The pod that receives `/analyze` enqueues the task; any pod can pick it up; `/status/<task_id>` checks the shared store. This is the correct approach for production at scale.

For the Helm chart, `sessionAffinity: ClientIP` is the immediate fix:

```yaml
# fastmoda-service.yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 600
```

---

## Quick-Reference Checklist

| Step | Command / Action |
|------|-----------------|
| Install Tailscale operator | `helm install tailscale-operator tailscale/tailscale-operator …` |
| Annotate FastMODA Service | `tailscale.com/expose: "true"`, `hostname: fastmoda` |
| Set ACL tags | Tag server `tag:moda-server`, phone `tag:moda-client`, allow `:5000` |
| Scale replicas | `kubectl scale deployment fastmoda --replicas=N` or enable HPA |
| Update app URL | Settings → `http://fastmoda:5000` |
| Disable public ingress | `ingress.enabled: false` in `values.yaml`, `helm upgrade` |
