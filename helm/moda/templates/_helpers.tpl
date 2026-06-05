{{- define "moda.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "moda.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "moda.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "moda.labels" -}}
helm.sh/chart: {{ include "moda.chart" . }}
{{ include "moda.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "moda.selectorLabels" -}}
app.kubernetes.io/name: {{ include "moda.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "moda.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "moda.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
moda.clusterDomain
  Resolves the cluster's base domain from, in priority order:
    1. .Values.ingress.domain          (explicit override)
    2. ConfigMap kube-system/clusteros-config .data.domain   (ClusterOS)
    3. Rancher local Cluster CR's `domain` metadata label    (legacy)
  Returns an empty string when none is set (caller should then fall back
  to nip.io via `moda.nodeIp` or accept path-based routing).
*/}}
{{- define "moda.clusterDomain" -}}
{{- $d := "" -}}
{{- if and .Values.ingress (hasKey .Values.ingress "domain") -}}
{{- $d = .Values.ingress.domain | toString -}}
{{- end -}}
{{- if not $d -}}
{{- $cm := lookup "v1" "ConfigMap" "kube-system" "clusteros-config" -}}
{{- if and $cm $cm.data -}}
{{- $d = index $cm.data "domain" | default "" -}}
{{- end -}}
{{- end -}}
{{- if not $d -}}
{{- $cluster := lookup "management.cattle.io/v3" "Cluster" "" "local" -}}
{{- if and $cluster $cluster.metadata $cluster.metadata.labels -}}
{{- $d = index $cluster.metadata.labels "domain" | default "" -}}
{{- end -}}
{{- end -}}
{{- $d -}}
{{- end }}

{{/*
moda.nodeIp
  First InternalIP of any node, used for nip.io fallback when no domain
  is configured. Returns an empty string if the cluster lookup fails.
*/}}
{{- define "moda.nodeIp" -}}
{{- $ip := "" -}}
{{- $nodes := lookup "v1" "Node" "" "" -}}
{{- if and $nodes $nodes.items -}}
{{- range $nodes.items -}}
{{- if not $ip -}}
{{- range .status.addresses -}}
{{- if and (eq .type "InternalIP") (not $ip) -}}
{{- $ip = .address -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $ip -}}
{{- end }}

{{/*
moda.host
  Build a fully-qualified hostname for a given subdomain prefix.
  Args: dict "ctx" $ "prefix" "moda" "explicit" "optional.full.host"
  Resolution order:
    1. explicit host if non-empty
    2. <prefix>.<cluster-domain> if cluster domain resolves
    3. <prefix>-<dashed-node-ip>.nip.io  (nip.io fallback)
    4. empty string  → caller may emit a path-based rule instead
*/}}
{{- define "moda.host" -}}
{{- $ctx := .ctx -}}
{{- $prefix := .prefix | default "moda" -}}
{{- $explicit := .explicit | default "" -}}
{{- if $explicit -}}
{{- $explicit -}}
{{- else -}}
{{- $domain := include "moda.clusterDomain" $ctx -}}
{{- if $domain -}}
{{- printf "%s.%s" $prefix $domain -}}
{{- else -}}
{{- $ip := include "moda.nodeIp" $ctx -}}
{{- if $ip -}}
{{- printf "%s-%s.nip.io" $prefix (replace "." "-" $ip) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}
