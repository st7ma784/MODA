# The Web App (FastMODA)

FastMODA is a Flask-based web application and REST API that exposes MODA's algorithms
for browser-based and programmatic use, including GPU-accelerated variants and
machine-learning feature extraction.

## Running it

```bash
cd FastMODA
pip install -r requirements.txt
python app.py
```

By default this serves the app at `http://localhost:5000`. See
[Installation](../getting-started/installation.md#fastmoda-web-app-requirements) for
prerequisites.

## Browser UI

Once running, the app exposes a page per analysis type:

- **Spectral analysis** (`/analyze`-backed page) — single-signal time-frequency
  features, band powers, changepoint detection.
- **MODWT** — multi-scale wavelet decomposition.
- **Coherence** — pairwise phase synchronization across 2–6 signals (GPU-accelerated).
- **Bispectrum** — quadratic phase coupling detection (GPU-accelerated).
- **Bayesian inference** — directional coupling analysis (GPU-accelerated).
- **`/tests`** — a diagnostics page for exercising surrogate-generation and other
  internal test routines directly from the browser.

Each page uploads a signal file, submits it to the corresponding endpoint, and polls
for results, rendering them as interactive Plotly charts.

## Uploading data

Accepted file formats: `.mat` (reads the `sig` or `signal` variable), `.npy`, and
`.csv` (first column used). Multi-signal endpoints (coherence, bispectrum, Bayesian)
accept multiple file uploads in one request.

## Using it programmatically

Everything the browser UI does is backed by the plain REST API described in
[REST API Reference](../api-and-ml/rest-api-reference.md) — the browser is just one
client of it. See [Programmatic Usage](../api-and-ml/programmatic-usage.md) for calling
it directly from Python, and [Training a Classifier](../api-and-ml/training-a-classifier.md)
for turning its output into ML features.

## GPU acceleration

Coherence, bispectrum, and Bayesian inference endpoints can use GPU acceleration via
PyTorch when available. Check availability with `GET /api/gpu-info`, and control it
with the `USE_GPU` environment variable (`auto` | `true` | `false`).
