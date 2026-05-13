# MODA App — Development Plan

## Overview

The MODA App is a frontend (Flutter web / mobile) that sends signal data to a
running **FastMODA** service and displays the results.  All analysis is handled
server-side; the app is responsible only for data input, display, and UX.

**FastMODA base URL (default):** `http://localhost:5000`  
Configurable at runtime — can point at a local server, a Raspberry Pi on the
home network, or any deployment of the FastMODA Docker image.

---

## Architecture

```
┌─────────────────────────────────────────┐
│              MODA App (Flutter)         │
│                                         │
│  ┌────────────┐   ┌──────────────────┐  │
│  │ Input Layer│   │  Display Layer   │  │
│  │            │   │                  │  │
│  │ • File     │   │ • Signal plot    │  │
│  │   upload   │   │ • Spectrogram    │  │
│  │ • BLE      │   │ • Band powers    │  │
│  │   sensor   │   │ • Freq timeline  │  │
│  │   (Phase 3)│   │ • MODWT viewer   │  │
│  └────────────┘   │ • Coherence      │  │
│        │          │ • Bispectrum     │  │
│        └──────────┤ • Bayesian       │  │
│                   └──────────────────┘  │
│                           │             │
│              ┌────────────┘             │
│              ▼                          │
│   FastMODA API Client                   │
│   (HTTP + polling /status/<task_id>)    │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   FastMODA Service  │
    │   (Python/Flask)    │
    │                     │
    │  POST /analyze      │
    │  POST /analyze_modwt│
    │  POST /analyze_coh  │
    │  POST /analyze_bisp │
    │  POST /analyze_bay  │
    │  GET  /status/:id   │
    │  GET  /health       │
    └─────────────────────┘
```

---

## FastMODA API Contract

See `FastMODA/API.md` for the full reference.  Key points for the app:

### Workflow (all analysis types)

1. POST file + params → receive `{ task_id }`
2. Poll `GET /status/<task_id>` every ~500 ms
3. Response `status: "complete"` → render `results`
4. Response `status: "error"` → show error message

### Endpoints used by the app

| Endpoint | Input | What the app shows |
|---|---|---|
| `POST /analyze` | 1 signal, `fs`, `win`, `pen` | Signal + spectrogram + band powers + freq timeline + changepoints |
| `POST /analyze_modwt` | 1 signal, `fs`, `wavelet`, `level` | Coefficient subplots + heatmap + energy per scale |
| `POST /analyze_coherence` | 2–6 signals, `fs` | Coherence heatmap per pair + phase difference |
| `POST /analyze_bispectrum` | 1–2 signals, `fs` | Bispectrum 2D plot + top couplings table |
| `POST /analyze_bayesian` | 2 signals, `fs`, bands | Coupling strength + direction over time |
| `GET /api/gpu-info` | — | Badge showing GPU / CPU mode |
| `GET /health` | — | Server connection status indicator |

### Result format

All plot fields are Plotly JSON strings — render with `plotly_flutter` or the
`flutter_inappwebview` approach of injecting JSON into a Plotly CDN page.

---

## Development Phases

### Phase 1 — UI Shell (Week 1–2)  ← **Start here**

Goal: a fully navigable app with realistic layouts and **mock data** — no API
calls.  Design decisions should be made and locked before any backend wiring.

**Screens to build:**

| Screen | Contents |
|---|---|
| **Home / Dashboard** | Server status chip, GPU badge, quick-action buttons per analysis type |
| **Spectral Analysis** | File picker + params form, progress bar, 6-panel result view (signal, spectrogram, band powers, freq timeline, instantaneous freq, periodicity) |
| **MODWT** | File picker + wavelet/level selectors, 4-panel result view (coefficients, heatmap, reconstruction, energy bar chart) |
| **Coherence** | Multi-file picker (2–6), result tabs per signal pair |
| **Bispectrum** | 1–2 file pickers, freq range sliders, 2D heatmap + coupling table |
| **Bayesian** | 2-file picker, band range inputs, coupling strength / direction chart |
| **Settings** | FastMODA server URL, default sampling rate, theme |

**Mock data strategy:**

- Hardcode a small `.npy` file in `assets/` as example signal.
- For each result panel, embed a pre-generated Plotly JSON blob in
  `assets/mock/` — use the actual FastMODA service once to generate these.
- Flip a `USE_MOCK` constant to switch between mock and live.

**Deliverable:** Full app navigation works, every result panel renders with
mock data, all forms are functional (params validate, errors surface cleanly).

---

### Phase 2 — FastMODA API Integration (Week 3)

Goal: replace mock calls with real HTTP requests to FastMODA.

- Implement `FastModaClient` service class:
  - Base URL configurable via Settings screen (persisted in SharedPreferences)
  - `analyzeSpectral(file, params)` → polls status → emits progress stream
  - `analyzeMODWT(file, params)` → same pattern
  - `analyzeCoherence(files, params)` → same
  - `analyzeBispectrum(files, params)` → same
  - `analyzeBayesian(files, params)` → same
  - `getGpuInfo()` → one-shot GET
  - `healthCheck()` → used by status chip on Dashboard
- Wire each analysis screen's submit action to `FastModaClient`
- Display real Plotly results (replace mock JSON blobs)
- Error handling: network timeout, task error, server unreachable

**Deliverable:** App works end-to-end against a local FastMODA instance.
All 5 analysis types submit, poll, and display real results.

---

### Phase 3 — BLE Signal Input (Week 4–5)

Goal: add a Bluetooth sensor as an alternative signal source alongside file upload.

- BLE device discovery screen (scan + pair)
- Live streaming buffer: accumulate samples into a sliding window
- "Send to FastMODA" action: export buffer as `.npy`, submit to `/analyze`
- Connection quality indicator (RSSI, packet loss)
- Auto-save raw BLE data to local SQLite for later re-analysis

See `BLUETOOTH_PROTOCOL.md` for the custom MODA BLE specification.

**Deliverable:** User can connect a BLE device, stream a signal, and submit
it for spectral analysis on FastMODA with one tap.

---

### Phase 4 — Polish (Week 6)

- Session history (list past analyses, re-open results)
- Export: save Plotly charts as PNG, export raw signal as CSV / MAT
- Offline notice: graceful message when FastMODA is unreachable
- Accessibility: semantic labels, sufficient contrast
- Dark mode

---

## Technology Stack

| Concern | Choice | Notes |
|---|---|---|
| Framework | Flutter | Targets iOS, Android, and web from one codebase |
| HTTP client | `dio` | Multipart file upload + retry |
| Plotly rendering | `flutter_inappwebview` | Inject Plotly CDN + JSON into a headless WebView per chart |
| BLE | `flutter_blue_plus` | Phase 3 |
| State management | Riverpod | Provider-per-screen, `AsyncNotifier` for API calls |
| Local storage | `shared_preferences` (settings) + `sqflite` (session history) |
| Charts (non-Plotly) | `fl_chart` | Progress bars, band-power summary cards |

### Plotly rendering approach

FastMODA returns Plotly JSON strings.  The simplest reliable approach in Flutter:

```dart
// assets/plotly_host.html — loaded once, JS bridge exposed
final String plotlyHost = await rootBundle.loadString('assets/plotly_host.html');
controller.loadHtmlString(plotlyHost);
// Then inject:
controller.runJavascript('renderPlot($plotlyJson)');
```

`plotly_host.html` loads Plotly from CDN and exposes `window.renderPlot(json)`.

---

## FastMODA Service Configuration

The app needs to know where FastMODA is running.  Default: `http://localhost:5000`.

For mobile testing against a local server, the device and server must share the
same network.  Common setups:

| Setup | URL to enter in Settings |
|---|---|
| Dev machine (Flutter web in browser) | `http://localhost:5000` |
| Android emulator → host machine | `http://10.0.2.2:5000` |
| Physical phone → same WiFi | `http://<machine-ip>:5000` |
| Raspberry Pi on home network | `http://moda-server.local:5000` |

---

## What is NOT in scope for MVP

- On-device signal processing (deferred; FastMODA handles all analysis)
- Cloud backend (no uploads outside the local network unless user configures it)
- Neural network / TFLite inference

---

## References

- `FastMODA/API.md` — full API reference with request/response schemas
- `BLUETOOTH_PROTOCOL.md` — BLE packet format for Phase 3
- `TESTING_STRATEGY.md` — test plan
- `SECURITY_ARCHITECTURE.md` — auth / network security notes
