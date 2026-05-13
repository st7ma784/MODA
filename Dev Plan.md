# Dev Plan

## Current focus: APP Phase 1 — UI Shell

Build every screen with mock data before any API calls.
FastMODA is the backend for all analysis; no on-device processing.

See `APP/MOBILE_APP_PLAN.md` for the full plan.

---

## Screens to build (Phase 1)

| Screen | Status |
|---|---|
| Home / Dashboard (server status, GPU badge, nav) | TODO |
| Spectral Analysis (file picker → 6-panel result view) | TODO |
| MODWT (file picker → 4-panel result view) | TODO |
| Coherence (multi-file → per-pair tabs) | TODO |
| Bispectrum (file picker → heatmap + couplings table) | TODO |
| Bayesian (2-file → coupling + direction chart) | TODO |
| Settings (FastMODA URL, sampling rate, theme) | TODO |

## After Phase 1

- Phase 2: Wire up `FastModaClient` HTTP service, replace mock JSON   ---> Assumption about hosting on university VM 
- Phase 3: BLE input
- Phase 4: Session history, export, polish

---

## FastMODA endpoints in use

```
GET  /health                → Dashboard server status chip
GET  /api/gpu-info          → Dashboard GPU badge
POST /analyze               → Spectral Analysis screen
POST /analyze_modwt         → MODWT screen
POST /analyze_coherence     → Coherence screen
POST /analyze_bispectrum    → Bispectrum screen
POST /analyze_bayesian      → Bayesian screen
GET  /status/<task_id>      → progress polling (all screens)
```

Default URL: `http://localhost:5000`
Configurable in Settings.


## Priorities: 

[x] Fix MATLAB's memory and UI Issues
[x] Bring MODA up to date with 2026a
[ ] Make API For MODA
[ ] Build APP UI 
[ ] Test Connections to app
[ ] Ship
