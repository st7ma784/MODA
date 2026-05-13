#!/usr/bin/env python3
"""
MODA App Emulator
=================
Browser-based phone-frame mirror of the Flutter mobile app.

Architecture
------------
  signal_generator  (port 8090) — synthetic BLE signal + on-device DSP
  fastmoda          (port 5000) — all analysis endpoints
  emulator          (port 8080) — this file; proxies between both + serves UI

Usage
-----
  # Development (all local):
  python signal_generator.py --port 8090 &
  python emulator.py --port 8080 --fastmoda http://localhost:5000 \
                     --signal http://localhost:8090

  # Docker (uses service names on emulator-net):
  docker-compose -f docker-compose.emulator.yml up
"""

import argparse
import io
import os

import numpy as np
import requests as req
from flask import Flask, Response, jsonify, request

# ── config ─────────────────────────────────────────────────────────────────────

FASTMODA_API_KEY = "moda_8e6695088c2e3114cbb25e3554544f2577cd53c58a3672ac"

_fastmoda_url = os.environ.get("FASTMODA_URL", "http://localhost:5000")
_signal_url   = os.environ.get("SIGNAL_URL",   "http://localhost:8090")

# ── Flask ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)


def _sig(path, **kwargs):
    """GET from signal_generator."""
    return req.get(f"{_signal_url}{path}", timeout=5, **kwargs)


def _sig_post(path, **kwargs):
    """POST to signal_generator."""
    return req.post(f"{_signal_url}{path}", timeout=5, **kwargs)


def _fm(path, **kwargs):
    """GET/POST to FastMODA with API key injected."""
    if "headers" not in kwargs:
        kwargs["headers"] = {}
    kwargs["headers"]["X-API-Key"] = FASTMODA_API_KEY
    return kwargs


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


# ── Signal proxy ───────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    try:
        return jsonify(_sig("/metrics").json())
    except Exception as e:
        return jsonify({"error": str(e), "streaming": False}), 200


@app.route("/api/stream/start", methods=["POST"])
def api_start():
    _sig_post("/stream/start")
    return jsonify({"ok": True})


@app.route("/api/stream/stop", methods=["POST"])
def api_stop():
    _sig_post("/stream/stop")
    return jsonify({"ok": True})


@app.route("/api/preset", methods=["POST"])
def api_preset():
    _sig_post("/preset", json=request.json)
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    global _fastmoda_url, _signal_url
    if request.method == "POST":
        d = request.json or {}
        if "fastmoda_url" in d:
            _fastmoda_url = d["fastmoda_url"].rstrip("/")
        try:
            _sig_post("/settings", json=d)
        except Exception:
            pass
    try:
        s = _sig("/settings").json()
    except Exception:
        s = {}
    s["fastmoda_url"] = _fastmoda_url
    s["signal_url"]   = _signal_url
    return jsonify(s)


# ── FastMODA proxy ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    try:
        r = req.get(f"{_fastmoda_url}/health", timeout=3,
                    headers={"X-API-Key": FASTMODA_API_KEY})
        return jsonify({"ok": True, "detail": r.json()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/gpu")
def api_gpu():
    try:
        r = req.get(f"{_fastmoda_url}/api/gpu-info", timeout=3,
                    headers={"X-API-Key": FASTMODA_API_KEY})
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/status/<task_id>")
def api_status(task_id):
    try:
        r = req.get(f"{_fastmoda_url}/status/{task_id}", timeout=5,
                    headers={"X-API-Key": FASTMODA_API_KEY})
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Unified analysis dispatcher.
    Fetches signal from signal_generator, forwards to the right FastMODA endpoint.
    Two-channel analyses self-pair the signal.
    """
    body    = request.json or {}
    atype   = body.get("type", "analyze")
    fs      = body.get("fs", 256.0)
    hdrs    = {"X-API-Key": FASTMODA_API_KEY}

    # Fetch current signal buffer
    try:
        npy_bytes = _sig("/signal.npy").content
    except Exception as e:
        return jsonify({"error": f"Cannot reach signal generator: {e}"}), 500

    fm = _fastmoda_url

    try:
        # ── single-channel ────────────────────────────────────────────────────
        if atype == "analyze":
            r = req.post(f"{fm}/analyze",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs)}, headers=hdrs, timeout=12)

        elif atype == "stft":
            r = req.post(f"{fm}/analyze_stft",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs), "window": body.get("window", "hann"),
                      "window_size": str(body.get("window_size", 256)),
                      "kaiser_beta": str(body.get("kaiser_beta", 8.6))},
                headers=hdrs, timeout=12)

        elif atype == "wft":
            r = req.post(f"{fm}/analyze_wft",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs)}, headers=hdrs, timeout=12)

        elif atype == "cwt":
            r = req.post(f"{fm}/analyze_cwt",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs),
                      "wavelet": body.get("wavelet", "lognorm"),
                      "n_cycles": str(body.get("n_cycles", 6.0)),
                      "freq_min": str(body.get("freq_min", 0.5)),
                      "freq_max": str(body.get("freq_max", fs / 2)),
                      "n_freqs": str(body.get("n_freqs", 50)),
                      "padding": body.get("padding", "symmetric"),
                      "cut_edges": "true" if body.get("cut_edges") else "false"},
                headers=hdrs, timeout=12)

        elif atype == "ridge":
            r = req.post(f"{fm}/analyze_ridge",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs), "freq_min": "0.5",
                      "freq_max": str(fs / 2), "n_freqs": "50"},
                headers=hdrs, timeout=15)

        elif atype == "hilbert":
            r = req.post(f"{fm}/analyze_hilbert",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs)}, headers=hdrs, timeout=12)

        elif atype == "butter":
            r = req.post(f"{fm}/filter_butter",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs),
                      "f_low":  str(body.get("f_low",  8.0)),
                      "f_high": str(body.get("f_high", 12.0)),
                      "order":  str(body.get("order",   4)),
                      "detrend_degree": str(body.get("detrend_degree", 0))},
                headers=hdrs, timeout=12)

        elif atype == "bispectrum":
            r = req.post(f"{fm}/analyze_bispectrum",
                files=[("files", ("s.npy", npy_bytes))],
                data={"fs": str(fs)}, headers=hdrs, timeout=12)

        elif atype == "modwt":
            r = req.post(f"{fm}/analyze_modwt",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs), "wavelet": "la8", "level": "5"},
                headers=hdrs, timeout=30)

        elif atype == "surrogates":
            r = req.post(f"{fm}/analyze_surrogates",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs),
                      "test_type": body.get("test_type", "spectral"),
                      "n_surrogates": str(body.get("n_surrogates", 19)),
                      "surrogate_method": body.get("method", "phase_randomization")},
                headers=hdrs, timeout=60)

        elif atype == "features":
            r = req.post(f"{fm}/analyze_features",
                files={"file": ("signal.npy", npy_bytes)},
                data={"fs": str(fs), "analyses": "spectral,phase"},
                headers=hdrs, timeout=20)

        # ── two-channel (self-paired) ──────────────────────────────────────────
        elif atype == "coherence":
            r = req.post(f"{fm}/analyze_coherence",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs), "win": "1.0"}, headers=hdrs, timeout=20)

        elif atype == "bispec4":
            r = req.post(f"{fm}/analyze_bispectrum4",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs)}, headers=hdrs, timeout=20)

        elif atype == "biphase":
            r = req.post(f"{fm}/analyze_biphase",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs),
                      "f1": str(body.get("f1", 6.0)),
                      "f2": str(body.get("f2", 10.0)),
                      "wavelet": body.get("wavelet", "lognorm")},
                headers=hdrs, timeout=20)

        elif atype == "coupling":
            r = req.post(f"{fm}/analyze_coupling",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs), "bn": "2", "win_s": "1",
                      "band1_low": "8", "band1_high": "12",
                      "band2_low": "8", "band2_high": "12"},
                headers=hdrs, timeout=30)

        elif atype == "bayesian":
            r = req.post(f"{fm}/analyze_bayesian",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs), "band1_low": "8", "band1_high": "12",
                      "band2_low": "8", "band2_high": "12", "window_s": "1"},
                headers=hdrs, timeout=30)

        elif atype == "syncmap":
            r = req.post(f"{fm}/analyze_syncmap",
                files=[("files", ("s0.npy", npy_bytes)),
                       ("files", ("s1.npy", npy_bytes))],
                data={"fs": str(fs), "bn": "2", "win_s": "1",
                      "band1_low": "8", "band1_high": "12",
                      "band2_low": "8", "band2_high": "12"},
                headers=hdrs, timeout=30)

        else:
            return jsonify({"error": f"Unknown analysis type: {atype}"}), 400

        return jsonify(r.json()), r.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MODA App Emulator</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#121212;--sur:#1E1E1E;--card:#272727;
  --pri:#00897B;--pril:#4DB6AC;
  --hi:rgba(255,255,255,.87);--med:rgba(255,255,255,.54);
  --lo:rgba(255,255,255,.38);--div:rgba(255,255,255,.08);
}
body{background:radial-gradient(ellipse at top,#0d1f1e,#080808 70%);
     display:flex;flex-direction:column;align-items:center;
     justify-content:flex-start;min-height:100vh;
     font-family:-apple-system,'Roboto',sans-serif;padding:24px 0;}
.badge{font-size:11px;letter-spacing:2px;color:var(--pril);
       text-transform:uppercase;margin-bottom:14px;opacity:.7;}
.phone{width:393px;background:#1c1c1e;border-radius:50px;padding:14px;
       box-shadow:0 40px 100px rgba(0,0,0,.8),inset 0 0 0 1px rgba(255,255,255,.1);}
.screen{background:var(--bg);border-radius:38px;overflow:hidden;
        display:flex;flex-direction:column;height:812px;}
.island{position:absolute;top:12px;left:50%;transform:translateX(-50%);
        width:120px;height:34px;background:#000;border-radius:17px;z-index:10;}
.status-bar{height:52px;display:flex;align-items:flex-end;
            justify-content:space-between;padding:0 20px 8px;
            font-size:12px;color:var(--med);flex-shrink:0;position:relative;}
.screen-body{flex:1;overflow-y:auto;overflow-x:hidden;}
.screen-body::-webkit-scrollbar{width:2px}
.screen-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,.15);border-radius:2px}
.bottom-nav{height:72px;background:rgba(18,18,18,.95);border-top:1px solid var(--div);
            display:flex;align-items:flex-start;padding-top:8px;flex-shrink:0;}
.nav-btn{flex:1;background:none;border:none;cursor:pointer;
         display:flex;flex-direction:column;align-items:center;gap:3px;
         color:var(--lo);font-size:9px;transition:color .2s;}
.nav-btn svg{width:24px;height:24px}
.nav-btn.active{color:var(--pril)}
.tab{display:none;padding:12px 14px 80px;}
.tab.active{display:block}
.app-bar{font-size:22px;font-weight:600;color:var(--hi);padding:8px 0 16px;}
.card{background:var(--card);border-radius:14px;padding:14px;margin-bottom:10px;}
.lbl{font-size:10px;font-weight:700;color:var(--pril);text-transform:uppercase;
     letter-spacing:.9px;margin-bottom:10px;}
.m-row{display:flex;gap:8px;margin-bottom:10px;}
.m-chip{flex:1;background:var(--card);border-radius:10px;padding:10px 12px;}
.m-lbl{font-size:9px;color:var(--lo);margin-bottom:2px;}
.m-val{font-size:14px;font-weight:700;color:var(--pril);}
.band-row{display:flex;align-items:center;gap:8px;margin-bottom:7px;}
.band-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.band-info{width:72px;flex-shrink:0;}
.band-name{font-size:12px;font-weight:600;color:var(--hi);}
.band-hz{font-size:9px;color:var(--lo);}
.bar-wrap{flex:1;height:5px;background:var(--div);border-radius:3px;overflow:hidden;}
.bar{height:100%;border-radius:3px;transition:width .35s ease;}
.band-pct{width:28px;text-align:right;font-size:10px;color:var(--med);}
.sdot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.sdot.up{background:#4CAF50}.sdot.down{background:#F44336}.sdot.unknown{background:var(--lo)}
.btn{width:100%;padding:14px;border-radius:12px;border:none;font-size:14px;
     font-weight:500;cursor:pointer;transition:opacity .2s;}
.btn:disabled{opacity:.35;cursor:not-allowed;}
.btn-fill{background:var(--pri);color:#fff;}
.btn-out{background:transparent;border:1px solid var(--pri);color:var(--pril);}
/* Analysis cards */
.a-card{background:var(--card);border-radius:14px;margin-bottom:6px;overflow:hidden;}
.a-hdr{display:flex;align-items:center;padding:11px 14px;gap:10px;}
.a-ico{color:var(--pril);font-size:16px;width:22px;text-align:center;flex-shrink:0;}
.a-text{flex:1;min-width:0;}
.a-title{font-size:13px;color:var(--hi);}
.a-sub{font-size:10px;color:var(--lo);margin-top:1px;}
.a-sub.warn{color:#FF9800;}
.a-btn{background:none;border:none;color:var(--pril);cursor:pointer;font-size:17px;padding:4px 2px;flex-shrink:0;}
.a-result{padding:0 14px 10px;display:none;}
.a-result.show{display:block;}
.r-row{display:flex;font-size:11px;padding:1px 0;gap:5px;}
.r-key{color:var(--lo);flex-shrink:0;}.r-val{color:var(--hi);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.prog-wrap{height:3px;background:var(--div);border-radius:2px;margin-top:4px;}
.prog-bar{height:100%;border-radius:2px;background:var(--pri);transition:width .4s;}
/* Sub-tabs */
.sub-bar{display:flex;border-bottom:1px solid var(--div);margin:0 -14px 12px;padding:0 14px;}
.sub-tab{padding:9px 10px;font-size:12px;cursor:pointer;color:var(--lo);
         border-bottom:2px solid transparent;transition:color .2s;}
.sub-tab.active{color:var(--pril);border-bottom-color:var(--pril);}
.sub-panel{display:none;}.sub-panel.active{display:block;}
/* Section labels */
.sec-lbl{font-size:10px;font-weight:700;color:var(--med);text-transform:uppercase;
         letter-spacing:.9px;margin:14px 0 6px;}
.sec-lbl:first-child{margin-top:0;}
/* Settings */
.set-row{padding:12px 0;border-bottom:1px solid var(--div);}
.set-row:last-child{border:none;}
.set-lbl{font-size:12px;color:var(--hi);margin-bottom:6px;}
.set-inp{width:100%;background:rgba(255,255,255,.05);border:1px solid var(--div);
         border-radius:8px;padding:8px 12px;color:var(--hi);font-size:12px;}
.set-inp:focus{outline:none;border-color:var(--pri);}
/* BLE device */
.dev-name{font-size:15px;font-weight:700;color:var(--hi);margin-bottom:2px;}
.dev-sub{font-size:11px;color:var(--pril);}
.dev-detail{font-size:11px;color:var(--med);margin:2px 0;}
.divider{height:1px;background:var(--div);margin:10px 0;}
/* Changepoint chips */
.cp-wrap{display:flex;flex-wrap:wrap;gap:6px;}
.cp-chip{background:rgba(0,137,123,.18);border:1px solid rgba(0,137,123,.4);
         border-radius:14px;padding:3px 10px;font-size:11px;color:var(--pril);}
/* Spinner */
@keyframes spin{to{transform:rotate(360deg)}}
.spin{width:14px;height:14px;border:2px solid rgba(255,255,255,.2);
      border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;
      display:inline-block;vertical-align:middle;margin-right:5px;}
/* 2ch badge */
.badge-2ch{font-size:9px;background:rgba(255,152,0,.15);border:1px solid rgba(255,152,0,.4);
           color:#FF9800;border-radius:8px;padding:1px 5px;margin-left:4px;}
</style>
</head>
<body>
<div class="badge">&#9679; MODA App Emulator</div>
<div class="phone">
<div class="screen">
  <div class="island"></div>
  <div class="status-bar" style="position:relative;">
    <span>9:41</span>
    <span id="ble-badge" style="display:none;font-size:10px;color:var(--pril)">&#9679; MODA-SIM</span>
    <span>&#9646;&#9646; 87%</span>
  </div>
  <div class="screen-body" id="screen-body">

  <!-- DASHBOARD -->
  <div id="tab-dashboard" class="tab active">
    <div class="app-bar">Dashboard</div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
      <span id="ble-status-txt" style="font-size:11px;color:var(--lo)">No device connected</span>
      <span style="font-size:11px;color:var(--lo)">Server <span class="sdot unknown" id="sdot-dash"></span></span>
    </div>
    <div class="card">
      <div class="lbl">Live Signal</div>
      <div style="height:110px;position:relative"><canvas id="wave-dash"></canvas></div>
    </div>
    <div class="m-row">
      <div class="m-chip"><div class="m-lbl">Sample Rate</div><div class="m-val" id="m-fs">256 Hz</div></div>
      <div class="m-chip"><div class="m-lbl">Dominant Freq</div><div class="m-val" id="m-dom">— Hz</div></div>
      <div class="m-chip"><div class="m-lbl">Quality</div><div class="m-val" id="m-qual">—%</div></div>
    </div>
    <div class="m-row">
      <div class="m-chip"><div class="m-lbl">Entropy</div><div class="m-val" id="m-entr">—</div></div>
      <div class="m-chip"><div class="m-lbl">Flatness</div><div class="m-val" id="m-flat">—</div></div>
    </div>
    <div class="card">
      <div class="lbl">Band Power</div>
      <div id="bands-dash"></div>
    </div>
  </div>

  <!-- DEVICES -->
  <div id="tab-ble" class="tab">
    <div class="app-bar">Devices</div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
        <div><div class="dev-name">MODA-SIM-001</div><div class="dev-sub">Simulated MODA Device</div></div>
        <span style="font-size:10px;color:var(--lo)">-62 dBm</span>
      </div>
      <div class="dev-detail">Firmware: SIM-1.0.0 &bull; Battery: 87%</div>
      <div class="dev-detail">Max channels: 8 &bull; Max rate: 1024 Hz</div>
      <div class="divider"></div>
      <div class="dev-detail"><b style="color:var(--hi)">Sample rate:</b> <span id="cfg-fs">256 Hz</span></div>
      <div class="dev-detail"><b style="color:var(--hi)">Channels:</b> 1 &bull; Format: Float32 LE</div>
      <div class="divider"></div>
      <div id="stream-status" style="font-size:11px;color:var(--lo);margin-bottom:10px;">Status: Idle</div>
      <button id="stream-btn" class="btn btn-fill" onclick="toggleStream()">Start Streaming</button>
    </div>
    <div class="card" style="margin-top:10px;">
      <div class="lbl">Signal Preset</div>
      <select class="set-inp" onchange="applyPreset(this.value)" style="margin-top:4px;">
        <option value="resting">Resting EEG &mdash; &alpha; dominant (10 Hz)</option>
        <option value="active">Active / Eyes Open &mdash; &beta; dominant</option>
        <option value="drowsy">Drowsy &mdash; &theta; increasing</option>
        <option value="sleep">Sleep N2 &mdash; &delta; + spindles</option>
        <option value="noise">White Noise &mdash; high entropy</option>
      </select>
    </div>
  </div>

  <!-- ANALYSIS -->
  <div id="tab-analysis" class="tab">
    <div class="app-bar">Analysis</div>
    <div class="sub-bar">
      <div class="sub-tab active" onclick="subTab('live',this)">Live</div>
      <div class="sub-tab" onclick="subTab('spectral',this)">Spectral</div>
      <div class="sub-tab" onclick="subTab('server',this)">
        Server<span class="sdot unknown" id="sdot-analysis" style="margin-left:5px;"></span>
      </div>
      <div class="sub-tab" onclick="subTab('history',this)">History</div>
    </div>

    <!-- Live -->
    <div id="sub-live" class="sub-panel active">
      <div class="card"><div class="lbl">Real-Time Signal</div>
        <div style="height:140px;position:relative"><canvas id="wave-analysis"></canvas></div></div>
      <div class="card"><div class="lbl">Power Spectrum</div>
        <div style="height:120px;position:relative"><canvas id="spec-chart"></canvas></div></div>
      <div class="card">
        <div class="lbl">Changepoints <span style="font-size:9px;color:var(--lo);font-weight:400;">(on-device)</span></div>
        <div id="cp-display"><div style="font-size:11px;color:var(--lo);padding:10px 0;text-align:center;">Collecting…</div></div>
      </div>
    </div>

    <!-- Spectral -->
    <div id="sub-spectral" class="sub-panel">
      <div class="m-row">
        <div class="m-chip"><div class="m-lbl">Entropy (0%=tone)</div><div class="m-val" id="m-entropy2">—</div></div>
        <div class="m-chip"><div class="m-lbl">Flatness (0%=tonal)</div><div class="m-val" id="m-flatness2">—</div></div>
      </div>
      <div class="card"><div class="lbl">Power Spectral Density</div>
        <div style="height:160px;position:relative"><canvas id="psd-chart"></canvas></div></div>
      <div class="lbl" style="margin-top:4px;">Band Power Breakdown</div>
      <div id="bands-spectral" style="margin-top:8px;"></div>
    </div>

    <!-- Server -->
    <div id="sub-server" class="sub-panel">
      <!-- Server status -->
      <div class="card" style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <span id="srv-icon" style="font-size:20px;">&#9679;</span>
        <div style="flex:1;"><div id="srv-txt" style="font-size:13px;color:var(--hi);">Checking…</div>
          <div id="srv-url" style="font-size:10px;color:var(--lo);"></div></div>
        <button onclick="checkHealth()" style="background:none;border:none;color:var(--pril);cursor:pointer;font-size:14px;">&#8635;</button>
        <button onclick="resetAll()" title="Reset stuck cards" style="background:none;border:none;color:#FF9800;cursor:pointer;font-size:11px;padding:4px;">&#10005;</button>
      </div>

      <!-- Activity log -->
      <div style="background:var(--card);border-radius:14px;padding:10px 14px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:9px;font-weight:700;color:var(--med);text-transform:uppercase;letter-spacing:.9px;">Activity Log</span>
          <button onclick="clearLog()" style="background:none;border:none;color:var(--lo);font-size:9px;cursor:pointer;">clear</button>
        </div>
        <div id="log-entries" style="font-size:10px;color:var(--med);line-height:1.6;min-height:14px;font-family:monospace;">
          <span style="color:var(--lo);">Click ▶ on any card to run an analysis</span>
        </div>
      </div>

      <!-- ── Time-Frequency ── -->
      <div class="sec-lbl">Time-Frequency Analysis</div>
      <div id="cards-tf"></div>

      <!-- ── Spectral ── -->
      <div class="sec-lbl">Spectral Analysis</div>
      <div id="cards-spectral"></div>

      <!-- ── Coupling (2ch self-paired) ── -->
      <div class="sec-lbl">Coupling Analysis <span class="badge-2ch">self-paired</span></div>
      <div id="cards-coupling"></div>

      <!-- ── Statistics ── -->
      <div class="sec-lbl">Statistical</div>
      <div id="cards-stats"></div>

      <!-- Export -->
      <div id="export-section" style="display:none;">
        <div class="sec-lbl">Export</div>
        <div style="display:flex;gap:8px;">
          <button class="btn btn-out" onclick="exportCsv()" style="font-size:12px;padding:10px;">Signal CSV</button>
          <button class="btn btn-out" onclick="exportJson()" style="font-size:12px;padding:10px;">Last Result JSON</button>
        </div>
      </div>
    </div>

    <!-- History -->
    <div id="sub-history" class="sub-panel">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span id="hist-count" style="font-size:11px;color:var(--lo)">No sessions yet</span>
        <button onclick="clearHistory()" style="background:none;border:none;color:#F44336;font-size:11px;cursor:pointer;">Clear</button>
      </div>
      <div id="hist-list"></div>
    </div>
  </div>

  <!-- SETTINGS -->
  <div id="tab-settings" class="tab">
    <div class="app-bar">Settings</div>
    <div class="card">
      <div class="lbl">Server</div>
      <div class="set-row"><div class="set-lbl">FastMODA URL</div>
        <input class="set-inp" id="inp-fastmoda" type="text" value="http://localhost:5000" onchange="saveSettings()"></div>
      <div class="set-row"><div class="set-lbl">Signal Generator URL</div>
        <input class="set-inp" id="inp-signal" type="text" value="http://localhost:8090" onchange="saveSettings()"></div>
    </div>
    <div class="card" style="margin-top:10px;">
      <div class="lbl">Signal Generation</div>
      <div class="set-row"><div class="set-lbl">Sample Rate</div>
        <select class="set-inp" id="inp-fs" onchange="saveSettings()">
          <option value="128">128 Hz</option><option value="256" selected>256 Hz</option>
          <option value="512">512 Hz</option>
        </select></div>
      <div class="set-row"><div class="set-lbl">Noise Level: <span id="noise-lbl">0.20</span></div>
        <input type="range" min="0" max="2" step="0.05" value="0.2"
               style="width:100%;accent-color:var(--pri);"
               oninput="document.getElementById('noise-lbl').textContent=parseFloat(this.value).toFixed(2)"
               onchange="saveSettings()"></div>
    </div>
    <div class="card" style="margin-top:10px;">
      <div class="lbl">About</div>
      <div style="font-size:11px;color:var(--med);line-height:1.7;">
        MODA App Emulator v2.0<br>
        Signal generator, FastMODA, and browser UI run as separate services.<br>
        All 17 analysis types supported.
      </div>
    </div>
  </div>

  </div><!-- screen-body -->

  <nav class="bottom-nav">
    <button class="nav-btn active" id="nav-dashboard" onclick="goTab('dashboard',this)">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/></svg>Dashboard
    </button>
    <button class="nav-btn" id="nav-ble" onclick="goTab('ble',this)">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.71 7.71 12 2h-1v7.59L6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 11 14.41V22h1l5.71-5.71-4.3-4.29 4.3-4.29zM13 5.83l1.88 1.88L13 9.59V5.83zm1.88 10.46L13 18.17v-3.76l1.88 1.88z"/></svg>Devices
    </button>
    <button class="nav-btn" id="nav-analysis" onclick="goTab('analysis',this)">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>Analysis
    </button>
    <button class="nav-btn" id="nav-settings" onclick="goTab('settings',this)">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.57 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>Settings
    </button>
  </nav>
</div><!-- screen -->
</div><!-- phone -->

<script>
// ── Analysis card definitions ─────────────────────────────────────────────────
const CARD_GROUPS = {
  'cards-tf': [
    {id:'stft',    title:'STFT',            sub:'Short-Time Fourier Transform',       icon:'📉'},
    {id:'wft',     title:'WFT',             sub:'Gaussian-windowed (optimal TF)',      icon:'🌊'},
    {id:'cwt',     title:'CWT (lognorm)',   sub:'Lognormal wavelet — MODA default',   icon:'🔊', params:{wavelet:'lognorm'}},
    {id:'cwt',     title:'CWT (morlet)',    sub:'Morlet wavelet',                      icon:'〰️', params:{wavelet:'morlet'}, unique:'cwt_morlet'},
    {id:'cwt',     title:'CWT (bump)',      sub:'Bump wavelet — compact support',      icon:'⛰️', params:{wavelet:'bump'},   unique:'cwt_bump'},
    {id:'ridge',   title:'Ridge Extraction',sub:'Instantaneous freq / amp / phase',    icon:'🏔️'},
    {id:'hilbert', title:'Hilbert Phase',   sub:'Analytic signal decomposition',       icon:'🔄'},
    {id:'butter',  title:'Butterworth Filter',sub:'8–12 Hz bandpass + detrend',        icon:'🎛️'},
  ],
  'cards-spectral': [
    {id:'analyze', title:'Signal Analysis', sub:'MODWT + changepoints + band power',  icon:'📊'},
    {id:'modwt',   title:'MODWT',           sub:'Maximal Overlap DWT decomposition',   icon:'🌊'},
    {id:'bispectrum',title:'Bispectrum',    sub:'Quadratic phase coupling',            icon:'⊞'},
    {id:'bispec4', title:'4-Way Bispectrum',sub:'b111 / b222 / b122 / b211',          icon:'⊟', twoChannel:true},
    {id:'biphase', title:'Biphase Series',  sub:'Time-resolved biphase (6+10 Hz)',     icon:'📐', twoChannel:true, params:{f1:6,f2:10}},
  ],
  'cards-coupling': [
    {id:'coherence',title:'Phase Coherence',  sub:'Wavelet phase synchrony',           icon:'🔗', twoChannel:true},
    {id:'bayesian', title:'Bayesian Inference',sub:'Coupling strength + direction',    icon:'🧠', twoChannel:true},
    {id:'coupling', title:'Coupling Functions',sub:'q₂₁/q₁₂ via Fourier OLS',        icon:'↔️', twoChannel:true},
    {id:'syncmap',  title:'Sync Map',         sub:'1:1 phase-locking detection',       icon:'🔒', twoChannel:true},
  ],
  'cards-stats': [
    {id:'surrogates',title:'Surrogate Test', sub:'Phase-rand significance testing',   icon:'🔬'},
    {id:'features',  title:'Feature Extraction',sub:'33-feature ML vector',           icon:'📋'},
  ],
};

// ── State ─────────────────────────────────────────────────────────────────────
let streaming=false, serverOk=false;
let lastMetrics=null, lastResult=null, pendingTask=null;
let busyMap={};
let history=[];

// ── Charts ─────────────────────────────────────────────────────────────────────
const noAnim={animation:false,responsive:true,maintainAspectRatio:false,
              plugins:{legend:{display:false},tooltip:{enabled:false}}};
const scaleOpts={x:{display:false},y:{display:false,grid:{display:false}}};

function lineChart(id,color){
  const ctx=document.getElementById(id);if(!ctx)return null;
  return new Chart(ctx,{type:'line',data:{labels:[],datasets:[{data:[],
    borderColor:color||'#4DB6AC',borderWidth:1.5,pointRadius:0,fill:false}]},
    options:{...noAnim,scales:scaleOpts}});
}
function barChart(id,color){
  const ctx=document.getElementById(id);if(!ctx)return null;
  return new Chart(ctx,{type:'bar',data:{labels:[],datasets:[{data:[],
    backgroundColor:color||'rgba(0,137,123,.65)',borderRadius:2,borderWidth:0}]},
    options:{...noAnim,scales:scaleOpts}});
}

const BANDS=[
  {key:'delta',label:'Delta',hz:'0.5–4 Hz', color:'#7C4DFF'},
  {key:'theta',label:'Theta',hz:'4–8 Hz',   color:'#2196F3'},
  {key:'alpha',label:'Alpha',hz:'8–12 Hz',  color:'#00BCD4'},
  {key:'beta', label:'Beta', hz:'12–30 Hz', color:'#FF9800'},
  {key:'gamma',label:'Gamma',hz:'30–100 Hz',color:'#F44336'},
];

function renderBands(id){
  const el=document.getElementById(id);if(!el)return;
  el.innerHTML=BANDS.map(b=>`
    <div class="band-row">
      <div class="band-dot" style="background:${b.color}"></div>
      <div class="band-info"><div class="band-name">${b.label}</div><div class="band-hz">${b.hz}</div></div>
      <div class="bar-wrap"><div class="bar" id="bar-${id}-${b.key}" style="background:${b.color};width:0%"></div></div>
      <div class="band-pct" id="pct-${id}-${b.key}">0%</div>
    </div>`).join('');
}

let wDash,wAnalysis,specC,psdC;
function initCharts(){
  wDash=lineChart('wave-dash');
  wAnalysis=lineChart('wave-analysis');
  specC=barChart('spec-chart');
  psdC=barChart('psd-chart');
  renderBands('bands-dash');
  renderBands('bands-spectral');
}

// ── Card definition lookup (avoids JSON-in-HTML-attribute encoding bug) ────────
const _cardDefs = {};

function buildCards(){
  for(const [containerId,cards] of Object.entries(CARD_GROUPS)){
    const el=document.getElementById(containerId);
    if(!el)continue;
    el.innerHTML=cards.map(c=>{
      const uid=c.unique||c.id;
      _cardDefs[uid]=c;                        // store here, NOT in onclick attribute
      return `<div class="a-card" id="card-${uid}">
        <div class="a-hdr">
          <span class="a-ico">${c.icon}</span>
          <div class="a-text">
            <div class="a-title">${c.title}${c.twoChannel?'<span class="badge-2ch">self-paired</span>':''}</div>
            <div class="a-sub" id="sub-${uid}">${c.sub}</div>
          </div>
          <button class="a-btn" id="btn-${uid}" onclick="runCard('${uid}')">&#9654;</button>
        </div>
        <div class="a-result" id="res-${uid}"></div>
      </div>`;
    }).join('');
  }
}

// ── Activity log ──────────────────────────────────────────────────────────────
const _logs=[];
function logEntry(msg, color){
  const ts=new Date().toLocaleTimeString('en-GB',{hour12:false});
  _logs.unshift({ts, msg, color:color||'var(--med)'});
  if(_logs.length>12)_logs.pop();
  const el=document.getElementById('log-entries');
  if(!el)return;
  el.innerHTML=_logs.map(e=>
    `<span style="color:var(--lo);">${e.ts}</span> <span style="color:${e.color};">${e.msg}</span><br>`
  ).join('');
}
function clearLog(){_logs.length=0;const el=document.getElementById('log-entries');if(el)el.innerHTML='<span style="color:var(--lo);">Cleared</span>';}
function resetAll(){
  Object.keys(busyMap).forEach(k=>{
    busyMap[k]=false;
    const b=document.getElementById('btn-'+k);
    if(b){b.disabled=false;b.innerHTML='&#9654;';}
  });
  pendingTask=null;
  logEntry('All cards reset','#FF9800');
}

// ── UI update ──────────────────────────────────────────────────────────────────
function set(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}

function updateAll(m){
  if(!m)return; lastMetrics=m; streaming=!!m.streaming;
  const ld=m.chart_data||[];
  [wDash,wAnalysis].forEach(c=>{if(!c)return;
    c.data.labels=ld.map((_,i)=>i);c.data.datasets[0].data=ld;c.update('none');});
  if(m.spec_mags){
    [specC,psdC].forEach(c=>{if(!c)return;
      c.data.labels=m.spec_freqs;c.data.datasets[0].data=m.spec_mags;c.update('none');});
  }
  const powers=BANDS.map(b=>m[b.key]||0);
  const mx=Math.max(...powers,1e-12);
  BANDS.forEach((b,i)=>{
    const pct=Math.round(powers[i]/mx*100);
    ['bands-dash','bands-spectral'].forEach(cid=>{
      const bar=document.getElementById('bar-'+cid+'-'+b.key);
      const pe=document.getElementById('pct-'+cid+'-'+b.key);
      if(bar)bar.style.width=pct+'%';if(pe)pe.textContent=pct+'%';
    });
  });
  set('m-fs',(m.sample_rate||256).toFixed(0)+' Hz');
  set('m-dom',m.dominant!=null?m.dominant.toFixed(1)+' Hz':'— Hz');
  set('m-qual',m.quality!=null?m.quality.toFixed(0)+'%':'—%');
  set('m-entr',m.entropy!=null?(m.entropy*100).toFixed(1)+'%':'—');
  set('m-flat',m.flatness!=null?(m.flatness*100).toFixed(1)+'%':'—');
  set('m-entropy2',m.entropy!=null?(m.entropy*100).toFixed(1)+'%':'—');
  set('m-flatness2',m.flatness!=null?(m.flatness*100).toFixed(1)+'%':'—');
  set('cfg-fs',(m.sample_rate||256).toFixed(0)+' Hz');
  const cpEl=document.getElementById('cp-display');
  if(cpEl){
    const cps=m.changepoints||[];
    cpEl.innerHTML=cps.length>0
      ?'<div class="cp-wrap">'+cps.map(t=>`<span class="cp-chip">${t}s</span>`).join('')+'</div>'
      :(m.total_samples>64?'<div style="font-size:11px;color:var(--lo);padding:8px 0;text-align:center;">No changepoints</div>'
        :'<div style="font-size:11px;color:var(--lo);padding:8px 0;text-align:center;">Collecting…</div>');
  }
  const badge=document.getElementById('ble-badge');
  const btxt=document.getElementById('ble-status-txt');
  if(badge)badge.style.display=streaming?'':'none';
  if(btxt)btxt.textContent=streaming?'MODA-SIM-001 connected':'No device connected';
  const btn=document.getElementById('stream-btn');
  const stxt=document.getElementById('stream-status');
  if(btn){btn.textContent=streaming?'Stop Streaming':'Start Streaming';
          btn.style.background=streaming?'#8B0000':'';}
  if(stxt){stxt.textContent=streaming?'Streaming • '+(m.total_samples||0)+' samples':'Status: Idle';
           stxt.style.color=streaming?'#4CAF50':'';}
  if(pendingTask)pollTask(pendingTask);
}

function updateServerDots(ok){
  serverOk=ok;
  const cls=ok===null?'sdot unknown':(ok?'sdot up':'sdot down');
  ['sdot-dash','sdot-analysis'].forEach(id=>{const e=document.getElementById(id);if(e)e.className=cls;});
  const icon=document.getElementById('srv-icon');
  const txt=document.getElementById('srv-txt');
  const urlEl=document.getElementById('srv-url');
  if(icon)icon.textContent=ok===null?'⚫':(ok?'🟢':'🔴');
  if(txt)txt.textContent=ok===null?'Checking…':(ok?'Server connected':'Server not reachable');
  if(urlEl){
    const inp=document.getElementById('inp-fastmoda');
    urlEl.textContent=inp?inp.value:'';
  }
}

// ── Poll loop ─────────────────────────────────────────────────────────────────
async function fetchMetrics(){
  try{const r=await fetch('/api/metrics');updateAll(await r.json());}catch(_){}
}
async function checkHealth(){
  updateServerDots(null);
  try{const r=await fetch('/api/health');const d=await r.json();updateServerDots(d.ok);}
  catch(_){updateServerDots(false);}
}

// ── Analysis runner ──────────────────────────────────────────────────────────
async function runCard(uid){
  const cardDef=_cardDefs[uid];
  if(!cardDef){logEntry('Unknown card: '+uid,'#F44336');return;}
  if(busyMap[uid]){logEntry(cardDef.title+' already running','#FF9800');return;}
  busyMap[uid]=true;

  const resEl=document.getElementById('res-'+uid);
  const btnEl=document.getElementById('btn-'+uid);
  const subEl=document.getElementById('sub-'+uid);

  if(btnEl){btnEl.disabled=true;btnEl.innerHTML='<span class="spin"></span>';}
  if(resEl){
    resEl.className='a-result show';
    resEl.innerHTML='<div id="stage-'+uid+'" style="font-size:11px;color:var(--lo);padding:2px 0;">Submitting to FastMODA…</div>'
      +'<div class="prog-wrap"><div class="prog-bar" id="prog-'+uid+'" style="width:5%"></div></div>';
  }
  if(subEl) subEl.style.color='var(--pril)';
  logEntry('Running: '+cardDef.title);

  try{
    const payload={type:cardDef.id,...(cardDef.params||{})};
    const r=await fetch('/api/analyze',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)});

    if(!r.ok){
      const txt=await r.text();
      throw new Error('HTTP '+r.status+': '+txt.slice(0,80));
    }
    const d=await r.json();
    if(d.error){
      logEntry('Error: '+d.error.slice(0,60),'#F44336');
      showResult(resEl,{error:d.error});resetBtn(btnEl,uid,subEl);busyMap[uid]=false;return;
    }
    if(d.task_id){
      logEntry('Task '+d.task_id.slice(0,8)+'… polling');
      pendingTask={id:d.task_id,uid,resEl,btnEl,subEl,title:cardDef.title};
      setStage(uid,'Queued…',0);
    }else{
      showResult(resEl,d);resetBtn(btnEl,uid,subEl);busyMap[uid]=false;
      logEntry('Done: '+cardDef.title,'#4CAF50');
    }
  }catch(e){
    logEntry('Failed: '+String(e).slice(0,70),'#F44336');
    showResult(resEl,{error:String(e)});resetBtn(btnEl,uid,subEl);busyMap[uid]=false;
  }
}

function setStage(uid, stage, pct){
  const stEl=document.getElementById('stage-'+uid);
  const prEl=document.getElementById('prog-'+uid);
  if(stEl) stEl.textContent=stage;
  if(prEl && pct!=null) prEl.style.width=Math.max(5,pct)+'%';
}

async function pollTask(pt){
  let d;
  try{
    const r=await fetch('/api/status/'+pt.id);
    if(!r.ok){logEntry('Poll HTTP '+r.status,'#FF9800');return;}
    d=await r.json();
  }catch(e){
    logEntry('Poll error: '+String(e).slice(0,50),'#F44336');
    return;
  }

  const s=d.status||'';
  const prog=d.progress!=null?d.progress:null;
  const stage=d.stage||'Processing…';

  setStage(pt.uid, stage, prog);

  if(s==='complete'||s==='done'||s==='success'){
    showResult(pt.resEl,d);
    resetBtn(pt.btnEl,pt.uid,pt.subEl);
    lastResult=d; pendingTask=null; busyMap[pt.uid]=false;
    document.getElementById('export-section').style.display='';
    addHistory(pt.uid,d);
    logEntry('Complete: '+pt.title+' ('+stage+')','#4CAF50');
  }else if(s==='error'||s==='failed'){
    const errMsg=d.error||'Server error';
    showResult(pt.resEl,{error:errMsg});
    resetBtn(pt.btnEl,pt.uid,pt.subEl);
    pendingTask=null; busyMap[pt.uid]=false;
    logEntry('Error: '+pt.title+' — '+errMsg.slice(0,60),'#F44336');
  }else{
    logEntry(pt.title+': '+stage+(prog!=null?' ('+prog+'%)':''));
  }
}

function showResult(el,data){
  if(!el)return;
  const skip=new Set(['status','results','signal_plot','result','_plot_bispec4',
                       'coupling_plot','bispectrum_plot','direction_plot']);
  const rows=Object.entries(data)
    .filter(([k,v])=>!skip.has(k)&&(typeof v==='number'||typeof v==='boolean'||
            (typeof v==='string'&&v.length<100&&!v.startsWith('{'))))
    .slice(0,10)
    .map(([k,v])=>`<div class="r-row"><span class="r-key">${k}:</span><span class="r-val">${v}</span></div>`)
    .join('');
  el.className='a-result show';
  el.innerHTML=rows||'<div style="font-size:11px;color:var(--lo);padding:4px 0;">Analysis complete</div>';
}

function resetBtn(btn,uid,subEl){
  if(btn){btn.disabled=false;btn.innerHTML='&#9654;';}
  if(subEl) subEl.style.color='';
}

// ── Navigation ────────────────────────────────────────────────────────────────
function goTab(name,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if(btn)btn.classList.add('active');
}
function subTab(name,el){
  document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('sub-'+name).classList.add('active');
}

// ── Device controls ────────────────────────────────────────────────────────────
async function toggleStream(){
  await fetch(streaming?'/api/stream/stop':'/api/stream/start',{method:'POST'});
}
async function applyPreset(p){
  await fetch('/api/preset',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({preset:p})});
}

// ── Settings ──────────────────────────────────────────────────────────────────
async function saveSettings(){
  const fm=document.getElementById('inp-fastmoda').value||'';
  const sg=document.getElementById('inp-signal').value||'';
  const fs=document.getElementById('inp-fs').value;
  const noiseEl=document.querySelector('input[type=range]');
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fastmoda_url:fm,signal_url:sg,
                         sample_rate:parseFloat(fs),
                         noise:noiseEl?parseFloat(noiseEl.value):0.2})});
  checkHealth();
}

// ── Export ────────────────────────────────────────────────────────────────────
function exportCsv(){
  if(!lastMetrics?.chart_data)return;
  const fs=lastMetrics.sample_rate||256;
  const rows=['index,time_s,amplitude',
    ...lastMetrics.chart_data.map((v,i)=>`${i},${(i/fs).toFixed(4)},${v.toFixed(6)}`)];
  dl('moda_signal.csv',rows.join('\\n'));
}
function exportJson(){
  if(!lastResult)return;
  const out={};
  for(const[k,v]of Object.entries(lastResult)){
    if(typeof v==='string'&&v.length>500)continue;out[k]=v;
  }
  dl('moda_result.json',JSON.stringify(out,null,2));
}
function dl(name,text){
  const a=document.createElement('a');
  a.href='data:text/plain;charset=utf-8,'+encodeURIComponent(text);
  a.download=name;a.click();
}

// ── History ───────────────────────────────────────────────────────────────────
function addHistory(type,result){
  const res=result.results||result;
  const scalars=Object.fromEntries(Object.entries(res)
    .filter(([k,v])=>typeof v==='number'||typeof v==='boolean'));
  history.unshift({type,ts:new Date().toLocaleTimeString(),scalars});
  if(history.length>20)history.pop();
  renderHistory();
}
function renderHistory(){
  const el=document.getElementById('hist-list');
  if(!el)return;
  set('hist-count',history.length>0?history.length+' sessions':'No sessions yet');
  el.innerHTML=history.map((h,i)=>`
    <div class="a-card">
      <div class="a-hdr">
        <span class="a-ico">📋</span>
        <div class="a-text">
          <div class="a-title">${h.type.toUpperCase()}</div>
          <div class="a-sub">${h.ts}</div>
        </div>
      </div>
      ${Object.keys(h.scalars).length>0?`<div class="a-result show">`+
        Object.entries(h.scalars).slice(0,4).map(([k,v])=>
          `<div class="r-row"><span class="r-key">${k}:</span><span class="r-val">${typeof v==='number'?v.toFixed(4):v}</span></div>`
        ).join('')+'</div>':''}`
  ).join('');
}
function clearHistory(){history=[];renderHistory();}

// ── Boot ──────────────────────────────────────────────────────────────────────
initCharts();
buildCards();
fetch('/api/settings').then(r=>r.json()).then(d=>{
  const fm=document.getElementById('inp-fastmoda');
  const sg=document.getElementById('inp-signal');
  if(fm&&d.fastmoda_url)fm.value=d.fastmoda_url;
  if(sg&&d.signal_url)sg.value=d.signal_url;
  const urlEl=document.getElementById('srv-url');
  if(urlEl&&fm)urlEl.textContent=fm.value;
  checkHealth();
});
setTimeout(()=>fetch('/api/stream/start',{method:'POST'}),500);
setInterval(fetchMetrics,500);
setInterval(checkHealth,30000);
</script>
</body>
</html>"""


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    global _fastmoda_url, _signal_url

    parser = argparse.ArgumentParser(description="MODA App Emulator")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--fastmoda", default=None)
    parser.add_argument("--signal",   default=None)
    args = parser.parse_args()

    if args.fastmoda:
        _fastmoda_url = args.fastmoda.rstrip("/")
    if args.signal:
        _signal_url = args.signal.rstrip("/")

    print(f"""
╔══════════════════════════════════════════════════════╗
║            MODA App Emulator v2                      ║
╠══════════════════════════════════════════════════════╣
║  Browser UI        →  http://localhost:{args.port:<5}         ║
║  FastMODA          →  {_fastmoda_url:<33}║
║  Signal generator  →  {_signal_url:<33}║
╚══════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
