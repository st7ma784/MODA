#!/usr/bin/env python3
"""
MODA Signal Generator Service
==============================
Standalone HTTP service that generates continuous synthetic EEG/biomedical
signal data and exposes it for consumption by the browser emulator, tests,
or any other client.

Runs independently of the emulator and FastMODA so each can be scaled,
replaced, or updated without touching the others.

Usage
-----
    python signal_generator.py [--port 8090] [--fs 256]

API
---
    GET  /health           — {"status":"ok","streaming":bool,"preset":str}
    GET  /metrics          — DSP metrics: waveform, bands, entropy, flatness, changepoints
    GET  /signal.npy       — Current buffer as NumPy float32 .npy (for analysis submission)
    POST /stream/start     — Begin signal generation
    POST /stream/stop      — Stop signal generation
    POST /preset           — {"preset":"resting"|"active"|"drowsy"|"sleep"|"noise"}
    POST /settings         — {"sample_rate":256,"noise":0.2}
"""

import argparse
import io
import math
import random
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, Response

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_FS   = 256.0
BUFFER_SIZE  = 512

PRESETS = {
    "resting": dict(alpha=1.0, theta=0.3, beta=0.12, delta=0.10, gamma=0.05, noise=0.20),
    "active":  dict(alpha=0.3, theta=0.2, beta=0.80, delta=0.05, gamma=0.20, noise=0.30),
    "drowsy":  dict(alpha=0.5, theta=0.9, beta=0.05, delta=0.30, gamma=0.02, noise=0.15),
    "sleep":   dict(alpha=0.1, theta=0.3, beta=0.04, delta=1.20, gamma=0.02, noise=0.10),
    "noise":   dict(alpha=0.1, theta=0.1, beta=0.10, delta=0.10, gamma=0.10, noise=1.50),
}

# ── signal state ───────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self.buf       = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.head      = 0
        self.total     = 0
        self.fs        = DEFAULT_FS
        self.streaming = False
        self.preset    = "resting"
        self._lock     = threading.Lock()
        self._thread   = None
        self.apply_preset("resting")

    def apply_preset(self, name: str):
        p = PRESETS.get(name, PRESETS["resting"])
        self.alpha = p["alpha"]; self.theta = p["theta"]
        self.beta  = p["beta"];  self.delta = p["delta"]
        self.gamma = p["gamma"]; self.noise = p["noise"]
        self.preset = name

    def start(self):
        if self.streaming:
            return
        self.streaming = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.streaming = False

    def _run(self):
        chunk = 16
        while self.streaming:
            t0 = self.total / self.fs
            samples = []
            for i in range(chunk):
                t = t0 + i / self.fs
                s = (self.alpha * math.sin(2 * math.pi * 10.0 * t)
                     + self.theta * math.sin(2 * math.pi *  6.0 * t)
                     + self.beta  * math.sin(2 * math.pi * 18.0 * t)
                     + self.delta * math.sin(2 * math.pi *  2.0 * t)
                     + self.gamma * math.sin(2 * math.pi * 40.0 * t)
                     + self.noise * random.gauss(0, 1))
                samples.append(s)
            with self._lock:
                for s in samples:
                    self.buf[self.head % BUFFER_SIZE] = s
                    self.head += 1
                self.total += chunk
            time.sleep(chunk / self.fs)

    def recent(self, n=None):
        with self._lock:
            count = min(self.total, BUFFER_SIZE)
            if self.total < BUFFER_SIZE:
                data = self.buf[:count].copy()
            else:
                pos  = self.head % BUFFER_SIZE
                data = np.concatenate([self.buf[pos:], self.buf[:pos]])
        return data[-n:] if (n and len(data) >= n) else data


sig = _State()

# ── DSP ────────────────────────────────────────────────────────────────────────

def _metrics():
    data = sig.recent(256)
    if len(data) < 64:
        return None

    N    = len(data)
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / max(N - 1, 1)))
    fft  = np.fft.rfft(data * hann)
    mags = np.abs(fft) / N
    freqs = np.fft.rfftfreq(N, 1.0 / sig.fs)

    def band(lo, hi):
        return float(np.sum(mags[(freqs >= lo) & (freqs < hi)] ** 2))

    mags_ndc = mags.copy(); mags_ndc[0] = 0
    dom_idx  = int(np.argmax(mags_ndc))
    dominant = float(freqs[dom_idx])

    total   = float(np.sum(mags ** 2))
    mean    = total / max(len(mags), 1)
    quality = min(100.0, float(mags_ndc[dom_idx]) ** 2 / max(mean, 1e-12) / max(len(mags), 1) * 100)

    eps  = 1e-12
    spec = mags[1:]
    tot  = float(np.sum(spec ** 2))
    entropy = 0.0
    if tot > eps:
        p = spec ** 2 / tot
        p = p[p > eps]
        entropy = float(-np.sum(p * np.log(p)) / max(math.log(len(spec)), eps))

    geo      = float(np.exp(np.mean(np.log(spec + eps))))
    arith    = float(np.mean(spec + eps))
    flatness = min(1.0, geo / arith) if arith > eps else 0.0

    # Changepoints via variance-ratio
    cps = _changepoints(data)

    # Waveform for chart (≤150 pts)
    recent_all = sig.recent()
    step = max(1, len(recent_all) // 150)
    chart = recent_all[::step][-150:].tolist()

    bins = min(60, len(mags) - 1)
    return {
        "chart_data":    chart,
        "spec_mags":     mags[1:bins+1].tolist(),
        "spec_freqs":    [round(float(f), 2) for f in freqs[1:bins+1]],
        "delta":   band(0.5,  4.0),  "theta": band(4.0,  8.0),
        "alpha":   band(8.0, 12.0),  "beta":  band(12.0, 30.0),
        "gamma":   band(30.0,100.0),
        "dominant":  round(dominant, 1),
        "quality":   round(quality,  1),
        "entropy":   round(entropy,  4),
        "flatness":  round(float(flatness), 4),
        "changepoints": [round(i / sig.fs, 2) for i in cps],
        "sample_rate": sig.fs,
        "total_samples": sig.total,
        "streaming": sig.streaming,
        "preset": sig.preset,
    }


def _changepoints(data, W=32, threshold=3.0):
    n = len(data)
    if n < W * 2:
        return []
    step = max(1, W // 2)
    variances, starts = [], []
    for i in range(0, n - W + 1, step):
        variances.append(float(np.var(data[i:i+W])) + 1e-10)
        starts.append(i)
    cps = []
    for i in range(1, len(variances)):
        r = variances[i] / variances[i-1]
        if r > threshold or r < 1.0 / threshold:
            idx = starts[i]
            if not cps or idx - cps[-1] > W:
                cps.append(idx)
    return cps


def _npy_bytes(data):
    buf = io.BytesIO()
    np.save(buf, data.astype(np.float32))
    return buf.getvalue()


# ── Flask ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "streaming": sig.streaming,
                    "preset": sig.preset, "total_samples": sig.total})


@app.route("/metrics")
def metrics():
    m = _metrics()
    return jsonify(m or {"streaming": sig.streaming, "total_samples": sig.total,
                         "preset": sig.preset})


@app.route("/signal.npy")
def signal_npy():
    data = sig.recent(512)
    npy  = _npy_bytes(data)
    return Response(npy, mimetype="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=signal.npy"})


@app.route("/stream/start", methods=["POST"])
def stream_start():
    sig.start()
    return jsonify({"ok": True})


@app.route("/stream/stop", methods=["POST"])
def stream_stop():
    sig.stop()
    return jsonify({"ok": True})


@app.route("/preset", methods=["POST"])
def preset():
    name = (request.json or {}).get("preset", "resting")
    try:
        sig.apply_preset(name)
        return jsonify({"ok": True, "preset": name})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        d = request.json or {}
        if "sample_rate" in d:
            sig.fs = float(d["sample_rate"])
        if "noise" in d:
            sig.noise = float(d["noise"])
    return jsonify({"sample_rate": sig.fs, "noise": sig.noise,
                    "preset": sig.preset, "streaming": sig.streaming})


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MODA Signal Generator")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--fs",   type=float, default=256.0, help="Sample rate")
    parser.add_argument("--no-autostart", action="store_true")
    args = parser.parse_args()

    sig.fs = args.fs
    if not args.no_autostart:
        sig.start()

    print(f"\n  MODA Signal Generator → http://0.0.0.0:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
