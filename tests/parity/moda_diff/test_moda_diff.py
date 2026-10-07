"""Direct numerical diff: real MODA (MATLAB) vs FastMODA's legacy path.

Reference outputs come from ``gen_moda_diff.m`` (needs a local MATLAB); without
them every case here is skipped. With them, each case runs the FastMODA code on
exactly MODA's inputs and measures

    rel = max|X - Y| / max|X|          over cells finite in both

the same metric ``tests/transform_parity`` uses, plus how far the NaN
(cone-of-influence) masks disagree.

Downstream stages (ridge, coherence) are fed **MODA's own WT**, so what they
measure is the downstream algorithm alone, not a difference inherited from the
transform. The bispectrum is end-to-end (FastMODA's WT and FastMODA's
algorithm) because ``bispecWavNew`` builds its own WTs internally.

Print the full table with:
    python tests/parity/moda_diff/test_moda_diff.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
EXACT = 1e-8          # "same answer" threshold for the report


def _load(path):
    from scipy.io import loadmat
    return loadmat(path, squeeze_me=True)


def relerr(x, y):
    """max|x-y| / max|x| over cells finite in both; (rel, nan-mask mismatch frac)."""
    x = np.asarray(x); y = np.asarray(y)
    both = np.isfinite(x) & np.isfinite(y)
    mism = float(np.mean(np.isfinite(x) != np.isfinite(y))) if x.size else 0.0
    if not both.any():
        return float("nan"), mism
    den = np.max(np.abs(x[both]))
    return float(np.max(np.abs(x[both] - y[both])) / (den if den else 1.0)), mism


def medrel(x, y):
    """median|x-y| / max|x| — tells a few bad cells apart from a wholesale shift."""
    x = np.asarray(x); y = np.asarray(y)
    both = np.isfinite(x) & np.isfinite(y)
    if not both.any():
        return float("nan")
    den = np.max(np.abs(x[both]))
    return float(np.median(np.abs(x[both] - y[both])) / (den if den else 1.0))


def _s(v):
    return str(v) if not isinstance(v, np.ndarray) else str(v.item())


# ── transforms ──────────────────────────────────────────────────────────────

def compare_transform(d):
    from fastmoda.legacy_moda import wt_legacy, wft_legacy
    fmin = float(d["fmin"]); fmax = float(d["fmax"])
    kw = dict(fmin=None if np.isnan(fmin) else fmin,
              fmax=None if np.isnan(fmax) else fmax,
              f0=float(d["f0"]), padding=_s(d["pad"]),
              preprocess=_s(d["pre"]) == "on", cut_edges=_s(d["cut"]) == "on")
    sig, fs = np.asarray(d["sig"], float), float(d["fs"])
    if _s(d["kind"]) == "wt":
        W, fr = wt_legacy(sig, fs, wavelet=_s(d["kernel"]), **kw)
    else:
        W, fr = wft_legacy(sig, fs, window=_s(d["kernel"]), **kw)
    M, mf = np.atleast_2d(d["WT"]), np.atleast_1d(d["freq"]).astype(float)
    out = {"bins_moda": len(mf), "bins_fm": len(fr)}
    # Compare on the frequencies both grids share; a bin-count mismatch is
    # reported separately rather than silently trimmed.
    common = [(i, j) for i, f in enumerate(mf)
              for j in np.flatnonzero(np.isclose(fr, f, rtol=1e-9, atol=0))]
    if not common:
        out.update(rel=float("nan"), freq_rel=float("nan"), nanmask=float("nan"))
        return out
    im, jf = map(np.array, zip(*common))
    out["freq_rel"] = float(np.max(np.abs(mf[im] - fr[jf]) / mf[im]))
    out["rel"], out["nanmask"] = relerr(M[im], W[jf])
    out["med"] = medrel(M[im], W[jf])
    return out


def transform_cases():
    return sorted(glob.glob(os.path.join(REF, "wt_*.mat")) +
                  glob.glob(os.path.join(REF, "wft_*.mat")))


def case_id(path):
    d = _load(path)
    b = "default" if np.isnan(float(d["fmin"])) else "0.3-8"
    return (f"{_s(d['kind'])}|{_s(d['kernel'])}|f0={float(d['f0']):g}|pre={_s(d['pre'])}"
            f"|pad={_s(d['pad'])}|cut={_s(d['cut'])}|band={b}")


# ── downstream ──────────────────────────────────────────────────────────────

def compare_ridge(d):
    from fastmoda.ridge_gpu import extract_ridge
    W, fr, fs = d["W1"], np.asarray(d["freq"], float), float(d["fs"])
    out = {}
    for smooth in (0, 5):          # 5 = /analyze_ridge default
        r = extract_ridge(W, fr, fs, smooth_len=smooth)
        for k, a, b in (("ifreq", d["ifreq"], r["ifreq"]),
                        ("iamp", d["iamp"], r["iamp"]),
                        ("iphi", np.exp(1j * d["iphi"]), np.exp(1j * r["iphi"]))):
            out[f"{k}_s{smooth}_max"] = relerr(a, b)[0]
            out[f"{k}_s{smooth}_med"] = medrel(a, b)
    return out


def _app():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "FastMODA"))
    import app
    return app


def compare_coherence(d):
    from fastmoda.ridge_gpu import time_localized_coherence
    W1, W2 = d["W1"], d["W2"]
    fr, fs = np.asarray(d["freq"], float), float(d["fs"])
    tpc = time_localized_coherence(W1, W2, fr, fs, numcycles=10)
    # /analyze_coherence, legacy=false: mean of the local TPC, amplitude-
    # weighted phase difference
    with np.errstate(all="ignore"):
        phcoh_fast = np.nanmean(tpc, axis=1)
        phdiff_fast = np.angle(np.nanmean(W1 * np.conj(W2), axis=1))
    # /analyze_coherence, legacy=true: the wphcoh port
    phcoh_leg, phdiff_leg = _app()._wphcoh(W1, W2)
    u = lambda p: np.exp(1j * np.asarray(p))
    return {
        "tlphcoh_vs_TPC": relerr(d["TPC"], tpc)[0],
        "tlphcoh_nanmask": relerr(d["TPC"], tpc)[1],
        "phcoh_legacy_vs_wphcoh": relerr(d["phcoh"], phcoh_leg)[0],
        "phdiff_legacy_vs_wphcoh": relerr(u(d["phdiff"]), u(phdiff_leg))[0],
        "phcoh_fast_vs_wphcoh_max": relerr(d["phcoh"], phcoh_fast)[0],
        "phcoh_fast_vs_wphcoh_med": medrel(d["phcoh"], phcoh_fast),
        "phdiff_fast_vs_wphcoh_max": relerr(u(d["phdiff"]), u(phdiff_fast))[0],
        "phdiff_fast_vs_wphcoh_med": medrel(u(d["phdiff"]), u(phdiff_fast)),
    }


def compare_bispectrum(d):
    _wavelet_bispectrum_legacy = _app()._wavelet_bispectrum_legacy
    s1, s2, fs = np.asarray(d["sig1"], float), np.asarray(d["sig2"], float), float(d["fs"])
    mf = np.asarray(d["bfreq"], float)
    r = _wavelet_bispectrum_legacy([s1, s2], fs, 0.3, 8.0, 64, "122", f0=1.0)
    ff = np.asarray(r["freq"], float)
    idx = np.array([int(np.argmin(np.abs(mf - f))) for f in ff])
    same_grid = bool(np.allclose(mf[idx], ff, rtol=1e-9))
    M = np.asarray(d["Bisp"])[np.ix_(idx, idx)]
    B = np.asarray(r["bispectrum"])
    B = np.where(np.isfinite(M), B, np.nan)          # MODA's valid region only
    rel_all, _ = relerr(M, B)
    # cells MODA computes but FastMODA zeroes (its f1+f2 <= fmax guard differs)
    zeroed = float(np.mean((B == 0) & np.isfinite(M) & (M != 0)))
    return {"bins_moda": len(mf), "bins_fm": len(ff), "grid_subset": same_grid,
            "rel": rel_all, "med": medrel(M, B), "zeroed_frac": zeroed}


# ── pytest ──────────────────────────────────────────────────────────────────

needs_ref = pytest.mark.skipif(not os.path.isdir(REF) or not os.listdir(REF),
                               reason="no MODA reference — run gen_moda_diff.m")


# Known, documented gaps (docs/validation/changelog-vs-moda.md). Each is an
# xfail so the suite stays green on them but flips to XPASS — and gets
# noticed — the moment one is closed.
FCAST = pytest.mark.xfail(strict=True, reason="predictive padding: fcast not ported")
FMIN = pytest.mark.xfail(strict=True, reason="default fmin: sqeps support approximated")
WIN = pytest.mark.xfail(strict=True, reason="compact-support WFT window: fixed-grid "
                                            "integration in place of MODA's quadgk")


def _transform_param(p):
    d = _load(p)
    marks = []
    if _s(d["pad"]) == "predictive":
        marks.append(FCAST)
    elif np.isnan(float(d["fmin"])):
        marks.append(FMIN)
    elif _s(d["kind"]) == "wft" and _s(d["kernel"]) not in ("Gaussian", "Exp"):
        marks.append(WIN)
    return pytest.param(p, marks=marks, id=case_id(p))


@needs_ref
@pytest.mark.parametrize("path", [_transform_param(p) for p in transform_cases()])
def test_transform_matches_moda(path):
    r = compare_transform(_load(path))
    assert r["bins_moda"] == r["bins_fm"], r
    assert r["freq_rel"] < 1e-12, r
    assert r["rel"] < EXACT, r


@needs_ref
@pytest.mark.xfail(strict=True, reason="ridge path is argmax, not MODA's ecurve")
def test_ridge_on_moda_wt():
    r = compare_ridge(_load(os.path.join(REF, "ridge.mat")))
    assert r["ifreq_s0_max"] < EXACT, r


@needs_ref
def test_coherence_legacy_matches_moda():
    r = compare_coherence(_load(os.path.join(REF, "coherence.mat")))
    assert r["tlphcoh_vs_TPC"] < EXACT, r
    assert r["tlphcoh_nanmask"] == 0, r
    assert r["phcoh_legacy_vs_wphcoh"] < EXACT, r
    assert r["phdiff_legacy_vs_wphcoh"] < EXACT, r


@needs_ref
@pytest.mark.xfail(strict=True, reason="nearest-bin f1+f2, not bispecWavNew's exact WT at f1+f2")
def test_bispectrum_matches_moda():
    r = compare_bispectrum(_load(os.path.join(REF, "bispectrum.mat")))
    assert r["rel"] < EXACT, r


# ── report ──────────────────────────────────────────────────────────────────

def report():
    def flag(v):
        return "" if not np.isfinite(v) or v <= EXACT else "  <-- > 1e-8"
    print(f"{'case':66s} {'bins M/F':>9s} {'max rel':>10s} {'med rel':>10s} {'nanmask':>8s}")
    for p in transform_cases():
        r = compare_transform(_load(p))
        print(f"{case_id(p):66s} {r['bins_moda']:4d}/{r['bins_fm']:<4d} "
              f"{r['rel']:10.2e} {r['med']:10.2e} {r['nanmask']:8.3f}{flag(r['rel'])}")
    for name, fn in (("ridge", compare_ridge), ("coherence", compare_coherence),
                     ("bispectrum", compare_bispectrum)):
        print(f"\n[{name}]")
        for k, v in fn(_load(os.path.join(REF, f"{name}.mat"))).items():
            print(f"  {k:32s} {v!s:>12}" + (flag(v) if isinstance(v, float) else ""))


if __name__ == "__main__":
    report()
