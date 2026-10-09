"""Direct numerical diff: real MODA (MATLAB) vs FastMODA's legacy path.

Reference outputs come from ``gen_moda_diff.m`` (needs a local MATLAB); without
them every case here is skipped. With them, each case runs the FastMODA code on
exactly MODA's inputs and measures

    rel = max|X - Y| / max|X|          over cells finite in both

the same metric ``tests/transform_parity`` uses, plus how far the NaN
(cone-of-influence) masks disagree.

Downstream stages (ridge, coherence) are fed **MODA's own WT**, so what they
measure is the downstream algorithm alone, not a difference inherited from the
transform. The bispectrum is measured both ways: from MODA's own transforms
and padding (the algorithm alone) and end to end (FastMODA's transform too).

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

RIDGE_CASES = ("ridge", "ridge_morlet", "ridge_cut", "ridge_wft")


def compare_ridge(d):
    """ecurve + rectfr('direct') ports on MODA's own transform and constants."""
    from fastmoda.legacy_ridge import ecurve_legacy, rectfr_legacy
    W, fr, fs = d["W1"], np.asarray(d["freq"], float), float(d["fs"])
    D, omg = float(d["D"]), float(d["omg"])
    kw = dict(D=None if np.isnan(D) else D, omg=None if np.isnan(omg) else omg)
    ts = ecurve_legacy(W, fr, fs)
    iamp, iphi, ifreq = rectfr_legacy(ts, W, fr, fs, float(d["C"]), **kw)
    out = {}
    for k, a, b in (("ridge_freq", d["tfsupp"][0], ts[0]),
                    ("support_lo", d["tfsupp"][1], ts[1]),
                    ("support_hi", d["tfsupp"][2], ts[2]),
                    ("ifreq", d["ifreq"], ifreq), ("iamp", d["iamp"], iamp),
                    ("iphi", np.exp(1j * d["iphi"]), np.exp(1j * iphi))):
        out[k], out[k + "_nanmask"] = relerr(a, b)
    return out


def compare_ridge_argmax(d):
    """The fast path (legacy=false): per-sample argmax, |W| at one bin."""
    from fastmoda.ridge_gpu import extract_ridge
    W, fr, fs = d["W1"], np.asarray(d["freq"], float), float(d["fs"])
    r = extract_ridge(W, fr, fs, smooth_len=5)
    return {k + "_fast_max": relerr(a, b)[0]
            for k, a, b in (("ifreq", d["ifreq"], r["ifreq"]), ("iamp", d["iamp"], r["iamp"]))}


def compare_ridge_constants():
    """C and D as FastMODA computes them, against the ones wt.m stored."""
    from fastmoda.legacy_ridge import wavelet_constants
    out = {}
    for name, wav in (("ridge", "Lognorm"), ("ridge_morlet", "Morlet")):
        d = _load(os.path.join(REF, name + ".mat"))
        C, D = wavelet_constants(wav, 1.0)
        out[f"C_{wav}"] = abs(C - float(d["C"])) / float(d["C"])
        mD = float(d["D"])
        out[f"D_{wav}"] = (0.0 if np.isinf(D) and np.isinf(mD)
                           else abs(D - mD) / mD)
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


BISP_CASES = ("bispectrum_zero", "bispectrum_auto", "bispectrum_cut", "bispectrum")


def _bisp_kwargs(d):
    return dict(fmin=float(d["fmin"]), fmax=float(d["fmax"]), wavelet="Lognorm",
                f0=float(d["f0"]), padding=_s(d["pad"]),
                preprocess=_s(d["pre"]) == "on", cut_edges=_s(d["cut"]) == "on")


def compare_bispectrum(d):
    """bispecWavNew port. ``iso`` runs the bispectrum on MODA's own transforms,
    padding and wavelet support, so it measures the algorithm alone; ``e2e``
    is FastMODA's transform and FastMODA's bispectrum together."""
    from fastmoda.legacy_moda import bispec_wav_legacy, wt_at_freqs
    s1, s2, fs = np.asarray(d["sig1"], float), np.asarray(d["sig2"], float), float(d["fs"])
    M, mf = np.asarray(d["Bisp"]), np.asarray(d["bfreq"], float)
    B, fr, opt, _, _ = bispec_wav_legacy(s1, s2, fs, **_bisp_kwargs(d))
    out = {"bins_moda": len(mf), "bins_fm": len(fr)}
    if len(fr) != len(mf):
        return out
    out["e2e"], out["e2e_nanmask"] = relerr(M, B)
    out["e2e_med"] = medrel(M, B)
    both = np.isfinite(M) & np.isfinite(B)
    out["e2e_cells_off"] = float(np.mean(
        np.abs(M[both] - B[both]) > EXACT * np.max(np.abs(M[both]))))

    o = dict(opt, padleft=np.asarray(d["padleft"], float),
             padright=np.asarray(d["padright"], float),
             t1e=float(d["t1e"]), t2e=float(d["t2e"]))
    W1, W2 = d["WT1"], d["WT2"]
    nf = len(mf)
    Bi = np.full((nf, nf), np.nan, complex)
    auto = np.array_equal(W1, W2, equal_nan=True)
    for j in range(nf):
        k = np.arange(j if auto else 0, nf)
        f3 = mf[j] + mf[k]
        i3 = np.searchsorted(mf, f3, "left")
        v = (f3 <= mf[-1]) & (mf[np.maximum(i3 - 1, 0)] > mf[np.maximum(j, k)])
        k = k[v]
        if len(k):
            xx = W1[j][None, :] * W2[k] * np.conj(wt_at_freqs(s2, fs, f3[v], o))
            n = np.sum(~np.isnan(xx), axis=1)
            Bi[j, k] = np.where(n > 0, np.nansum(xx, axis=1) / np.maximum(n, 1), np.nan)
    out["iso"], out["iso_nanmask"] = relerr(M, Bi)
    return out


def compare_bispectrum_endpoint(d):
    """/analyze_bispectrum's own wrapper, as the MODA GUI calls bispecWavNew."""
    r = _app()._wavelet_bispectrum_legacy(
        [np.asarray(d["sig1"], float), np.asarray(d["sig2"], float)], float(d["fs"]),
        float(d["fmin"]), float(d["fmax"]), 64, "122", f0=float(d["f0"]))
    return {"bins_moda": len(np.atleast_1d(d["bfreq"])), "bins_fm": len(r["freq"])}


# ── pytest ──────────────────────────────────────────────────────────────────

needs_ref = pytest.mark.skipif(not os.path.isdir(REF) or not os.listdir(REF),
                               reason="no MODA reference — run gen_moda_diff.m")


# Every case is held to EXACT (1e-8): all three wavelets and all six windows,
# under every padding MODA offers, with CutEdges on and off. There are no
# expected failures left; a case that drifts past EXACT fails the suite.


@needs_ref
@pytest.mark.parametrize("path", [pytest.param(p, id=case_id(p)) for p in transform_cases()])
def test_transform_matches_moda(path):
    d = _load(path)
    r = compare_transform(d)
    assert r["bins_moda"] == r["bins_fm"], r
    assert r["freq_rel"] < 1e-12, r
    assert r["rel"] < EXACT, r
    assert r["nanmask"] == 0, r


def _run_endpoint(route, data):
    """POST to a FastMODA route through Flask's test client and wait for the job."""
    import io
    import json
    import time
    client = _app().app.test_client()
    form = dict(data)
    for k, v in list(form.items()):
        if isinstance(v, np.ndarray):
            buf = io.BytesIO(); np.save(buf, v); buf.seek(0)
            form[k] = (buf, k + ".npy")
    r = client.post(route, data=form, content_type="multipart/form-data")
    if r.status_code != 200:
        return r.status_code, r.get_json()
    tid = r.get_json()["task_id"]
    for _ in range(600):
        st = json.loads(client.get("/status/" + tid).data)
        if st.get("status") in ("complete", "error"):
            return 200, st
        time.sleep(0.2)
    return 200, st


@needs_ref
def test_wft_endpoint_is_the_moda_port():
    """/analyze_wft defaults to wft_legacy: MODA's frequency grid, and no guessed f0."""
    path = next(p for p in transform_cases()
                if _s(_load(p)["kind"]) == "wft" and _s(_load(p)["kernel"]) == "Hann"
                and _s(_load(p)["pad"]) == "zero" and _s(_load(p)["cut"]) == "off")
    d = _load(path)
    sig, fs = np.asarray(d["sig"], float), float(d["fs"])
    code, body = _run_endpoint("/analyze_wft", {"file": sig, "fs": fs})
    assert code == 400 and "f0" in body["error"], body
    code, st = _run_endpoint("/analyze_wft", {
        "file": sig, "fs": fs, "f0": float(d["f0"]), "window": "hann",
        "freq_min": float(d["fmin"]), "freq_max": float(d["fmax"]), "padding": "zero"})
    assert st.get("status") == "complete", st
    res = st["results"]
    mf = np.atleast_1d(d["freq"]).astype(float)
    assert res["method"] == "wft_legacy" and res["n_freq_bins"] == len(mf), res
    assert abs(res["fstep"] - (mf[1] - mf[0])) < 1e-12, res
    code, st = _run_endpoint("/analyze_wft", {"file": sig, "fs": fs, "legacy": "false"})
    assert st.get("status") == "complete" and st["results"].get("method") is None, st


@needs_ref
@pytest.mark.parametrize("name", RIDGE_CASES)
def test_ridge_matches_moda(name):
    r = compare_ridge(_load(os.path.join(REF, name + ".mat")))
    for k, v in r.items():
        assert (v == 0) if k.endswith("_nanmask") else (v < EXACT), (k, r)


@needs_ref
def test_ridge_constants_match_moda():
    for k, v in compare_ridge_constants().items():
        assert v < 1e-12, (k, v)


@needs_ref
def test_coherence_legacy_matches_moda():
    r = compare_coherence(_load(os.path.join(REF, "coherence.mat")))
    assert r["tlphcoh_vs_TPC"] < EXACT, r
    assert r["tlphcoh_nanmask"] == 0, r
    assert r["phcoh_legacy_vs_wphcoh"] < EXACT, r
    assert r["phdiff_legacy_vs_wphcoh"] < EXACT, r


@needs_ref
@pytest.mark.parametrize("name", BISP_CASES)
def test_bispectrum_matches_moda(name):
    d = _load(os.path.join(REF, name + ".mat"))
    r = compare_bispectrum(d)
    assert r["bins_moda"] == r["bins_fm"], r
    # the algorithm, on MODA's own transforms and padding
    assert r["iso"] < EXACT and r["iso_nanmask"] == 0, r
    # end to end
    assert r["e2e_nanmask"] == 0, r
    assert r["e2e"] < EXACT and r["e2e_cells_off"] == 0, r


@needs_ref
def test_bispectrum_endpoint_uses_moda_grid():
    r = compare_bispectrum_endpoint(_load(os.path.join(REF, "bispectrum.mat")))
    assert r["bins_moda"] == r["bins_fm"], r


# ── report ──────────────────────────────────────────────────────────────────

def report():
    def flag(v):
        return "" if not np.isfinite(v) or v <= EXACT else "  <-- > 1e-8"
    print(f"{'case':66s} {'bins M/F':>9s} {'max rel':>10s} {'med rel':>10s} {'nanmask':>8s}")
    for p in transform_cases():
        r = compare_transform(_load(p))
        print(f"{case_id(p):66s} {r['bins_moda']:4d}/{r['bins_fm']:<4d} "
              f"{r['rel']:10.2e} {r['med']:10.2e} {r['nanmask']:8.3f}{flag(r['rel'])}")
    sections = ([(n, compare_ridge) for n in RIDGE_CASES]
                + [("ridge", compare_ridge_argmax), ("coherence", compare_coherence)]
                + [(n, compare_bispectrum) for n in BISP_CASES])
    for name, fn in sections:
        print(f"\n[{name}] {fn.__name__}")
        for k, v in fn(_load(os.path.join(REF, f"{name}.mat"))).items():
            print(f"  {k:32s} {v!s:>12}" + (flag(v) if isinstance(v, float) else ""))
    print("\n[constants]")
    for k, v in compare_ridge_constants().items():
        print(f"  {k:32s} {v!s:>12}")


if __name__ == "__main__":
    report()
