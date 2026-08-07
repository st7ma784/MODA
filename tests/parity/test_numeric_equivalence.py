"""Task B — numerical equivalence of FastMODA outputs.

"Identical to MODA" cannot be asserted bit-for-bit across MATLAB and
Python/NumPy (different FFT libs, edge handling, float ordering). We therefore
validate on two fronts:

  1. **Ground-truth correctness** — synthetic signals with an *analytically
     known* answer (a 10 Hz tone must peak at 10 Hz, identical signals must be
     perfectly coherent, quadratically phase-coupled tones must raise
     bicoherence at the coupling pair, a band-pass must reject out-of-band
     energy). These run everywhere, no MATLAB needed, and pin FastMODA to the
     same mathematics MODA implements.

  2. **Direct MODA diff (opt-in)** — if reference outputs exist under
     ``tests/parity/reference/`` (generate them with
     ``gen_moda_reference.m`` in MATLAB), each is compared to the FastMODA
     result with correlation / RMSE tolerances. Absent references are skipped,
     so the suite is green on a machine without MATLAB but becomes a true
     cross-implementation diff wherever MATLAB is present.

Run (inside the FastMODA image):
    pytest -q /repo/tests/parity/test_numeric_equivalence.py
"""

import os

import numpy as np
import pytest

FS = 40.0          # matches the melanoma laser-Doppler protocol
N = 4096
T = np.arange(N) / FS

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "reference")


def _peak_freq(freqs, Sxx):
    """Dominant frequency of a [F, T] magnitude spectrogram (time-averaged)."""
    power = np.nanmean(np.abs(Sxx) ** 2, axis=1)
    return float(freqs[int(np.argmax(power))])


# ── 1. Sliding-FFT spectrogram peaks at the true tone frequency ─────────────
def test_sliding_fft_peak_frequency():
    from fastmoda import sliding_fft
    f0 = 10.0
    x = np.sin(2 * np.pi * f0 * T)
    freqs, times, Sxx = sliding_fft(x, fs=FS, win_s=1.0)
    assert abs(_peak_freq(freqs, Sxx) - f0) <= 1.0


# ── 2. Windowed Fourier Transform (MODA wft.m equivalent) ───────────────────
def test_wft_peak_frequency():
    from fastmoda.filtering import wft
    f0 = 8.0
    x = np.sin(2 * np.pi * f0 * T)
    freqs, times, Sxx = wft(x, fs=FS, window_size=256, hop_size=128,
                            window="gaussian")
    assert abs(_peak_freq(freqs, Sxx) - f0) <= 1.0


# ── 3. Continuous wavelet transform peaks at the true tone frequency ────────
def test_cwt_peak_frequency():
    from fastmoda.analysis_gpu import cwt_gpu
    f0 = 6.0
    x = np.sin(2 * np.pi * f0 * T)
    freqs, times, cwt_mag = cwt_gpu(x, fs=FS, freq_range=(1.0, 15.0), n_freqs=60)
    assert abs(_peak_freq(freqs, cwt_mag) - f0) <= 1.0


# ── 4. Hilbert instantaneous frequency recovers a clean tone ────────────────
def test_hilbert_instantaneous_frequency():
    from fastmoda.analysis_gpu import compute_instantaneous_phase_gpu
    f0 = 5.0
    x = np.sin(2 * np.pi * f0 * T)
    out = compute_instantaneous_phase_gpu(x, fs=FS)
    inst = np.asarray(out["frequency"])
    # ignore edge transients
    core = inst[len(inst) // 5: -len(inst) // 5]
    assert abs(np.median(core) - f0) <= 0.5


# ── 5. Butterworth band-pass keeps in-band, rejects out-of-band ─────────────
def test_bandpass_selectivity():
    from fastmoda.filtering import butterworth_bandpass
    f_in, f_out = 5.0, 15.0
    x = np.sin(2 * np.pi * f_in * T) + np.sin(2 * np.pi * f_out * T)
    y = butterworth_bandpass(x, fs=FS, f_low=3.0, f_high=7.0, order=4)

    def band_power(sig, f):
        # narrowband power via projection onto the tone
        c = np.mean(sig * np.exp(-1j * 2 * np.pi * f * T))
        return np.abs(c) ** 2

    # in-band tone should dominate the out-of-band tone after filtering
    assert band_power(y, f_in) > 25 * band_power(y, f_out)


# ── 6. Phase coherence: identical ≈ 1, independent ≈ low ────────────────────
def test_phase_coherence_extremes():
    from fastmoda.analysis_gpu import phase_coherence_gpu
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 3.0 * T) + 0.05 * rng.standard_normal(N)

    same = phase_coherence_gpu(x, x.copy(), fs=FS, window_size=128)
    assert np.nanmean(same["plv"]) > 0.95

    y = np.sin(2 * np.pi * 3.0 * T + rng.uniform(0, 2 * np.pi)) \
        + rng.standard_normal(N)          # independent phase + noise
    z = rng.standard_normal(N)
    indep = phase_coherence_gpu(y, z, fs=FS, window_size=128)
    assert np.nanmean(indep["plv"]) < np.nanmean(same["plv"])


# ── 7. Bispectrum peaks at the interacting frequency triad ──────────────────
def test_bispectrum_quadratic_coupling():
    # A quadratic triad {f1, f2, f1+f2} makes the bispectrum B(f1,f2) large,
    # while a pair whose sum has no component (e.g. (f2,f2)→16 Hz, absent) stays
    # near zero. This is the defining ground-truth signature of the bispectrum.
    from fastmoda.analysis_gpu import bispectrum_gpu
    f1, f2 = 5.0, 8.0
    x = (np.sin(2 * np.pi * f1 * T)
         + np.sin(2 * np.pi * f2 * T)
         + np.sin(2 * np.pi * (f1 + f2) * T))

    res = bispectrum_gpu(x, fs=FS, nfft=256)
    f = np.asarray(res["frequencies"])
    B = np.abs(np.asarray(res["bispectrum"]))

    def at(a, b):
        return B[int(np.argmin(np.abs(f - a))), int(np.argmin(np.abs(f - b)))]

    # interacting pair dominates non-interacting control pairs by orders of mag
    assert at(f1, f2) > 1e3 * max(at(f2, f2), at(f1, f1))
    # and the global bispectral peak sits at the true interacting pair
    pi, pj = np.unravel_index(np.argmax(B), B.shape)
    assert abs(f[pi] - f1) <= 1.0 and abs(f[pj] - f2) <= 1.0


# ── 8. MODWT perfect reconstruction (torch-dependent → skip if absent) ──────
def test_modwt_reconstruction():
    try:
        import torch  # noqa: F401
        from fastmoda.modwt_gpu import modwt_gpu, imodwt_gpu
    except Exception as exc:
        pytest.skip(f"MODWT needs torch, unavailable: {exc}")
    import torch
    x = torch.from_numpy(np.sin(2 * np.pi * 4.0 * T).astype(np.float32))
    w, v = modwt_gpu(x, wavelet="la8", level=4)
    xr = imodwt_gpu(w, v, wavelet="la8")
    err = float(torch.mean((x - xr[: len(x)]) ** 2))
    assert err < 1e-6


# ── 9. Opt-in direct diff against real MODA reference outputs ────────────────
def _load_refs():
    if not os.path.isdir(REF_DIR):
        return []
    import scipy.io as sio
    refs = []
    for fn in sorted(os.listdir(REF_DIR)):
        if fn.startswith("moda_") and fn.endswith(".mat"):
            refs.append((fn, sio.loadmat(os.path.join(REF_DIR, fn))))
    return refs


@pytest.mark.parametrize("case", _load_refs() or [pytest.param(None, marks=pytest.mark.skip(
    reason="no MODA reference .mat under tests/parity/reference/ — "
           "generate with gen_moda_reference.m in MATLAB"))],
    ids=lambda c: c[0] if c else "no-refs")
def test_matches_moda_reference(case):
    """Correlate FastMODA output against a stored MODA reference (tol-based)."""
    name, ref = case
    algo = str(ref.get("algorithm", [""])[0]) if "algorithm" in ref else name
    sig = np.asarray(ref["signal"]).squeeze().astype(np.float64)
    fs = float(np.asarray(ref["fs"]).squeeze())
    moda_out = np.asarray(ref["result"]).squeeze()

    if "wt" in algo or "cwt" in algo or "wavelet" in name.lower():
        from fastmoda.analysis_gpu import cwt_gpu
        _, _, fm = cwt_gpu(sig, fs=fs, n_freqs=moda_out.shape[0])
        fm = np.abs(fm)
    elif "wft" in algo or "wft" in name.lower():
        from fastmoda.filtering import wft
        _, _, fm = wft(sig, fs=fs)
        fm = np.abs(fm)
    else:
        pytest.skip(f"no FastMODA mapping for reference algo '{algo}'")

    a = moda_out.ravel()
    b = np.abs(fm).ravel()
    m = min(len(a), len(b))
    r = np.corrcoef(a[:m], b[:m])[0, 1]
    assert r > 0.9, f"{name}: MODA↔FastMODA correlation {r:.3f} below 0.9"
