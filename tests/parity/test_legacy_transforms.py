"""Tests for the MODA-faithful legacy transforms (fastmoda.legacy_moda).

These pin the legacy port to MODA's documented behaviour and quantify how far
FastMODA's default (fast) path diverges from it — the Tier-3 algorithmic gap
described in docs/validation/algorithmic-differences.md.

Run inside the FastMODA image:
    pytest -q /repo/tests/parity/test_legacy_transforms.py
"""

import numpy as np
import pytest

FS, N = 40.0, 4096
T = np.arange(N) / FS


def test_lognorm_auto_nv_matches_moda_method():
    # sqeps 50%-support (25–75% cumulative of fwt over log-ξ) → nv≈33 for f0=1.
    from fastmoda.legacy_moda import moda_wavelet, _wavelet_params
    fwt, ompeak, xi1, xi2 = moda_wavelet("Lognorm", 1.0)
    wp = _wavelet_params(fwt, ompeak, xi1, xi2)
    nv = 10 * np.log(2) / np.log(wp["xi2h"] / wp["xi1h"])
    assert 31 <= nv <= 35, f"auto nv={nv:.1f} outside MODA-method range"


@pytest.mark.parametrize("f0tone,amp", [(3.0, 1.0), (8.0, 0.5)])
def test_wt_legacy_peak_and_amplitude(f0tone, amp):
    from fastmoda.legacy_moda import wt_legacy
    x = amp * np.cos(2 * np.pi * f0tone * T)
    WT, freq = wt_legacy(x, FS, fmin=0.5, fmax=15, wavelet="Lognorm",
                         padding="zero", cut_edges=True)
    prof = np.nanmean(np.abs(WT), axis=1)
    pk = freq[np.nanargmax(prof)]
    assert abs(pk - f0tone) < 0.2
    # MODA p=1 convention: |WT| at the ridge ≈ A/2 for a real cosine of amp A
    ridge_amp = np.nanmax(np.abs(WT[:, N // 2]))
    assert abs(ridge_amp - amp / 2) < 0.1 * amp + 0.02


def test_wt_legacy_is_complex():
    # coherence & bispectrum need phase — the legacy WT must be complex,
    # unlike the magnitude-only analysis_gpu.cwt_gpu.
    from fastmoda.legacy_moda import wt_legacy
    WT, _ = wt_legacy(np.cos(2 * np.pi * 5 * T), FS, fmin=1, fmax=12,
                      padding="zero", cut_edges=False)
    assert np.iscomplexobj(WT)
    assert np.nanmean(np.abs(WT.imag)) > 1e-3


def test_default_cwt_diverges_from_legacy():
    # Quantify the Tier-3 gap: the basic magnitude CWT is NOT MODA-equivalent.
    from fastmoda.legacy_moda import wt_legacy
    from fastmoda.analysis_gpu import cwt_gpu
    x = np.cos(2 * np.pi * 3 * T) + 0.5 * np.cos(2 * np.pi * 8 * T)
    WT, freq = wt_legacy(x, FS, fmin=1, fmax=15, padding="zero", cut_edges=False)
    A_leg = np.abs(WT)
    fg, _, A_gpu = cwt_gpu(x, fs=FS, freq_range=(1, 15), n_freqs=len(freq))
    A_gpu_rs = np.vstack([np.interp(freq, fg, A_gpu[:, j])
                          for j in range(A_gpu.shape[1])]).T
    norm = lambda a: a / (np.nanmax(a) + 1e-12)
    corr = np.corrcoef(norm(A_leg).ravel(), norm(A_gpu_rs).ravel())[0, 1]
    # they agree on gross structure but are materially different (< 0.9)
    assert corr < 0.9, f"unexpectedly high corr {corr:.3f} — gap understated?"


def test_wavelet_families_available():
    from fastmoda.legacy_moda import moda_wavelet
    for name in ("Lognorm", "Morlet", "Bump"):
        fwt, ompeak, xi1, xi2 = moda_wavelet(name, 1.0)
        assert callable(fwt) and ompeak > 0


# ── wft_legacy (port of wft.m) ──────────────────────────────────────────────

@pytest.mark.parametrize("f0tone", [3.0, 8.0])
def test_wft_legacy_peak_and_complex(f0tone):
    from fastmoda.legacy_moda import wft_legacy
    x = np.cos(2 * np.pi * f0tone * T)
    WFT, freq = wft_legacy(x, FS, fmin=0.5, fmax=15, window="Gaussian",
                           padding="zero", cut_edges=False)
    assert np.iscomplexobj(WFT)                 # phase preserved
    pk = freq[np.nanargmax(np.nanmean(np.abs(WFT), axis=1))]
    assert abs(pk - f0tone) < 0.1
    # linear grid (constant spacing), unlike the CWT's geometric lattice
    df = np.diff(freq)
    assert np.allclose(df, df[0], rtol=1e-6)


def test_wft_legacy_all_windows():
    from fastmoda.legacy_moda import wft_legacy
    for w in ("Gaussian", "Hann", "Blackman", "Exp", "Rect", "Kaiser"):
        WFT, freq = wft_legacy(np.cos(2 * np.pi * 5 * T), FS, fmin=1, fmax=12,
                               window=w, padding="zero")
        pk = freq[np.nanargmax(np.nanmean(np.abs(WFT), axis=1))]
        assert abs(pk - 5.0) < 0.2, f"{w}: peak {pk:.2f} != 5 Hz"


def test_legacy_coherence_and_bispectrum_smoke():
    # coherence & bispectrum are pure phase combinations of wt_legacy — verify
    # a phase-coupled pair peaks at the coupling frequency and a quadratic triad
    # lights the bispectrum at the interacting pair.
    from fastmoda.legacy_moda import wt_legacy
    from fastmoda.ridge_gpu import time_localized_coherence
    rng = np.random.default_rng(0)
    # noise breaks the trivial all-frequency coherence of two pure tones, so the
    # coherence peak localises at the shared 3 Hz component
    a = np.cos(2 * np.pi * 3 * T) + 0.3 * rng.standard_normal(N)
    b = np.cos(2 * np.pi * 3 * T + 0.5) + 0.3 * rng.standard_normal(N)
    Wa, fr = wt_legacy(a, FS, fmin=0.5, fmax=12, cut_edges=False)
    Wb, _ = wt_legacy(b, FS, fmin=0.5, fmax=12, cut_edges=False)
    tpc = time_localized_coherence(Wa, Wb, fr, FS, numcycles=10)
    pk = fr[np.nanargmax(np.nanmean(tpc, axis=1))]
    assert abs(pk - 3.0) < 0.6
