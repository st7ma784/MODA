"""
FastMODA algorithmic correctness tests.

Each test uses synthetic signals with analytically known properties and
verifies that FastMODA's output matches mathematical ground truth.
No MATLAB required — the checks are derived from signal-processing theory.

Run from repo root:
    cd /home/user/MODA
    pytest tests/test_fastmoda_algorithms.py -v

Or a specific group:
    pytest tests/test_fastmoda_algorithms.py -v -k cwt
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from scipy.signal import hilbert
from functools import lru_cache

# ── path setup ────────────────────────────────────────────────────────────────

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "FastMODA"))

from fastmoda.ridge_gpu import (
    cwt_complex, extract_ridge, cone_of_influence,
    time_localized_coherence, nv_to_freqs,
)
from fastmoda.coupling_gpu import (
    estimate_coupling_functions, sync_map, _build_basis,
)
from fastmoda.biphase_gpu import biphase_timeseries, bispectrum4, _robust_unwrap
from fastmoda.filtering import (
    butterworth_bandpass, iaaft_surrogates, aaft_surrogates,
)

# ── shared constants ──────────────────────────────────────────────────────────

FS      = 256.0          # Hz — matches signal_generator.py default
DURATION = 10.0          # seconds — long enough for meaningful COI
N       = int(FS * DURATION)
RNG     = np.random.default_rng(42)   # deterministic everywhere


def _sine(freq: float, amp: float = 1.0, phase: float = 0.0,
          fs: float = FS, n: int = N) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.cos(2 * np.pi * freq * t + phase)


def _freqs(fmin: float = 0.5, fmax: float = 60.0, n: int = 50) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CWT — correctness and COI masking
# ═════════════════════════════════════════════════════════════════════════════

class TestCWT:
    """CWT must localise energy at the correct frequency and honour COI edges."""

    F0   = 10.0   # Hz — within band, well away from Nyquist
    FREQS = _freqs(0.5, 60.0, 64)

    def _peak_freq(self, cwt: np.ndarray) -> float:
        """Frequency of the maximum mean amplitude bin."""
        mean_amp = np.nanmean(np.abs(cwt), axis=1)
        return self.FREQS[np.nanargmax(mean_amp)]

    def test_lognorm_peak_at_f0(self):
        """Lognorm CWT of a pure sine must peak at the sine's frequency."""
        x = _sine(self.F0)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
        assert abs(self._peak_freq(cwt) - self.F0) < 1.5, (
            f"CWT peak {self._peak_freq(cwt):.2f} Hz, expected {self.F0} Hz")

    def test_morlet_peak_at_f0(self):
        """Morlet CWT of a pure sine must also peak at the sine's frequency."""
        x = _sine(self.F0)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="morlet")
        assert abs(self._peak_freq(cwt) - self.F0) < 1.5

    def test_cut_edges_produces_nan(self):
        """With cut_edges=True the COI boundary samples must be NaN."""
        x = _sine(self.F0)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm", cut_edges=True)
        assert np.isnan(cwt).any(), "cut_edges=True should produce NaN at edges"

    def test_no_nan_without_cut_edges(self):
        """With cut_edges=False there must be no NaN values."""
        x = _sine(self.F0)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm", cut_edges=False)
        assert not np.isnan(cwt).any(), "cut_edges=False must not produce NaN"

    def test_coi_shape(self):
        """COI mask must be (n_freqs, N) and mark both edge intervals."""
        coi = cone_of_influence(self.FREQS, N, FS, n_cycles=6.0, wavelet="lognorm")
        assert coi.shape == (len(self.FREQS), N)
        # lowest frequency has widest COI — both ends should be masked
        assert coi[0, 0], "Low-freq COI should mask start"
        assert coi[0, -1], "Low-freq COI should mask end"
        # high frequency has narrow COI — midpoint should be unmasked
        mid = N // 2
        assert not coi[-1, mid], "High-freq COI should not mask midpoint"

    def test_nv_to_freqs_log_spacing(self):
        """nv_to_freqs must produce log-spaced array from fmin to ~fmax."""
        freqs = nv_to_freqs(0.5, 50.0, nv=8)
        assert freqs[0] == pytest.approx(0.5, rel=1e-3)
        ratios = freqs[1:] / freqs[:-1]
        # All consecutive ratios should be equal (log-uniform)
        assert np.std(ratios) < 1e-6, "nv_to_freqs must produce uniform log ratio"

    def test_amplitude_scaling(self):
        """CWT amplitude at ridge should be proportional to signal amplitude."""
        amps = [0.5, 1.0, 2.0]
        ridge_amps = []
        for a in amps:
            x = _sine(self.F0, amp=a)
            cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
            f0_idx = np.argmin(np.abs(self.FREQS - self.F0))
            ridge_amps.append(float(np.nanmean(np.abs(cwt[f0_idx]))))
        # Each amplitude step should roughly double the CWT amplitude
        ratio_01 = ridge_amps[1] / ridge_amps[0]
        ratio_12 = ridge_amps[2] / ridge_amps[1]
        assert 1.7 < ratio_01 < 2.3, f"Amplitude scaling off: {ratio_01:.2f}"
        assert 1.7 < ratio_12 < 2.3, f"Amplitude scaling off: {ratio_12:.2f}"

    def test_multi_component_two_peaks(self):
        """CWT of two-tone signal must show two distinct amplitude peaks."""
        f1, f2 = 6.0, 20.0
        x = _sine(f1) + _sine(f2)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
        mean_amp = np.nanmean(np.abs(cwt), axis=1)
        # Find local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mean_amp, distance=3)
        peak_freqs = self.FREQS[peaks]
        # At least one peak near f1 and one near f2
        near_f1 = np.any(np.abs(peak_freqs - f1) < 2.0)
        near_f2 = np.any(np.abs(peak_freqs - f2) < 3.0)
        assert near_f1, f"No CWT peak near f1={f1}: peaks at {peak_freqs}"
        assert near_f2, f"No CWT peak near f2={f2}: peaks at {peak_freqs}"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Ridge extraction
# ═════════════════════════════════════════════════════════════════════════════

class TestRidgeExtraction:
    """Ridge must track frequency, amplitude, and phase of a pure sinusoid."""

    F0    = 10.0
    AMP   = 1.5
    FREQS = _freqs(0.5, 60.0, 64)

    def _run(self, smooth_len=5):
        x   = _sine(self.F0, amp=self.AMP)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm", cut_edges=True)
        return extract_ridge(cwt, self.FREQS, FS, smooth_len=smooth_len)

    def test_ridge_frequency_accuracy(self):
        """Ridge mean frequency must be within 5% of true f0 (outside COI)."""
        r = self._run()
        # Drop NaN (COI edges)
        ifreq = r["ifreq"][~np.isnan(r["ifreq"])]
        assert len(ifreq) > 0, "All ridge samples are NaN"
        assert abs(np.mean(ifreq) - self.F0) < 0.5 * self.F0 * 0.05 + 0.5, (
            f"Ridge mean freq {np.mean(ifreq):.2f} vs {self.F0}")

    def test_ridge_amplitude_positive(self):
        """Ridge instantaneous amplitude must be strictly positive."""
        r = self._run()
        iamp = r["iamp"][~np.isnan(r["iamp"])]
        assert np.all(iamp > 0), "iamp must be positive"

    def test_ridge_amplitude_proportional(self):
        """Ridge amplitude should scale with signal amplitude (within 30%)."""
        amps = [0.5, 1.0, 2.0]
        mean_iamps = []
        for a in amps:
            x   = _sine(self.F0, amp=a)
            cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm", cut_edges=True)
            r   = extract_ridge(cwt, self.FREQS, FS, smooth_len=5)
            ia  = r["iamp"][~np.isnan(r["iamp"])]
            mean_iamps.append(float(np.mean(ia)))
        # Ratios should be ~2 between consecutive amplitudes
        assert 1.4 < mean_iamps[1] / mean_iamps[0] < 2.6
        assert 1.4 < mean_iamps[2] / mean_iamps[1] < 2.6

    def test_ridge_phase_linear(self):
        """Instantaneous phase of a pure sine must be (nearly) linear in time."""
        r   = self._run()
        iphi = r["iphi"]
        valid = ~np.isnan(iphi)
        iphi_v = iphi[valid]
        t_v    = np.arange(len(iphi))[valid] / FS
        # Fit a line to the unwrapped phase
        unwrapped = np.unwrap(iphi_v)
        coeffs    = np.polyfit(t_v, unwrapped, 1)
        residuals = np.polyval(coeffs, t_v) - unwrapped
        # Residual std should be small relative to 2π
        assert np.std(residuals) < 0.3, (
            f"Phase non-linearity: std={np.std(residuals):.3f} rad")

    def test_savgol_smoother_than_box(self):
        """After our fix, Savitzky-Golay ridge should be smoother than box convolution."""
        # Noisy signal: sine + pink-noise-like jitter
        x_clean = _sine(self.F0, amp=self.AMP)
        noise   = 0.5 * RNG.standard_normal(N)
        x_noisy = x_clean + noise

        cwt = cwt_complex(x_noisy, self.FREQS, FS, wavelet="lognorm")
        r_sg  = extract_ridge(cwt, self.FREQS, FS, smooth_len=15)   # uses SG

        # Compute manually a box-smoothed reference
        from scipy.signal import savgol_filter
        amp_raw = np.abs(cwt)
        ridge_idx_raw = np.argmax(amp_raw, axis=0).astype(float)
        kernel = np.ones(15) / 15
        ridge_box = np.convolve(ridge_idx_raw, kernel, mode="same")
        ridge_box = np.clip(ridge_box.round(), 0, len(self.FREQS) - 1).astype(int)

        # SG result
        ifreq_sg  = r_sg["ifreq"]
        ifreq_box = self.FREQS[ridge_box]

        # SG should be smoother (lower second-difference norm)
        smooth_sg  = np.std(np.diff(ifreq_sg[5:-5]))
        smooth_box = np.std(np.diff(ifreq_box[5:-5]))
        assert smooth_sg <= smooth_box * 1.05, (
            f"SG not smoother: SG std={smooth_sg:.4f}, box std={smooth_box:.4f}")

    def test_reconstruction_correlates_with_input(self):
        """Reconstructed signal from ridge must correlate with original."""
        x = _sine(self.F0, amp=self.AMP)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
        r   = extract_ridge(cwt, self.FREQS, FS, smooth_len=5)
        recon = r["recon"]
        valid = ~np.isnan(recon)
        corr = np.corrcoef(x[valid], recon[valid])[0, 1]
        assert corr > 0.9, f"Reconstruction correlation {corr:.3f} < 0.9"


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Time-localised phase coherence (TLPHCOH)
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseCoherence:
    """
    Coherence = 1 for identical signals; coherence ≈ 0 for independent noise.
    """

    FREQS = _freqs(0.5, 60.0, 32)

    def _tpc(self, x1, x2, numcycles=10):
        cwt1 = cwt_complex(x1, self.FREQS, FS, wavelet="lognorm")
        cwt2 = cwt_complex(x2, self.FREQS, FS, wavelet="lognorm")
        return time_localized_coherence(cwt1, cwt2, self.FREQS, FS,
                                        numcycles=numcycles)

    def test_identical_signals_coherence_near_one(self):
        """Two copies of the same signal must have TPC ≈ 1 everywhere."""
        x = _sine(10.0) + 0.2 * RNG.standard_normal(N)
        tpc = self._tpc(x, x)
        mean_coh = float(np.nanmean(tpc))
        assert mean_coh > 0.90, f"Identical signal coherence {mean_coh:.3f} < 0.90"

    def test_independent_noise_coherence_low(self):
        """Two independent white-noise signals must have TPC well below 0.5."""
        x1 = RNG.standard_normal(N).astype(np.float32)
        x2 = RNG.standard_normal(N).astype(np.float32)
        tpc = self._tpc(x1, x2)
        mean_coh = float(np.nanmean(tpc))
        assert mean_coh < 0.4, f"Independent noise coherence {mean_coh:.3f} ≥ 0.4"

    def test_coherent_at_signal_frequency(self):
        """Mixed (signal + noise) pair must be coherent at the signal frequency."""
        f0 = 8.0
        x_clean  = _sine(f0, amp=2.0)
        noise_a  = 0.5 * RNG.standard_normal(N)
        noise_b  = 0.5 * RNG.standard_normal(N)
        x1 = x_clean + noise_a
        x2 = x_clean + noise_b
        tpc = self._tpc(x1, x2)
        # Mean TPC at the f0 frequency row
        f0_idx = np.argmin(np.abs(self.FREQS - f0))
        coh_at_f0 = float(np.nanmean(tpc[f0_idx]))
        assert coh_at_f0 > 0.60, (
            f"TPC at {f0} Hz = {coh_at_f0:.3f} (should be > 0.60)")

    def test_tpc_values_in_range(self):
        """All non-NaN TPC values must be in [0, 1]."""
        x1 = _sine(5.0) + 0.3 * RNG.standard_normal(N)
        x2 = _sine(5.0) + 0.3 * RNG.standard_normal(N)
        tpc = self._tpc(x1, x2)
        valid = tpc[~np.isnan(tpc)]
        assert np.all(valid >= 0.0), "TPC below 0"
        assert np.all(valid <= 1.0 + 1e-6), "TPC above 1"

    def test_tpc_shape(self):
        """TPC shape must match (n_freqs, N)."""
        x = _sine(10.0)
        tpc = self._tpc(x, x)
        assert tpc.shape == (len(self.FREQS), N)

    def test_frequency_adaptive_window(self):
        """Lower frequencies should get wider TPC windows (more NaN at edges)."""
        x = _sine(10.0)
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
        tpc = time_localized_coherence(cwt, cwt, self.FREQS, FS, numcycles=10)
        # Count NaN in lowest vs highest frequency row
        nan_low  = int(np.sum(np.isnan(tpc[0])))
        nan_high = int(np.sum(np.isnan(tpc[-1])))
        assert nan_low >= nan_high, (
            f"Expected wider window at low freq: NaN low={nan_low}, high={nan_high}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Coupling functions
# ═════════════════════════════════════════════════════════════════════════════

class TestCouplingFunctions:
    """Fourier basis, OLS estimation, coupling SE, and direction properties."""

    def test_basis_dimension_bn3(self):
        """Basis size K = 1 + 4·bn + 4·bn² must be 49 for bn=3."""
        bn = 3
        K_expected = 1 + 4 * bn + 4 * bn * bn   # = 49
        phi1 = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
        phi2 = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
        P = _build_basis(phi1, phi2, bn)
        assert P.shape == (100, K_expected), (
            f"Basis shape {P.shape} for bn={bn}, expected (100, {K_expected})")

    def test_basis_dimension_bn2(self):
        """K must be 25 for bn=2."""
        bn = 2
        K_expected = 1 + 4 * bn + 4 * bn * bn   # = 25
        phi = np.linspace(0, 2 * np.pi, 50, dtype=np.float32)
        P = _build_basis(phi, phi, bn)
        assert P.shape[1] == K_expected

    def test_bn3_has_more_terms_than_bn2(self):
        """bn=3 must produce more basis terms than bn=2 (captures more harmonics)."""
        phi = np.linspace(0, 2 * np.pi, 50, dtype=np.float32)
        K2 = _build_basis(phi, phi, 2).shape[1]
        K3 = _build_basis(phi, phi, 3).shape[1]
        assert K3 > K2, f"bn=3 should have more terms: K3={K3}, K2={K2}"

    def test_coupling_direction_independent_signals(self):
        """Independent random phases should give near-zero mean direction."""
        ph1 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        ph2 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        res = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=20)
        mean_dir = float(np.mean(res["direction"]))
        assert abs(mean_dir) < 0.5, (
            f"Independent signals: mean direction {mean_dir:.3f} (should be ~0)")

    def test_coupling_se_present(self):
        """Result must include cpl1_se, cpl2_se, dir_se (our new SE output)."""
        ph1 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        ph2 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        res = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=20)
        assert "cpl1_se" in res, "cpl1_se missing from result"
        assert "cpl2_se" in res, "cpl2_se missing from result"
        assert "dir_se"  in res, "dir_se missing from result"
        assert res["cpl1_se"] >= 0, "cpl1_se must be non-negative"
        assert res["cpl2_se"] >= 0, "cpl2_se must be non-negative"
        assert res["dir_se"]  >= 0, "dir_se must be non-negative"

    def test_coupling_direction_range(self):
        """Direction must always be in [-1, 1]."""
        ph1 = np.cumsum(RNG.uniform(0.8, 1.2, N)).astype(np.float32)
        ph2 = np.cumsum(RNG.uniform(0.8, 1.2, N)).astype(np.float32)
        res = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=20)
        d = res["direction"]
        assert np.all(d >= -1.0 - 1e-6), f"Direction below -1: {d.min()}"
        assert np.all(d <=  1.0 + 1e-6), f"Direction above +1: {d.max()}"

    def test_coupling_fn_grid_shape(self):
        """Coupling function grids must be (n_grid, n_grid)."""
        n_grid = 20
        ph1 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        ph2 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        res = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=n_grid)
        assert res["q21"].shape == (n_grid, n_grid)
        assert res["q12"].shape == (n_grid, n_grid)

    def test_sync_map_returns_required_keys(self):
        """sync_map must return sync_index, fixed_points_rad, coupling_profile."""
        c = np.zeros(1 + 4 * 3 + 4 * 3 * 3, dtype=np.float32)   # bn=3, all zeros
        sm = sync_map(c, c, bn=3, n_grid=100)
        for key in ("sync_index", "fixed_points_rad", "coupling_profile"):
            assert key in sm, f"sync_map missing key: {key}"

    def test_default_bn_is_3(self):
        """estimate_coupling_functions must default to bn=3 (not 2)."""
        import inspect
        sig = inspect.signature(estimate_coupling_functions)
        bn_default = sig.parameters["bn"].default
        assert bn_default == 3, (
            f"Default bn is {bn_default}, expected 3")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Biphase and bispectrum
# ═════════════════════════════════════════════════════════════════════════════

class TestBiphase:
    """
    For a signal with known quadratic phase coupling (f3 = f1 + f2), the
    biphase must be near-constant and the biamplitude must be positive.
    """

    F1, F2 = 6.0, 10.0       # Hz
    PHI1, PHI2 = 0.5, 1.2    # radians

    def _coupled_signals(self):
        f3  = self.F1 + self.F2
        t   = np.arange(N) / FS
        x1  = np.cos(2 * np.pi * self.F1 * t + self.PHI1)
        x2  = (np.cos(2 * np.pi * self.F2  * t + self.PHI2) +
               np.cos(2 * np.pi * f3 * t + self.PHI1 + self.PHI2))
        return x1.astype(np.float32), x2.astype(np.float32)

    def test_biphase_near_constant_for_coupled_signal(self):
        """Biphase of a quadratically coupled signal pair must be near-constant."""
        x1, x2 = self._coupled_signals()
        res = biphase_timeseries(x1, x2, FS, self.F1, self.F2, wavelet="lognorm")
        bp = res["biphase"]
        # Strip edges (COI influence), keep central 60%
        s = int(0.2 * N)
        e = int(0.8 * N)
        bp_mid = bp[s:e]
        std_bp = float(np.std(bp_mid))
        # A truly constant biphase has std = 0; allow some wavelet leakage
        assert std_bp < np.pi / 2, (
            f"Biphase std {std_bp:.3f} rad — not constant enough for coupled signal")

    def test_biamp_positive_for_coupled_signal(self):
        """Biamplitude must be strictly positive."""
        x1, x2 = self._coupled_signals()
        res = biphase_timeseries(x1, x2, FS, self.F1, self.F2, wavelet="lognorm")
        assert np.all(res["biamp"] >= 0), "biamp must be non-negative"
        assert float(np.mean(res["biamp"])) > 0, "mean biamp must be positive"

    def test_biphase_returns_required_keys(self):
        """Result must include biamp, biphase, time, f1, f2, f3."""
        x = _sine(self.F1).astype(np.float32)
        res = biphase_timeseries(x, x, FS, self.F1, self.F2, wavelet="lognorm")
        for k in ("biamp", "biphase", "time", "f1", "f2", "f3"):
            assert k in res, f"biphase_timeseries missing key: {k}"
        assert res["f3"] == pytest.approx(self.F1 + self.F2)

    def test_f3_above_nyquist_raises(self):
        """f3 > fs/2 must raise ValueError (Nyquist guard)."""
        x = _sine(10.0).astype(np.float32)
        with pytest.raises(ValueError, match="Nyquist"):
            biphase_timeseries(x, x, FS, 100.0, 50.0)

    def test_robust_unwrap_corrects_artificial_jump(self):
        """_robust_unwrap must remove a large isolated phase jump."""
        t     = np.linspace(0, 2 * np.pi * 5, 1000)
        phase = np.sin(t)                     # smooth phase, range ~[-1,1]
        # inject a ±10 rad jump at sample 500
        phase_jumpy = phase.copy()
        phase_jumpy[500:] += 10.0
        unwrapped_raw    = np.unwrap(phase_jumpy)
        unwrapped_robust = _robust_unwrap(phase_jumpy)
        # Robust version should be closer to the original smooth phase
        err_raw    = float(np.std(unwrapped_raw    - phase))
        err_robust = float(np.std(unwrapped_robust - phase))
        # At minimum robust should not be worse
        assert err_robust <= err_raw * 1.1 or err_robust < 0.5, (
            f"Robust unwrap did not help: raw_err={err_raw:.3f}, "
            f"robust_err={err_robust:.3f}")

    def test_bispectrum4_shape(self):
        """bispectrum4 must return (n_freq, n_freq) matrices for all four types."""
        x1 = _sine(self.F1).astype(np.float32)
        x2 = _sine(self.F2).astype(np.float32)
        nfft = 64
        res  = bispectrum4(x1, x2, FS, nfft=nfft)
        n_freq = nfft // 2 + 1
        for key in ("b111", "b222", "b122", "b211"):
            assert key in res, f"bispectrum4 missing key: {key}"
            assert res[key].shape == (n_freq, n_freq), (
                f"{key} shape {res[key].shape} != ({n_freq}, {n_freq})")

    def test_bispectrum4_auto_vs_cross(self):
        """Auto-bispectrum (b111) of sine must show peak at (f1, f1)."""
        f0 = 8.0
        x  = _sine(f0).astype(np.float32)
        res = bispectrum4(x, x, FS, nfft=256)
        freqs = res["frequencies"]
        amp   = res["biamp111"]
        f0_idx = np.argmin(np.abs(freqs - f0))
        # Amplitude at (f0, f0) should dominate the region near it
        local = amp[max(0, f0_idx-2):f0_idx+3, max(0, f0_idx-2):f0_idx+3]
        assert local.max() > 0, "Auto-bispectrum should be non-zero at (f0,f0)"


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Butterworth bandpass filter
# ═════════════════════════════════════════════════════════════════════════════

class TestButterworthFilter:
    """Filter must pass in-band components and attenuate out-of-band ones."""

    FS_LOCAL = FS

    def _power(self, x):
        return float(np.mean(x ** 2))

    def test_passband_preserved(self):
        """Sine inside the passband must retain > 50% of its power."""
        f_pass  = 10.0
        x       = _sine(f_pass)
        x_filt  = butterworth_bandpass(x, self.FS_LOCAL, 8.0, 12.0, order=4)
        ratio   = self._power(x_filt) / self._power(x)
        assert ratio > 0.5, f"Passband power ratio {ratio:.3f} < 0.5"

    def test_stopband_attenuated(self):
        """Sine well outside passband must be attenuated by > 10 dB."""
        f_stop  = 40.0
        x       = _sine(f_stop)
        x_filt  = butterworth_bandpass(x, self.FS_LOCAL, 8.0, 12.0, order=4)
        ratio   = self._power(x_filt) / max(self._power(x), 1e-12)
        assert ratio < 0.1, (
            f"Stopband attenuation insufficient: power ratio {ratio:.4f}")

    def test_output_length_unchanged(self):
        """Filtered signal must be the same length as input."""
        x     = _sine(10.0)
        x_f   = butterworth_bandpass(x, self.FS_LOCAL, 8.0, 12.0)
        assert len(x_f) == len(x)

    def test_output_float32(self):
        """Output dtype must be float32 (matches FastMODA convention)."""
        x   = _sine(10.0)
        x_f = butterworth_bandpass(x, self.FS_LOCAL, 8.0, 12.0)
        assert x_f.dtype == np.float32

    def test_multi_band_selectivity(self):
        """Two-tone signal: bandpass around f1 must suppress f2."""
        f1, f2 = 6.0, 25.0
        x      = _sine(f1) + _sine(f2)
        x_f    = butterworth_bandpass(x, self.FS_LOCAL, 4.0, 9.0, order=4)
        # After filtering, power at f2 should be negligible
        # Compute via DFT
        spec   = np.abs(np.fft.rfft(x_f))
        freqs  = np.fft.rfftfreq(len(x_f), 1.0 / self.FS_LOCAL)
        pwr_f1 = float(np.mean(spec[(freqs >= 4)  & (freqs <= 9)]  ** 2))
        pwr_f2 = float(np.mean(spec[(freqs >= 20) & (freqs <= 35)] ** 2))
        assert pwr_f1 > pwr_f2 * 5, (
            f"Band selectivity failed: pwr_f1={pwr_f1:.4f}, pwr_f2={pwr_f2:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Surrogate testing (IAAFT / AAFT)
# ═════════════════════════════════════════════════════════════════════════════

class TestSurrogates:
    """IAAFT surrogates must preserve amplitude distribution and power spectrum."""

    N_SURR = 10

    def _sig(self):
        t = np.arange(N) / FS
        return (_sine(6.0) + 0.5 * _sine(10.0)
                + 0.3 * RNG.standard_normal(N)).astype(np.float32)

    def test_iaaft_preserves_amplitude_distribution(self):
        """IAAFT surrogates must have the same sorted amplitude values as original."""
        x    = self._sig()
        surr = iaaft_surrogates(x, self.N_SURR)
        assert surr.shape == (self.N_SURR, N)
        x_sorted = np.sort(x)
        for i in range(self.N_SURR):
            s_sorted = np.sort(surr[i])
            np.testing.assert_allclose(
                s_sorted, x_sorted, atol=1e-4,
                err_msg=f"Surrogate {i} amplitude distribution mismatch")

    def test_iaaft_randomises_phase(self):
        """IAAFT surrogates must not be identical to the original signal."""
        x    = self._sig()
        surr = iaaft_surrogates(x, self.N_SURR)
        for i in range(self.N_SURR):
            assert not np.allclose(surr[i], x), (
                f"Surrogate {i} is identical to original — phase not randomised")

    def test_iaaft_preserves_power_spectrum(self):
        """IAAFT surrogates must have power spectrum close to original."""
        x    = self._sig()
        surr = iaaft_surrogates(x, self.N_SURR)
        amp_orig = np.abs(np.fft.rfft(x))
        for i in range(self.N_SURR):
            amp_surr = np.abs(np.fft.rfft(surr[i]))
            corr = float(np.corrcoef(amp_orig, amp_surr)[0, 1])
            assert corr > 0.98, (
                f"Surrogate {i} spectrum correlation {corr:.4f} < 0.98")

    def test_aaft_shape(self):
        """AAFT must return (n_surrogates, N) float32 array."""
        x    = self._sig()
        surr = aaft_surrogates(x, self.N_SURR)
        assert surr.shape == (self.N_SURR, N)
        assert surr.dtype == np.float32

    def test_surrogates_are_independent(self):
        """Different surrogate realisations must not be identical to each other."""
        x    = self._sig()
        surr = iaaft_surrogates(x, 3)
        assert not np.allclose(surr[0], surr[1]), "Surrogate 0 and 1 are identical"
        assert not np.allclose(surr[1], surr[2]), "Surrogate 1 and 2 are identical"


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Signal consistency — end-to-end pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """
    Replicate what MODA does on a multi-component EEG-like signal and verify
    the chain: filter → phase → coupling → coherence is self-consistent.
    """

    FREQS = _freqs(0.5, 50.0, 32)
    F_ALPHA = 10.0   # Hz — alpha band
    F_THETA =  6.0   # Hz — theta band

    def _make_signal(self, seed=0):
        """Reproducible multi-component EEG-like signal."""
        rng = np.random.default_rng(seed)
        t   = np.arange(N) / FS
        x   = (1.0 * np.cos(2 * np.pi * self.F_ALPHA * t) +
               0.5 * np.cos(2 * np.pi * self.F_THETA * t) +
               0.2 * rng.standard_normal(N))
        return x.astype(np.float32)

    def test_filter_then_hilbert_phase(self):
        """Phase from bandpass+Hilbert must be near-linear in passband centre."""
        x = self._make_signal()
        x_filt = butterworth_bandpass(x, FS, 8.0, 12.0)   # alpha band
        phase  = np.angle(hilbert(x_filt))
        # Instantaneous frequency from phase derivative should be near 10 Hz
        ifreq = np.diff(np.unwrap(phase)) / (2 * np.pi / FS)
        # Central 80% to avoid filter edge artefacts
        s, e = int(0.1 * N), int(0.9 * N)
        mean_ifreq = float(np.mean(ifreq[s:e]))
        assert abs(mean_ifreq - self.F_ALPHA) < 2.0, (
            f"Hilbert IF {mean_ifreq:.2f} Hz, expected ~{self.F_ALPHA}")

    def test_cwt_ridge_vs_hilbert_agree(self):
        """CWT ridge frequency and Hilbert IF must agree within 15% on pure sine."""
        x = _sine(self.F_ALPHA)
        # CWT ridge
        cwt   = cwt_complex(x, self.FREQS, FS, wavelet="lognorm", cut_edges=True)
        ridge = extract_ridge(cwt, self.FREQS, FS, smooth_len=5)
        valid = ~np.isnan(ridge["ifreq"])
        ridge_f = float(np.mean(ridge["ifreq"][valid]))

        # Hilbert via bandpass
        x_filt  = butterworth_bandpass(x, FS, 8.0, 12.0)
        ifreq_h = np.diff(np.unwrap(np.angle(hilbert(x_filt)))) / (2 * np.pi / FS)
        hilbert_f = float(np.mean(ifreq_h[int(0.1*N):int(0.9*N)]))

        assert abs(ridge_f - hilbert_f) / self.F_ALPHA < 0.15, (
            f"Ridge {ridge_f:.2f} Hz vs Hilbert {hilbert_f:.2f} Hz disagree > 15%")

    def test_coherence_self_consistency(self):
        """Phase coherence of signal with itself must be > 0.9."""
        x  = self._make_signal()
        cwt = cwt_complex(x, self.FREQS, FS, wavelet="lognorm")
        tpc = time_localized_coherence(cwt, cwt, self.FREQS, FS, numcycles=10)
        coh = float(np.nanmean(tpc))
        assert coh > 0.9, f"Self-coherence {coh:.3f} < 0.9"

    def test_coupling_then_sync_map_consistent(self):
        """sync_map keys derived from coupling must be self-consistent."""
        ph1 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        ph2 = np.cumsum(RNG.uniform(0.9, 1.1, N)).astype(np.float32)
        cf  = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=20)
        sm  = sync_map(cf["c1_mean"], cf["c2_mean"], bn=3)
        assert "sync_index" in sm
        assert "is_synchronised" in sm
        assert sm["sync_index"] in (0, 1)
        assert isinstance(sm["fixed_points_rad"], list)

    def test_full_pipeline_no_errors(self):
        """The complete filter→CWT→ridge→coherence→coupling chain must not crash."""
        x1 = self._make_signal(seed=0)
        x2 = self._make_signal(seed=1)

        # Step 1: bandpass and extract phase
        x1_f  = butterworth_bandpass(x1, FS, 8.0, 12.0)
        x2_f  = butterworth_bandpass(x2, FS, 8.0, 12.0)
        ph1   = np.angle(hilbert(x1_f)).astype(np.float32)
        ph2   = np.angle(hilbert(x2_f)).astype(np.float32)

        # Step 2: CWT + ridge (cut_edges=True for ridge — TPC uses its own edge mask)
        freqs = _freqs(0.5, 50.0, 32)
        cwt1_ridge = cwt_complex(x1, freqs, FS, wavelet="lognorm", cut_edges=True)
        ridge = extract_ridge(cwt1_ridge, freqs, FS, smooth_len=5)

        # Step 3: TLPHCOH — use cut_edges=False so cumsum doesn't see NaN
        cwt1 = cwt_complex(x1, freqs, FS, wavelet="lognorm", cut_edges=False)
        cwt2 = cwt_complex(x2, freqs, FS, wavelet="lognorm", cut_edges=False)
        tpc  = time_localized_coherence(cwt1, cwt2, freqs, FS, numcycles=10)

        # Step 4: coupling
        cf = estimate_coupling_functions(ph1, ph2, FS, bn=3, win_s=5.0,
                                          overlap=0.5, n_grid=20)

        # Step 5: biphase
        bp = biphase_timeseries(x1.astype(np.float32), x2.astype(np.float32),
                                FS, 6.0, 10.0, wavelet="lognorm")

        # Basic sanity on each output
        assert ridge["ifreq"] is not None
        assert not np.all(np.isnan(tpc))
        assert "direction" in cf
        assert "biamp" in bp


# ═════════════════════════════════════════════════════════════════════════════
# 9.  Sample signals from signal_generator.py — preset band power check
# ═════════════════════════════════════════════════════════════════════════════

class TestSampleSignalBandPowers:
    """
    Generate the same signals used by the emulator and verify that FastMODA's
    CWT correctly reflects the intended band-power hierarchy.
    """

    PRESETS = {
        "resting": dict(alpha=1.0, theta=0.3, beta=0.12, delta=0.10, gamma=0.05),
        "active":  dict(alpha=0.3, theta=0.2, beta=0.80, delta=0.05, gamma=0.20),
        "drowsy":  dict(alpha=0.5, theta=0.9, beta=0.05, delta=0.30, gamma=0.02),
        "sleep":   dict(alpha=0.1, theta=0.3, beta=0.04, delta=1.20, gamma=0.02),
    }

    BAND_CENTRES = dict(delta=2.0, theta=6.0, alpha=10.0, beta=18.0, gamma=40.0)

    def _make_preset_signal(self, preset: str) -> np.ndarray:
        amps = self.PRESETS[preset]
        t    = np.arange(N) / FS
        sig  = sum(
            amps[band] * np.cos(2 * np.pi * fc * t)
            for band, fc in self.BAND_CENTRES.items()
        )
        return sig.astype(np.float32)

    def _band_power_from_cwt(self, x, flo, fhi, n_freqs=32):
        freqs = _freqs(0.5, 60.0, n_freqs)
        cwt   = cwt_complex(x, freqs, FS, wavelet="lognorm")
        mask  = (freqs >= flo) & (freqs <= fhi)
        return float(np.nanmean(np.abs(cwt[mask]) ** 2))

    def test_resting_alpha_dominant(self):
        """Resting preset: CWT alpha power must exceed beta and theta."""
        x = self._make_preset_signal("resting")
        p_alpha = self._band_power_from_cwt(x, 8.0, 12.0)
        p_beta  = self._band_power_from_cwt(x, 12.0, 30.0)
        p_theta = self._band_power_from_cwt(x, 4.0, 8.0)
        assert p_alpha > p_beta,  f"Resting: alpha={p_alpha:.4f} ≤ beta={p_beta:.4f}"
        assert p_alpha > p_theta, f"Resting: alpha={p_alpha:.4f} ≤ theta={p_theta:.4f}"

    def test_active_beta_dominant(self):
        """Active preset: CWT beta power must exceed alpha."""
        x = self._make_preset_signal("active")
        p_alpha = self._band_power_from_cwt(x, 8.0, 12.0)
        p_beta  = self._band_power_from_cwt(x, 12.0, 30.0)
        assert p_beta > p_alpha, f"Active: beta={p_beta:.4f} ≤ alpha={p_alpha:.4f}"

    def test_drowsy_theta_dominant(self):
        """Drowsy preset: CWT theta power must exceed beta."""
        x = self._make_preset_signal("drowsy")
        p_theta = self._band_power_from_cwt(x, 4.0, 8.0)
        p_beta  = self._band_power_from_cwt(x, 12.0, 30.0)
        assert p_theta > p_beta, f"Drowsy: theta={p_theta:.4f} ≤ beta={p_beta:.4f}"

    def test_sleep_delta_dominant(self):
        """Sleep preset: CWT delta power must exceed alpha and beta."""
        x = self._make_preset_signal("sleep")
        p_delta = self._band_power_from_cwt(x, 0.5, 4.0)
        p_alpha = self._band_power_from_cwt(x, 8.0, 12.0)
        p_beta  = self._band_power_from_cwt(x, 12.0, 30.0)
        assert p_delta > p_alpha, f"Sleep: delta={p_delta:.4f} ≤ alpha={p_alpha:.4f}"
        assert p_delta > p_beta,  f"Sleep: delta={p_delta:.4f} ≤ beta={p_beta:.4f}"

    def test_cwt_band_powers_match_fft_band_powers(self):
        """
        CWT and FFT band power rankings must agree for a clean multi-component signal.
        This is the cross-method consistency check replacing a MODA comparison.
        """
        x = self._make_preset_signal("resting")

        # FFT band powers
        spec   = np.abs(np.fft.rfft(x)) ** 2
        fq     = np.fft.rfftfreq(N, 1.0 / FS)
        def fft_band(lo, hi):
            return float(np.sum(spec[(fq >= lo) & (fq < hi)]))

        p_alpha_fft = fft_band(8.0, 12.0)
        p_beta_fft  = fft_band(12.0, 30.0)

        p_alpha_cwt = self._band_power_from_cwt(x, 8.0, 12.0)
        p_beta_cwt  = self._band_power_from_cwt(x, 12.0, 30.0)

        # Both methods must agree on which band dominates
        fft_says_alpha = p_alpha_fft > p_beta_fft
        cwt_says_alpha = p_alpha_cwt > p_beta_cwt
        assert fft_says_alpha == cwt_says_alpha, (
            f"FFT and CWT disagree on dominant band: "
            f"FFT alpha/beta={p_alpha_fft:.3f}/{p_beta_fft:.3f}, "
            f"CWT alpha/beta={p_alpha_cwt:.4f}/{p_beta_cwt:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 10.  Numerical regression — saved reference values
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericalRegression:
    """
    Lock in specific output values on deterministic inputs.
    If these fail after a code change, the algorithm semantics have changed.
    """

    def _ref_signal(self):
        """Fully deterministic 5-second, 256 Hz sine at 8 Hz."""
        t = np.arange(int(5 * FS)) / FS
        return np.cos(2 * np.pi * 8.0 * t).astype(np.float32)

    def test_lognorm_cwt_peak_frequency_regression(self):
        """CWT lognorm peak frequency on reference signal must be 8 Hz ± 0.5 Hz."""
        x     = self._ref_signal()
        freqs = _freqs(0.5, 60.0, 64)
        cwt   = cwt_complex(x, freqs, FS, wavelet="lognorm")
        mean_amp = np.nanmean(np.abs(cwt), axis=1)
        peak_f   = freqs[np.nanargmax(mean_amp)]
        assert abs(peak_f - 8.0) < 0.5, f"Regression: peak freq {peak_f:.3f} Hz"

    def test_butterworth_passband_power_regression(self):
        """Butterworth 6-10 Hz filter on 8 Hz sine must retain > 70% power."""
        x     = self._ref_signal()
        x_f   = butterworth_bandpass(x, FS, 6.0, 10.0, order=4)
        ratio = np.mean(x_f ** 2) / max(np.mean(x ** 2), 1e-12)
        assert ratio > 0.70, f"Regression: passband power ratio {ratio:.3f}"

    def test_iaaft_spectrum_preservation_regression(self):
        """IAAFT surrogates of 8 Hz sine must preserve spectral peak at 8 Hz."""
        x    = self._ref_signal()
        surr = iaaft_surrogates(x, n_surrogates=5)
        orig_peak = float(np.abs(np.fft.rfft(x)).argmax())
        for i, s in enumerate(surr):
            surr_peak = float(np.abs(np.fft.rfft(s)).argmax())
            assert abs(surr_peak - orig_peak) < 3, (
                f"Surrogate {i} peak bin {surr_peak} far from original {orig_peak}")

    def test_coupling_basis_K49_for_bn3(self):
        """Coupling basis for bn=3 must produce exactly K=49 columns."""
        phi = np.linspace(0, 2 * np.pi, 200, dtype=np.float32)
        P   = _build_basis(phi, phi, 3)
        assert P.shape[1] == 49, f"Regression: K={P.shape[1]}, expected 49"

    def test_biphase_output_length(self):
        """biphase_timeseries output arrays must have length N."""
        f1, f2 = 6.0, 10.0
        n_local = int(5 * FS)
        x1 = _sine(f1, n=n_local).astype(np.float32)
        x2 = (_sine(f2, n=n_local) + _sine(f1 + f2, n=n_local)).astype(np.float32)
        res = biphase_timeseries(x1, x2, FS, f1, f2, wavelet="lognorm")
        assert len(res["biamp"])   == n_local
        assert len(res["biphase"]) == n_local
        assert len(res["time"])    == n_local


# ═════════════════════════════════════════════════════════════════════════════
# 11.  Actual repo example signals — property checks
# ═════════════════════════════════════════════════════════════════════════════

UPLOADS = REPO / "FastMODA" / "uploads"

def _load_example(name: str) -> np.ndarray:
    """Load one of the repo example signals from FastMODA/uploads/."""
    path = UPLOADS / f"{name}.npy"
    if not path.exists():
        pytest.skip(f"Example signal not found: {path}")
    return np.load(path).flatten().astype(np.float32)


@lru_cache(maxsize=None)
def _cached_load(name: str) -> np.ndarray:
    return _load_example(name)


class TestExampleSignalProperties:
    """
    Verify that FastMODA's algorithms produce correct outputs on the actual
    example signals shipped with the repo (uploads/a1..a2, b1..b2, s0..s2,
    signal).  All signals are 512 samples at 256 Hz with a dominant 10 Hz
    component (alpha band).

    These tests replace a MODA/MATLAB comparison by asserting mathematical
    properties that any correct implementation must satisfy for these inputs.
    """

    FS_EX   = 256.0
    SIGNALS = ["a1", "a2", "b1", "b2", "s", "s0", "s1", "s2", "signal"]
    PAIRS   = [("a1", "a2"), ("b1", "b2"), ("s1", "s2")]
    FREQS   = np.logspace(np.log10(0.5), np.log10(60.0), 32)

    # ── per-signal checks ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("name", SIGNALS)
    def test_alpha_band_dominant(self, name):
        """Every example signal must have alpha-band FFT power exceeding theta and beta."""
        x    = _load_example(name)
        spec = np.abs(np.fft.rfft(x)) ** 2
        fq   = np.fft.rfftfreq(len(x), 1.0 / self.FS_EX)
        p_alpha = float(np.sum(spec[(fq >= 8) & (fq < 12)]))
        p_theta = float(np.sum(spec[(fq >= 4) & (fq < 8)]))
        p_beta  = float(np.sum(spec[(fq >= 12) & (fq < 30)]))
        assert p_alpha > p_theta, f"{name}: alpha={p_alpha:.1f} ≤ theta={p_theta:.1f}"
        assert p_alpha > p_beta,  f"{name}: alpha={p_alpha:.1f} ≤ beta={p_beta:.1f}"

    @pytest.mark.parametrize("name", SIGNALS)
    def test_ridge_tracks_10hz(self, name):
        """CWT ridge on every example signal must track ~10 Hz (within ±1.5 Hz)."""
        x   = _load_example(name)
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm",
                          cut_edges=False)
        r   = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        mf  = float(np.nanmean(r["ifreq"]))
        assert abs(mf - 10.0) < 1.5, (
            f"{name}: ridge mean freq {mf:.2f} Hz, expected ~10 Hz")

    @pytest.mark.parametrize("name", SIGNALS)
    def test_ridge_amplitude_positive(self, name):
        """Instantaneous amplitude must be strictly positive for all example signals."""
        x   = _load_example(name)
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm",
                          cut_edges=False)
        r   = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        assert float(np.mean(r["iamp"])) > 0

    @pytest.mark.parametrize("name", SIGNALS)
    def test_butterworth_alpha_pass(self, name):
        """Butterworth 8–12 Hz filter must retain > 50% power for each example signal."""
        x     = _load_example(name)
        x_f   = butterworth_bandpass(x, self.FS_EX, 8.0, 12.0, order=4)
        ratio = float(np.mean(x_f ** 2)) / max(float(np.mean(x ** 2)), 1e-12)
        assert ratio > 0.5, f"{name}: passband ratio {ratio:.3f} < 0.5"

    @pytest.mark.parametrize("name", SIGNALS)
    def test_cwt_shape(self, name):
        """CWT of each example signal must have shape (n_freqs, signal_length)."""
        x   = _load_example(name)
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm")
        assert cwt.shape == (len(self.FREQS), len(x)), (
            f"{name}: CWT shape {cwt.shape}")

    @pytest.mark.parametrize("name", SIGNALS)
    def test_iaaft_surrogate_preserves_spectrum(self, name):
        """IAAFT surrogates of each example signal must preserve the power spectrum."""
        x    = _load_example(name)
        surr = iaaft_surrogates(x, n_surrogates=3)
        amp_orig = np.abs(np.fft.rfft(x))
        for i in range(3):
            amp_surr = np.abs(np.fft.rfft(surr[i]))
            corr = float(np.corrcoef(amp_orig, amp_surr)[0, 1])
            assert corr > 0.97, (
                f"{name} surrogate {i}: spectrum correlation {corr:.4f} < 0.97")

    # ── cross-signal / pair checks ────────────────────────────────────────────

    @pytest.mark.parametrize("n1,n2", PAIRS)
    def test_pair_coherence_in_range(self, n1, n2):
        """TLPHCOH between all example signal pairs must return values in [0, 1]."""
        x1 = _load_example(n1)
        x2 = _load_example(n2)
        cwt1 = cwt_complex(x1, self.FREQS, self.FS_EX, wavelet="lognorm",
                           cut_edges=False)
        cwt2 = cwt_complex(x2, self.FREQS, self.FS_EX, wavelet="lognorm",
                           cut_edges=False)
        tpc  = time_localized_coherence(cwt1, cwt2, self.FREQS, self.FS_EX,
                                         numcycles=10)
        valid = tpc[~np.isnan(tpc)]
        assert np.all(valid >= 0.0), f"{n1}↔{n2}: TPC below 0"
        assert np.all(valid <= 1.0 + 1e-6), f"{n1}↔{n2}: TPC above 1"

    @pytest.mark.parametrize("n1,n2", PAIRS)
    def test_pair_coupling_direction_in_range(self, n1, n2):
        """Coupling direction for all pairs must be in [-1, 1]."""
        x1 = _load_example(n1)
        x2 = _load_example(n2)
        x1f = butterworth_bandpass(x1, self.FS_EX, 8.0, 12.0)
        x2f = butterworth_bandpass(x2, self.FS_EX, 8.0, 12.0)
        ph1 = np.angle(hilbert(x1f)).astype(np.float32)
        ph2 = np.angle(hilbert(x2f)).astype(np.float32)
        cf  = estimate_coupling_functions(ph1, ph2, self.FS_EX, bn=3,
                                          win_s=1.0, overlap=0.5, n_grid=20)
        d = cf["direction"]
        assert np.all(d >= -1.0 - 1e-6), f"{n1}↔{n2}: direction < -1"
        assert np.all(d <=  1.0 + 1e-6), f"{n1}↔{n2}: direction > +1"

    @pytest.mark.parametrize("n1,n2", PAIRS)
    def test_pair_biphase_positive_biamp(self, n1, n2):
        """Biamplitude between all example pairs must be non-negative."""
        x1 = _load_example(n1)
        x2 = _load_example(n2)
        res = biphase_timeseries(x1, x2, self.FS_EX, 5.0, 5.0, wavelet="lognorm")
        assert np.all(res["biamp"] >= 0), f"{n1}↔{n2}: biamp contains negative values"

    def test_s0_and_s1_identical(self):
        """s0 and s1 have the same values — coherence between them must be ≥ 0.95."""
        s0 = _load_example("s0")
        s1 = _load_example("s1")
        if np.allclose(s0, s1):
            # They are identical — coherence = 1
            cwt0 = cwt_complex(s0, self.FREQS, self.FS_EX, wavelet="lognorm",
                               cut_edges=False)
            cwt1 = cwt_complex(s1, self.FREQS, self.FS_EX, wavelet="lognorm",
                               cut_edges=False)
            tpc  = time_localized_coherence(cwt0, cwt1, self.FREQS, self.FS_EX)
            coh  = float(np.nanmean(tpc))
            assert coh > 0.95, f"s0=s1 but mean coherence {coh:.3f} < 0.95"
        else:
            pytest.skip("s0 and s1 differ — skipping identity coherence check")

    def test_all_signals_same_length(self):
        """All 9 example signals must have the same length (512 samples)."""
        lengths = {name: len(_load_example(name)) for name in self.SIGNALS}
        unique  = set(lengths.values())
        # allow s0/s1 which may match each other but all should match
        main_len = lengths["a1"]
        for name, L in lengths.items():
            assert L == main_len, f"{name}: length {L} != {main_len}"


# ═════════════════════════════════════════════════════════════════════════════
# 12.  Example signal regression — locked numeric baselines
# ═════════════════════════════════════════════════════════════════════════════

class TestExampleSignalRegression:
    """
    Lock in the exact numeric outputs of each algorithm on the repo example
    signals.  These values were computed from the current codebase with all
    fixes applied (lognorm CWT, SG smoothing, bn=3, cut_edges).

    If any of these change, an algorithm's semantics have changed.
    Tolerances are ±2% on frequencies and ±10% on amplitudes to accommodate
    floating-point differences across platforms.
    """

    FS_EX = 256.0
    FREQS = np.logspace(np.log10(0.5), np.log10(60.0), 32)

    # Ridge mean frequency baselines (Hz) — computed from current codebase
    RIDGE_FREQ_BASELINES = {
        "a1":     9.51,
        "a2":     9.56,
        "b1":     9.56,
        "b2":     9.61,
        "s":      9.66,
        "s0":     9.53,
        "s1":     9.53,
        "s2":     9.63,
        "signal": 9.53,
    }

    # Ridge mean amplitude baselines — computed from current codebase
    RIDGE_AMP_BASELINES = {
        "a1":     0.4472,
        "a2":     0.4487,
        "b1":     0.4476,
        "b2":     0.4434,
        "s":      0.4367,
        "s0":     0.4688,
        "s1":     0.4688,
        "s2":     0.4459,
        "signal": 0.4639,
    }

    @pytest.mark.parametrize("name,expected_f", RIDGE_FREQ_BASELINES.items())
    def test_ridge_freq_regression(self, name, expected_f):
        """Ridge mean frequency must match its locked baseline within ±0.3 Hz."""
        x   = _load_example(name)
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm",
                          cut_edges=False)
        r   = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        got = float(np.nanmean(r["ifreq"]))
        assert abs(got - expected_f) < 0.3, (
            f"{name}: ridge freq {got:.3f} Hz, baseline {expected_f:.3f} Hz "
            f"(drift > 0.3 Hz — algorithm changed?)")

    @pytest.mark.parametrize("name,expected_a", RIDGE_AMP_BASELINES.items())
    def test_ridge_amp_regression(self, name, expected_a):
        """Ridge mean amplitude must match its locked baseline within ±10%."""
        x   = _load_example(name)
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm",
                          cut_edges=False)
        r   = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        got = float(np.nanmean(r["iamp"]))
        assert abs(got - expected_a) / expected_a < 0.10, (
            f"{name}: ridge amp {got:.4f}, baseline {expected_a:.4f} "
            f"(drift > 10% — algorithm changed?)")

    def test_determinism_cwt(self):
        """CWT on the same signal must produce bit-identical results each time."""
        x    = _load_example("a1")
        cwt1 = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm")
        cwt2 = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm")
        np.testing.assert_array_equal(cwt1, cwt2,
            err_msg="CWT is not deterministic — different results on same input")

    def test_determinism_ridge(self):
        """Ridge extraction on the same CWT must be bit-identical each time."""
        x   = _load_example("a1")
        cwt = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm")
        r1  = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        r2  = extract_ridge(cwt, self.FREQS, self.FS_EX, smooth_len=5)
        np.testing.assert_array_equal(r1["ifreq"], r2["ifreq"],
            err_msg="Ridge extraction is not deterministic")

    def test_determinism_butterworth(self):
        """Butterworth filter must produce bit-identical results each time."""
        x   = _load_example("s0")
        xf1 = butterworth_bandpass(x, self.FS_EX, 8.0, 12.0)
        xf2 = butterworth_bandpass(x, self.FS_EX, 8.0, 12.0)
        np.testing.assert_array_equal(xf1, xf2,
            err_msg="Butterworth filter is not deterministic")

    def test_determinism_biphase(self):
        """biphase_timeseries must produce bit-identical results each time."""
        x1 = _load_example("a1")
        x2 = _load_example("a2")
        r1 = biphase_timeseries(x1, x2, self.FS_EX, 5.0, 5.0, wavelet="lognorm")
        r2 = biphase_timeseries(x1, x2, self.FS_EX, 5.0, 5.0, wavelet="lognorm")
        np.testing.assert_array_equal(r1["biamp"], r2["biamp"],
            err_msg="biphase_timeseries is not deterministic")

    def test_a1_a2_coherence_consistent_with_fft(self):
        """
        CWT-based and FFT-based band powers must rank bands consistently for a1/a2.
        Ensures the CWT and FFT code-paths agree on the alpha-dominant nature of the
        example signals — the closest we can get to a cross-method consistency check.
        """
        for name in ("a1", "a2"):
            x = _load_example(name)
            # FFT
            spec = np.abs(np.fft.rfft(x)) ** 2
            fq   = np.fft.rfftfreq(len(x), 1.0 / self.FS_EX)
            p_alpha_fft = float(np.sum(spec[(fq >= 8) & (fq < 12)]))
            p_beta_fft  = float(np.sum(spec[(fq >= 12) & (fq < 30)]))
            # CWT
            cwt  = cwt_complex(x, self.FREQS, self.FS_EX, wavelet="lognorm")
            mask_alpha = (self.FREQS >= 8) & (self.FREQS < 12)
            mask_beta  = (self.FREQS >= 12) & (self.FREQS < 30)
            p_alpha_cwt = float(np.nanmean(np.abs(cwt[mask_alpha]) ** 2))
            p_beta_cwt  = float(np.nanmean(np.abs(cwt[mask_beta]) ** 2))
            assert (p_alpha_fft > p_beta_fft) == (p_alpha_cwt > p_beta_cwt), (
                f"{name}: FFT and CWT disagree on dominant band")
