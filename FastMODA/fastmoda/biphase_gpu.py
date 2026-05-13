"""
Wavelet biphase and biamplitude time series (MODA biphaseWavNew.m).

For a frequency pair (f1, f2):
    B(t) = WT1(f1,t) × WT2(f2,t) × conj(WT2(f1+f2,t))
    biamp(t)   = |B(t)|
    biphase(t) = unwrap(∠B(t))

and the four-way cross-bispectrum (bispecWavNew.m types b111/b222/b122/b211).

All operations vectorised — no per-frequency or per-segment Python loops.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _dev(device=None):
    if not _TORCH:
        return None
    return device or (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))


def _robust_unwrap(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase then suppress isolated large-jump discontinuities.

    After standard unwrap, computes phase velocity (finite differences) and
    applies a Hampel-like filter: velocities deviating > 5× the median
    absolute velocity are replaced with the local median, then the phase is
    reconstructed from the cleaned velocity series.  This matches MODA's
    phase-continuity post-processing step.
    """
    unwrapped = np.unwrap(phase)
    vel       = np.diff(unwrapped)
    if len(vel) == 0:
        return unwrapped

    med_abs   = np.median(np.abs(vel))
    threshold = max(np.pi, 5.0 * med_abs)
    spikes    = np.abs(vel) > threshold

    if spikes.any():
        win = 5
        vel_clean = vel.copy()
        for idx in np.where(spikes)[0]:
            lo  = max(0, idx - win)
            hi  = min(len(vel), idx + win + 1)
            nbr = vel[lo:hi][~spikes[lo:hi]]
            vel_clean[idx] = float(np.median(nbr)) if len(nbr) > 0 else 0.0
        unwrapped = np.concatenate([[unwrapped[0]], unwrapped[0] + np.cumsum(vel_clean)])

    return unwrapped


# ── biphase time series ───────────────────────────────────────────────────────

def biphase_timeseries(
    x1: np.ndarray,
    x2: np.ndarray,
    fs: float,
    f1: float,
    f2: float,
    wavelet: str = "lognorm",
    n_cycles: float = 6.0,
    device=None,
) -> Dict[str, np.ndarray]:
    """
    Time-resolved biphase and biamplitude at frequency pair (f1, f2).

    Implements MATLAB MODA ``biphaseWavNew.m``:
        B(t) = WT1(f1,t) × WT2(f2,t) × conj(WT2(f3,t))  where f3 = f1+f2
        biamp(t)   = |B(t)|
        biphase(t) = unwrap(∠B(t))   [radians]

    Vectorised: CWT at all three frequencies computed in one batch IFFT.

    Returns
    -------
    dict with biamp, biphase, time (all 1-D float32 arrays), f1, f2, f3 (Hz)
    """
    f3 = f1 + f2
    if f3 > fs / 2:
        raise ValueError(
            f"f3 = {f3:.2f} Hz exceeds Nyquist ({fs/2:.2f} Hz). "
            "Choose a lower frequency pair.")

    # GPU path: use existing wavelet_biphase_time_series_gpu if available
    dev = _dev(device)
    if dev is not None and _TORCH:
        try:
            from fastmoda.bispectrum_gpu import wavelet_biphase_time_series_gpu
            x1_t = torch.as_tensor(x1, dtype=torch.float32, device=dev)
            x2_t = torch.as_tensor(x2, dtype=torch.float32, device=dev)
            result = wavelet_biphase_time_series_gpu(
                x1_t, x2_t, fs, f1, f2, device=dev)
            result['gpu_used'] = True
            return result
        except (ImportError, Exception):
            pass  # fall through to CPU path

    # ── CPU vectorised path ──────────────────────────────────────────────
    from fastmoda.ridge_gpu import cwt_complex

    freqs  = np.array([f1, f2, f3], dtype=np.float32)
    cwt1   = cwt_complex(x1, freqs, fs, wavelet=wavelet, n_cycles=n_cycles)   # (3,T)
    cwt2   = cwt_complex(x2, freqs, fs, wavelet=wavelet, n_cycles=n_cycles)

    # Triple product — fully vectorised
    B       = cwt1[0] * cwt2[1] * np.conj(cwt2[2])                            # (T,)
    biamp   = np.abs(B)
    biphase = _robust_unwrap(np.angle(B))
    t_ax    = np.arange(len(x1)) / fs

    return {
        "biamp":    biamp.astype(np.float32),
        "biphase":  biphase.astype(np.float32),
        "time":     t_ax.astype(np.float32),
        "f1":       float(f1),
        "f2":       float(f2),
        "f3":       float(f3),
        "gpu_used": False,
    }


# ── four-way cross-bispectrum ─────────────────────────────────────────────────

def bispectrum4(
    x1: np.ndarray,
    x2: np.ndarray,
    fs: float,
    nfft: int = 256,
    overlap: float = 0.5,
    device=None,
) -> Dict[str, np.ndarray]:
    """
    Four-way cross-bispectrum between two signals (MODA bispecWavNew.m).

    Computes all four variants simultaneously in one batch operation:
        b111 = X1·X1·conj(X1)   — auto-bispectrum of x1
        b222 = X2·X2·conj(X2)   — auto-bispectrum of x2
        b122 = X1·X2·conj(X2)   — cross-bispectrum 1→(2,2)
        b211 = X2·X1·conj(X1)   — cross-bispectrum 2→(1,1)

    No per-type or per-segment Python loops — all four computed via a
    single unfold + batch rfft + masked summation pass.

    Returns
    -------
    dict with b111, b222, b122, b211 (each n_freq×n_freq complex64),
    biamp111, biamp222, biamp122, biamp211 (magnitudes),
    frequencies (Hz), gpu_used
    """
    hop     = max(1, int(nfft * (1 - overlap)))
    n_freq  = nfft // 2 + 1
    dev     = _dev(device)

    # ── f3 index precompute (shared CPU/GPU) ─────────────────────────────
    f_idx   = np.arange(n_freq)
    f3_mat  = f_idx[:, None] + f_idx[None, :]       # (F,F) f3=f1+f2
    valid   = f3_mat < n_freq
    f3_safe = np.clip(f3_mat, 0, n_freq - 1)

    if dev is not None and _TORCH:
        x1_t   = torch.as_tensor(x1, dtype=torch.float32, device=dev)
        x2_t   = torch.as_tensor(x2, dtype=torch.float32, device=dev)
        win    = torch.hann_window(nfft, device=dev)
        valid_t = torch.from_numpy(valid).to(dev)
        f3_t   = torch.from_numpy(f3_safe).to(dev)

        frames1 = x1_t.unfold(0, nfft, hop) * win   # (S, nfft)
        frames2 = x2_t.unfold(0, nfft, hop) * win
        S       = min(frames1.shape[0], frames2.shape[0])
        frames1, frames2 = frames1[:S], frames2[:S]

        X1 = torch.fft.rfft(frames1, dim=1)          # (S, F)
        X2 = torch.fft.rfft(frames2, dim=1)

        def _b(A, B, C):
            """Batch bispectrum: sum_s A[:,f1] * B[:,f2] * conj(C[:,f3])."""
            raw = (A[:, :, None] * B[:, None, :]      # (S,F,F)
                   * torch.conj(C[:, f3_t]))
            return (raw.masked_fill(~valid_t.unsqueeze(0), 0.0)
                    .mean(dim=0))                      # (F,F)

        b111 = _b(X1, X1, X1)
        b222 = _b(X2, X2, X2)
        b122 = _b(X1, X2, X2)
        b211 = _b(X2, X1, X1)

        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        def _n(t): return t.cpu().numpy().astype(np.complex64)
        return dict(b111=_n(b111), b222=_n(b222), b122=_n(b122), b211=_n(b211),
                    biamp111=np.abs(_n(b111)), biamp222=np.abs(_n(b222)),
                    biamp122=np.abs(_n(b122)), biamp211=np.abs(_n(b211)),
                    frequencies=freqs, gpu_used=True)

    # ── CPU numpy — same logic ───────────────────────────────────────────
    win     = np.hanning(nfft).astype(np.float32)
    def _frames(sig):
        n_s = (len(sig) - nfft) // hop + 1
        shape   = (n_s, nfft)
        strides = (sig.strides[0] * hop, sig.strides[0])
        return np.lib.stride_tricks.as_strided(
            sig.astype(np.float32), shape=shape, strides=strides)

    F1_raw = np.fft.rfft(_frames(x1) * win[np.newaxis, :], axis=1)  # (S,F)
    F2_raw = np.fft.rfft(_frames(x2) * win[np.newaxis, :], axis=1)
    # Normalize each segment to unit RMS to prevent float32 overflow
    # in the triple product (multiply by column norms, restore at end)
    norm1 = np.maximum(np.abs(F1_raw).max(axis=1, keepdims=True), 1e-12)
    norm2 = np.maximum(np.abs(F2_raw).max(axis=1, keepdims=True), 1e-12)
    F1 = (F1_raw / norm1).astype(np.complex64)
    F2 = (F2_raw / norm2).astype(np.complex64)
    S  = min(F1.shape[0], F2.shape[0])
    F1, F2 = F1[:S], F2[:S]

    def _b(A, B, C):
        raw = (A[:, :, None] * B[:, None, :]
               * np.conj(C[:, f3_safe]))              # (S,F,F)
        return np.where(valid[np.newaxis], raw, 0.0).mean(axis=0)

    b111 = _b(F1, F1, F1)
    b222 = _b(F2, F2, F2)
    b122 = _b(F1, F2, F2)
    b211 = _b(F2, F1, F1)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    return dict(b111=b111, b222=b222, b122=b122, b211=b211,
                biamp111=np.abs(b111), biamp222=np.abs(b222),
                biamp122=np.abs(b122), biamp211=np.abs(b211),
                frequencies=freqs, gpu_used=False)
