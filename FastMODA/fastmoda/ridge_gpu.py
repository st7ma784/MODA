"""
Vectorized ridge extraction, time-localized phase coherence (TPC),
and analytic Morlet CWT — all GPU-first, zero Python-level loops.

Design principles
-----------------
* No per-sample or per-frequency Python for-loops.
* No explicit if/else branches in the computational kernels;
  conditional logic is replaced by masked tensor operations.
* GPU path via PyTorch (batch FFT + gather).
* CPU path via NumPy with identical vectorised logic.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ── device helper ─────────────────────────────────────────────────────────────

def _dev(device=None):
    """Return a torch.device or None (→ use numpy path)."""
    if not _TORCH:
        return None
    return device or (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))


# ── vectorised complex Morlet CWT ─────────────────────────────────────────────

def _wavelet_filter_bank(
    nu: np.ndarray,
    freqs: np.ndarray,
    wavelet: str = "lognorm",
    n_cycles: float = 6.0,
) -> np.ndarray:
    """
    Build a frequency-domain filter bank for any supported wavelet.

    All wavelet families are computed via broadcasting — no per-frequency loop.
    Conditional logic uses masked multiplication, not if-statements.

    Supported wavelets
    ------------------
    lognorm  (MODA default) — log-Gaussian, uniform log-freq resolution
    morlet   — complex Gaussian, good all-round choice
    bump     — compactly supported, sharpest spectral localisation

    Returns
    -------
    Psi : (n_freqs, N) real filter bank, analytic (negative freqs zeroed)
    """
    nu_br = nu[np.newaxis, :]       # (1, N)
    f_br  = freqs[:, np.newaxis]    # (NF, 1)
    ratio = np.maximum(nu_br / f_br, 1e-12)  # ν/f, guarded > 0

    if wavelet == "lognorm":
        sigma = 1.0 / n_cycles
        Psi   = np.exp(-0.5 * np.log(ratio) ** 2 / sigma ** 2)

    elif wavelet == "morlet":
        Psi   = np.exp(-0.5 * n_cycles ** 2 * (ratio - 1.0) ** 2)

    elif wavelet == "bump":
        eps    = 0.5 / n_cycles          # relative half-support width
        u      = (ratio - 1.0) / eps
        u_sq   = np.clip(u ** 2, 0.0, 1.0 - 1e-8)
        inside = (np.abs(u) < 1.0).astype(np.float32)
        # exp(1 − 1/(1−u²)) — NaN-safe via clip; multiply by inside (no if!)
        Psi    = inside * np.exp(1.0 - 1.0 / (1.0 - u_sq))

    else:
        raise ValueError(f"Unknown wavelet '{wavelet}'. Choose: lognorm, morlet, bump")

    # Analytic signal: zero negative frequencies — element-wise mask, no if
    Psi = Psi * (nu >= 0).astype(np.float32)[np.newaxis, :]
    return Psi.astype(np.float32)


def _wavelet_filter_bank_torch(nu, freqs, wavelet, n_cycles, dev):
    """Torch version of _wavelet_filter_bank."""
    nu_br = nu.unsqueeze(0)    # (1, N)
    f_br  = freqs.unsqueeze(1) # (NF, 1)
    ratio = torch.clamp(nu_br / f_br, min=1e-12)

    if wavelet == "lognorm":
        sigma = 1.0 / n_cycles
        Psi   = torch.exp(-0.5 * torch.log(ratio) ** 2 / sigma ** 2)

    elif wavelet == "morlet":
        Psi   = torch.exp(-0.5 * n_cycles ** 2 * (ratio - 1.0) ** 2)

    elif wavelet == "bump":
        eps    = 0.5 / n_cycles
        u      = (ratio - 1.0) / eps
        u_sq   = torch.clamp(u ** 2, 0.0, 1.0 - 1e-8)
        inside = (u.abs() < 1.0).float()
        Psi    = inside * torch.exp(1.0 - 1.0 / (1.0 - u_sq))

    else:
        raise ValueError(f"Unknown wavelet '{wavelet}'")

    Psi = Psi * (nu >= 0).float().unsqueeze(0)
    return Psi


def cone_of_influence(
    freqs: np.ndarray,
    T: int,
    fs: float,
    n_cycles: float = 6.0,
    wavelet: str = "lognorm",
) -> np.ndarray:
    """
    Compute a (n_freqs, T) boolean mask where True = edge-affected (COI).

    Based on the e-folding time of each wavelet's amplitude envelope:
        Lognorm / Morlet : τ_COI(f) = sqrt(2) · σ_t(f)
        Bump             : τ_COI(f) = sqrt(2) · ε / f

    Fully vectorised via broadcasting — no loops.
    """
    times     = np.arange(T) / fs                                # (T,)
    T_total   = T / fs

    if wavelet in ("lognorm",):
        sigma_t   = 1.0 / (n_cycles * 2.0 * np.pi * freqs)     # (NF,)
    elif wavelet == "morlet":
        sigma_t   = n_cycles / (2.0 * np.pi * freqs)
    elif wavelet == "bump":
        sigma_t   = (0.5 / n_cycles) / freqs
    else:
        sigma_t   = n_cycles / (2.0 * np.pi * freqs)            # morlet fallback

    half      = (np.sqrt(2) * sigma_t)[:, np.newaxis]           # (NF, 1)
    t_br      = times[np.newaxis, :]                             # (1, T)
    return (t_br < half) | (t_br > T_total - half)              # (NF, T) bool


def _pad(x: np.ndarray, n: int, mode: str) -> np.ndarray:
    """
    Pad *x* by *n* samples on each side using the chosen mode.
    Supported: symmetric, periodic, zero (or zeros), predictive.
    """
    if mode in ("zero", "zeros", "0"):
        return np.pad(x, n, mode="constant", constant_values=0)
    if mode == "symmetric":
        return np.pad(x, n, mode="reflect")
    if mode == "periodic":
        return np.pad(x, n, mode="wrap")
    if mode == "predictive":
        # AR(1) prediction using local mean of edge samples (stable, avoids ramps)
        edge = min(8, max(1, len(x) // 8))
        left  = np.full(n, float(np.mean(x[:edge])),  dtype=x.dtype)
        right = np.full(n, float(np.mean(x[-edge:])), dtype=x.dtype)
        return np.concatenate([left, x, right])
    # default: zero
    return np.pad(x, n, mode="constant", constant_values=0)


def nv_to_freqs(fmin: float, fmax: float, nv: int) -> np.ndarray:
    """
    Compute log-spaced frequency array with *nv* voices per octave.
    Equivalent to MODA's ``nv`` parameter in ``wt.m``.
    """
    n_freqs = max(1, int(np.log2(fmax / fmin) * nv) + 1)
    return fmin * 2 ** (np.arange(n_freqs) / nv)


def cwt_complex(
    x: np.ndarray,
    freqs: np.ndarray,
    fs: float,
    wavelet: str = "lognorm",
    n_cycles: float = 6.0,
    padding: str = "symmetric",
    cut_edges: bool = False,
    device=None,
) -> np.ndarray:
    """
    Fully-vectorised analytic CWT via batch FFT — no per-frequency loop.

    All wavelet families (lognorm / morlet / bump) are computed by building
    a frequency-domain filter bank via broadcasting, then applying a single
    batch IFFT across all target frequencies simultaneously.

    Parameters
    ----------
    wavelet   : 'lognorm' (MODA default), 'morlet', or 'bump'
    n_cycles  : resolution parameter (higher = better frequency, worse time)
    padding   : 'symmetric' (MODA default), 'periodic', 'zero', 'predictive'
    cut_edges : if True, set edge-affected coefficients to NaN (MODA CutEdges)

    Returns
    -------
    cwt : complex64 ndarray, shape (n_freqs, n_times)
          Edge regions are NaN when cut_edges=True.
    """
    T_orig = len(x)
    # Padding: extend signal by half the longest wavelet support
    pad_len = min(T_orig, int(np.ceil(np.sqrt(2) * n_cycles /
                                       (2 * np.pi * freqs.min() + 1e-12) * fs)))
    x_pad  = _pad(x.astype(np.float32), pad_len, padding)
    N      = len(x_pad)
    dev    = _dev(device)

    if dev is not None:
        x_t  = torch.as_tensor(x_pad, dtype=torch.float32, device=dev)
        f_t  = torch.as_tensor(freqs, dtype=torch.float32, device=dev)
        X    = torch.fft.fft(x_t, n=N)
        nu   = torch.fft.fftfreq(N, 1.0 / fs).to(dev)
        Psi  = _wavelet_filter_bank_torch(nu, f_t, wavelet, n_cycles, dev)
        cwt_pad = torch.fft.ifft(X.unsqueeze(0) * Psi, dim=1)
        cwt = cwt_pad[:, pad_len:pad_len + T_orig].cpu().numpy().astype(np.complex64)
    else:
        X       = np.fft.fft(x_pad, n=N)
        nu      = np.fft.fftfreq(N, 1.0 / fs)
        Psi     = _wavelet_filter_bank(nu, freqs, wavelet, n_cycles)
        cwt_pad = np.fft.ifft(X[np.newaxis, :] * Psi, axis=1)
        cwt     = cwt_pad[:, pad_len:pad_len + T_orig].astype(np.complex64)

    if cut_edges:
        mask = cone_of_influence(freqs, T_orig, fs, n_cycles, wavelet)
        cwt[mask] = np.nan

    return cwt


# Backward-compat alias
def morlet_cwt_complex(x, freqs, fs, n_cycles=6.0, device=None):
    """Morlet CWT (alias for cwt_complex with wavelet='morlet')."""
    return cwt_complex(x, freqs, fs, wavelet="morlet", n_cycles=n_cycles, device=device)


# ── ridge extraction ──────────────────────────────────────────────────────────

def extract_ridge(
    cwt_complex: np.ndarray,
    freqs: np.ndarray,
    fs: float,
    smooth_len: int = 0,
    device=None,
) -> Dict[str, np.ndarray]:
    """
    Vectorised ridge extraction from a complex CWT.

    Strategy
    --------
    1. Ridge index = argmax |CWT| along the frequency axis.
    2. Optional smoothing via 1-D box convolution (``F.conv1d``).
    3. Gather the complex CWT value at the ridge using advanced indexing.
    4. Instantaneous amplitude = |cwt_ridge|, phase = angle(cwt_ridge).
    5. Reconstruction = iamp × cos(iphi).

    No loops; all operations are element-wise or reduction over batch dims.
    """
    NF, NT = cwt_complex.shape
    dev = _dev(device)

    if dev is not None:
        W  = torch.as_tensor(cwt_complex, dtype=torch.complex64, device=dev)
        Ft = torch.as_tensor(freqs, dtype=torch.float32, device=dev)

        amp       = torch.abs(W)                                  # (NF, NT)
        ridge_idx = torch.argmax(amp, dim=0)                      # (NT,)

        # Ridge smoothing via Savitzky-Golay (polyorder 3, odd window)
        if smooth_len > 1:
            from scipy.signal import savgol_filter
            win  = smooth_len | 1
            poly = min(3, win - 1)
            ri_np = savgol_filter(
                ridge_idx.cpu().float().numpy(), win, poly)
            ridge_idx = (torch.from_numpy(ri_np)
                         .round().long().clamp(0, NF - 1).to(dev))

        t_idx      = torch.arange(NT, device=dev)
        cwt_ridge  = W[ridge_idx, t_idx]                          # (NT,)

        iamp  = torch.abs(cwt_ridge)
        iphi  = torch.angle(cwt_ridge)
        ifreq = Ft[ridge_idx]
        recon = iamp * torch.cos(iphi)

        return {k: v.cpu().numpy() for k, v in
                (("ifreq", ifreq), ("iamp", iamp),
                 ("iphi", iphi),   ("recon", recon))}

    # ── CPU numpy ────────────────────────────────────────────────────────
    amp       = np.abs(cwt_complex)
    ridge_idx = np.argmax(amp, axis=0)                            # (NT,)

    if smooth_len > 1:
        from scipy.signal import savgol_filter
        win       = smooth_len | 1
        poly      = min(3, win - 1)
        ridge_idx = np.clip(
            savgol_filter(ridge_idx.astype(float), win, poly).round().astype(int),
            0, NF - 1)

    t_idx     = np.arange(NT)
    cwt_ridge = cwt_complex[ridge_idx, t_idx]

    iamp  = np.abs(cwt_ridge)
    iphi  = np.angle(cwt_ridge)
    ifreq = freqs[ridge_idx]
    recon = iamp * np.cos(iphi)

    return {"ifreq": ifreq, "iamp": iamp, "iphi": iphi, "recon": recon}


# ── time-localised phase coherence (TPC) ─────────────────────────────────────

def time_localized_coherence(
    cwt1: np.ndarray,
    cwt2: np.ndarray,
    freqs: np.ndarray,
    fs: float,
    numcycles: int = 10,
    device=None,
) -> np.ndarray:
    """
    Vectorised time-localised wavelet phase coherence (MODA ``tlphcoh``).

    IMPORTANT: pass CWTs computed with ``cut_edges=False``.  NaN edge values
    propagate through the cumsum and make the whole TPC array NaN.
    TPC applies its own edge masking (the ``valid`` mask returned as NaN rows).

    For each frequency *f* the sliding window length is
        W(f) = round(numcycles / f × fs)  [odd, ≥ 1]
    The sliding mean is computed in one shot via cumsum + gather, eliminating
    any per-frequency loop.

    Returns
    -------
    TPC : float32 ndarray, shape (n_freqs, n_times), values ∈ [0, 1]
          NaN at edge regions where the full window is unavailable.
    """
    NF, T = cwt1.shape
    dev   = _dev(device)

    if dev is not None:
        W1  = torch.as_tensor(cwt1, dtype=torch.complex64, device=dev)
        W2  = torch.as_tensor(cwt2, dtype=torch.complex64, device=dev)
        Ft  = torch.as_tensor(freqs, dtype=torch.float32, device=dev)

        # Instantaneous phase coherence: unit-magnitude phasors  (NF, T)
        IPC = torch.exp(1j * torch.angle(W1 * torch.conj(W2)))

        # Adaptive window per frequency — vectorised
        raw_win = (numcycles / Ft * fs).round().long()
        raw_win = raw_win + 1 - raw_win % 2          # odd
        raw_win = raw_win.clamp(1, T)                 # (NF,)
        hw      = raw_win // 2                        # (NF,)

        # Cumulative sum for sliding mean (NF, T+1)
        cumIPC = torch.cat([
            torch.zeros(NF, 1, dtype=torch.complex64, device=dev),
            torch.cumsum(IPC, dim=1),
        ], dim=1)

        # Per-element gather indices — no loop, pure broadcasting
        t_idx = torch.arange(T, device=dev).long().unsqueeze(0)   # (1, T)
        hw2   = hw.unsqueeze(1)                                    # (NF, 1)

        start_idx = (t_idx - hw2).clamp(0, T)        # (NF, T)
        end_idx   = (t_idx + hw2 + 1).clamp(0, T)    # (NF, T)

        # Gather complex cumsum values — via real/imag split for compatibility
        cum_ri = torch.view_as_real(cumIPC)                        # (NF, T+1, 2)
        idx_s  = start_idx.unsqueeze(-1).expand(-1, -1, 2)        # (NF, T, 2)
        idx_e  = end_idx.unsqueeze(-1).expand(-1, -1, 2)

        cum_start = torch.view_as_complex(
            torch.gather(cum_ri, 1, idx_s).contiguous())          # (NF, T)
        cum_end   = torch.view_as_complex(
            torch.gather(cum_ri, 1, idx_e).contiguous())

        window_count = (end_idx - start_idx).float().clamp(min=1)
        TPC = torch.abs(cum_end - cum_start) / window_count        # (NF, T)

        # NaN at edges — via masked_fill (no if!)
        valid = (t_idx >= hw2) & (t_idx < T - hw2)
        TPC   = TPC.masked_fill(~valid, float("nan"))

        return TPC.float().cpu().numpy()

    # ── CPU numpy ────────────────────────────────────────────────────────
    IPC     = np.exp(1j * np.angle(cwt1 * np.conj(cwt2)))         # (NF, T)
    raw_win = np.round(numcycles / freqs * fs).astype(int)
    raw_win = raw_win + 1 - raw_win % 2
    raw_win = np.clip(raw_win, 1, T)
    hw      = raw_win // 2

    cumIPC  = np.concatenate([
        np.zeros((NF, 1), dtype=complex),
        np.cumsum(IPC, axis=1),
    ], axis=1)                                                      # (NF, T+1)

    t_idx = np.arange(T)[np.newaxis, :]                            # (1, T)
    hw2   = hw[:, np.newaxis]                                      # (NF, 1)

    start_idx = np.clip(t_idx - hw2, 0, T).astype(int)
    end_idx   = np.clip(t_idx + hw2 + 1, 0, T).astype(int)

    # Vectorised gather via fancy indexing
    f_idx     = np.arange(NF)[:, np.newaxis]
    cum_start = cumIPC[f_idx, start_idx]
    cum_end   = cumIPC[f_idx, end_idx]

    wcount    = np.maximum(end_idx - start_idx, 1).astype(float)
    TPC       = np.abs(cum_end - cum_start) / wcount               # (NF, T)

    valid     = (t_idx >= hw2) & (t_idx < T - hw2)
    TPC[~valid] = np.nan

    return TPC.astype(np.float32)
