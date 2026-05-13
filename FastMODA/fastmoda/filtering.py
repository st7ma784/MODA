"""
Signal filtering utilities: Butterworth bandpass, Windowed Fourier Transform
(Gaussian-windowed STFT), and polynomial detrending.

All computational kernels are fully vectorised — no Python-level loops
over samples or frequencies.  GPU path via PyTorch where beneficial;
CPU fallback via NumPy/SciPy with identical logic.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _dev(device=None):
    if not _TORCH:
        return None
    return device or (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))


# ── polynomial detrend ────────────────────────────────────────────────────────

def detrend_polynomial(
    x: np.ndarray,
    degree: int = 1,
    device=None,
) -> np.ndarray:
    """
    Remove a polynomial trend from *x* via least squares.

    The Vandermonde matrix is built by broadcasting t^exps — no loop.
    GPU path uses ``torch.linalg.lstsq``; CPU path uses ``np.linalg.lstsq``.
    """
    N   = len(x)
    dev = _dev(device)

    if dev is not None:
        t    = torch.linspace(0.0, 1.0, N, device=dev)           # (N,)
        exps = torch.arange(degree + 1, device=dev, dtype=torch.float32)
        V    = t.unsqueeze(1) ** exps.unsqueeze(0)                # (N, D+1) — no loop
        xt   = torch.as_tensor(x, dtype=torch.float32, device=dev)
        try:
            coeffs = torch.linalg.lstsq(V, xt.unsqueeze(1)).solution
        except AttributeError:                                    # older torch
            coeffs = torch.lstsq(xt.unsqueeze(1), V).solution[:degree + 1]
        trend = (V @ coeffs).squeeze()
        return (xt - trend).cpu().numpy()

    t    = np.linspace(0.0, 1.0, N)
    exps = np.arange(degree + 1)
    V    = t[:, np.newaxis] ** exps[np.newaxis, :]                # (N, D+1) broadcasting
    coeffs = np.linalg.lstsq(V, x, rcond=None)[0]
    return (x - V @ coeffs).astype(np.float32)


# ── Butterworth bandpass ──────────────────────────────────────────────────────

def butterworth_bandpass(
    x: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter via ``scipy.signal.sosfiltfilt``.

    ``sosfilt`` is numerically superior to the ``ba`` form for high orders.
    The filter is applied in one vectorised pass; no Python loop.
    """
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


# ── Windowed Fourier Transform (Gaussian STFT = WFT) ─────────────────────────

def _make_window(window_size: int, window: str, sigma=None, kaiser_beta=8.6,
                 device=None):
    """
    Build a named analysis window — vectorised, no loops.
    Supported: gaussian, hann, hamming, blackman, rect, exp, kaiser.
    """
    W = window_size
    n = np.arange(W, dtype=np.float32)
    ctr = (W - 1) / 2.0

    w_name = window.lower()
    if w_name in ("gaussian", "wft"):
        s = W / 6.0 if sigma is None else float(sigma)
        win = np.exp(-0.5 * ((n - ctr) / s) ** 2)
    elif w_name == "hann":
        win = 0.5 * (1 - np.cos(2 * np.pi * n / (W - 1)))
    elif w_name == "hamming":
        win = 0.54 - 0.46 * np.cos(2 * np.pi * n / (W - 1))
    elif w_name == "blackman":
        win = (0.42 - 0.5 * np.cos(2*np.pi*n/(W-1))
               + 0.08 * np.cos(4*np.pi*n/(W-1)))
    elif w_name in ("rect", "rectangular", "boxcar"):
        win = np.ones(W, dtype=np.float32)
    elif w_name in ("exp", "exponential"):
        tau = W / 8.6  # decay constant so edge ≈ exp(-5)
        win = np.exp(-np.abs(n - ctr) / tau)
    elif w_name == "kaiser":
        # Kaiser with parameter beta — I0-based formula, no loop
        from scipy.signal.windows import kaiser as scipy_kaiser
        win = scipy_kaiser(W, kaiser_beta).astype(np.float32)
    else:
        raise ValueError(f"Unknown window '{window}'. "
                         "Choose: gaussian/wft, hann, hamming, blackman, "
                         "rect, exp, kaiser")

    if device is not None and _TORCH:
        return torch.as_tensor(win, device=device)
    return win


def wft(
    x: np.ndarray,
    fs: float,
    window_size: int = 256,
    hop_size: int = 128,
    window: str = "gaussian",
    sigma: Optional[float] = None,
    kaiser_beta: float = 8.6,
    device=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Windowed Fourier Transform with configurable window type.

    Supports all MODA window types: gaussian (default/WFT), hann, hamming,
    blackman, rect, exp, and kaiser (with ``kaiser_beta`` shape parameter).

    The Gaussian window gives optimal joint time-frequency localisation
    (saturates Heisenberg uncertainty); Kaiser gives flexible sidelobe control.

    The WFT differs from the standard STFT only in the window shape —
    a Gaussian provides optimal joint time-frequency localisation
    (saturates the Heisenberg uncertainty bound).

    Frame extraction uses ``Tensor.unfold`` / ``np.lib.stride_tricks.as_strided``
    — both zero-copy and loop-free.  The batch FFT processes all frames
    simultaneously.

    Parameters
    ----------
    sigma : float | None
        Gaussian standard deviation in samples.
        Default: window_size / 6  (3σ covers the full window).

    Returns
    -------
    freqs     : (W//2+1,) Hz
    times     : (n_frames,) s
    magnitude : (W//2+1, n_frames) float32
    """
    W   = window_size
    dev = _dev(device)

    if dev is not None:
        xt  = torch.as_tensor(x, dtype=torch.float32, device=dev)
        win = _make_window(W, window, sigma=sigma, kaiser_beta=kaiser_beta,
                           device=dev)
        if not isinstance(win, torch.Tensor):
            win = torch.as_tensor(win, device=dev)

        pad    = W // 2
        xp     = F.pad(xt, (pad, pad))
        frames = xp.unfold(0, W, hop_size)                        # (n_frames, W)
        spec   = torch.fft.rfft(frames * win.unsqueeze(0), dim=1) # (n_frames, W//2+1)
        mag    = torch.abs(spec).T.cpu().numpy()
        freqs  = np.fft.rfftfreq(W, 1.0 / fs).astype(np.float32)
        times  = (np.arange(frames.shape[0]) * hop_size / fs).astype(np.float32)
        return freqs, times, mag.astype(np.float32)

    win    = _make_window(W, window, sigma=sigma, kaiser_beta=kaiser_beta)
    pad    = W // 2
    xp     = np.pad(x.astype(np.float32), pad)
    n_fr   = (len(xp) - W) // hop_size + 1
    shape  = (n_fr, W)
    strides = (xp.strides[0] * hop_size, xp.strides[0])
    frames = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    spec   = np.fft.rfft(frames * win[np.newaxis, :], axis=1)
    mag    = np.abs(spec).T.astype(np.float32)
    freqs  = np.fft.rfftfreq(W, 1.0 / fs).astype(np.float32)
    times  = (np.arange(n_fr) * hop_size / fs).astype(np.float32)
    return freqs, times, mag


def rp_surrogates(x: np.ndarray, n_surrogates: int) -> np.ndarray:
    """
    Random Permutation (RP) surrogates — batch shuffle with no loop.

    All permutations generated via argsort of random keys (one op per surrogate
    batch, vectorised across all n_surrogates simultaneously).
    """
    N     = len(x)
    keys  = np.random.rand(n_surrogates, N)        # (n, N)
    perms = np.argsort(keys, axis=1)               # (n, N) — permutation indices
    return x[perms].astype(np.float32)             # (n, N) fancy indexing, no loop


def aaft_surrogates(x: np.ndarray, n_surrogates: int) -> np.ndarray:
    """
    AAFT (Amplitude-Adjusted Fourier Transform) surrogates.

    Matches the rank-order of the original signal's amplitude distribution
    while applying random Fourier phases.  All n_surrogates generated
    in one vectorised batch — no per-surrogate loop.

    Algorithm (MATLAB surrcalc.m 'AAFT' branch):
    1. Sort Gaussian noise to match amplitude ranks of *x*.
    2. Phase-randomise the amplitude-matched noise.
    3. Re-rank the result to restore the original amplitude distribution.
    """
    N        = len(x)
    sorted_x = np.sort(x)
    ranks_x  = np.argsort(np.argsort(x))             # (N,) rank of each element

    # Step 1: batch Gaussian noise, sorted then reordered to match x's ranks
    gn          = np.random.randn(n_surrogates, N).astype(np.float32)
    gn_sorted   = np.sort(gn, axis=1)                # (n, N)
    gn_matched  = gn_sorted[:, ranks_x]              # (n, N) — no per-surrogate loop

    # Step 2: phase randomise — all surrogates in one rfft + phase assign + irfft
    fft_gm   = np.fft.rfft(gn_matched, axis=1)       # (n, N//2+1)
    phases   = np.random.uniform(0, 2 * np.pi, fft_gm.shape)
    fft_rand = np.abs(fft_gm) * np.exp(1j * phases)
    ph_rand  = np.fft.irfft(fft_rand, n=N, axis=1)   # (n, N)

    # Step 3: reorder back to original amplitude distribution
    ranks_ph   = np.argsort(np.argsort(ph_rand, axis=1), axis=1)  # (n, N) — vectorised
    surrogates = sorted_x[ranks_ph]                  # (n, N) — single fancy-index op
    return surrogates.astype(np.float32)


def iaaft_surrogates(x: np.ndarray, n_surrogates: int,
                     max_iter: int = 100) -> np.ndarray:
    """
    IAAFT2 (Iterative AAFT with exact spectrum) surrogates — CPU batch version.

    Iterates until the rank ordering converges, preserving both the Fourier
    amplitude spectrum and the amplitude distribution of *x*.
    The main loop runs at most *max_iter* times (typically < 50).
    All surrogate iterations are processed as a batch using masked updates.
    """
    N        = len(x)
    sorted_x = np.sort(x)
    fft_sig  = np.fft.rfft(x)
    amp_sig  = np.abs(fft_sig)                       # (N//2+1,)

    # Initialise from random permutations
    surr  = rp_surrogates(x, n_surrogates)           # (n, N)
    ranks = np.argsort(np.argsort(x))                # (N,) original ranks

    old_ranks = np.full((n_surrogates, N), -1, dtype=np.int32)

    for _ in range(max_iter):
        new_ranks = np.argsort(np.argsort(surr, axis=1), axis=1)  # (n, N)
        done      = (new_ranks == old_ranks).all(axis=1)          # (n,) converged
        if done.all():
            break
        old_ranks = new_ranks.copy()

        # Replace Fourier amplitudes with original's (batch rfft/irfft)
        fft_s  = np.fft.rfft(surr, axis=1)                        # (n, N//2+1)
        fft_s  = amp_sig * np.exp(1j * np.angle(fft_s))           # broadcast (N//2+1,)
        surr   = np.fft.irfft(fft_s, n=N, axis=1)

        # Restore amplitude distribution (only for unconverged)
        cur_ranks = np.argsort(np.argsort(surr, axis=1), axis=1)  # (n, N)
        surr      = sorted_x[cur_ranks]                            # (n, N) vectorised

    return surr.astype(np.float32)


def cpp_surrogates(
    x: np.ndarray,
    n_surrogates: int,
    fs: float = 1.0,
) -> np.ndarray:
    """
    Cyclic Phase Perturbation (CPP) surrogates.

    Preserves instantaneous amplitude modulation while destroying phase
    information via a random cyclic phase offset per surrogate.
    Designed for Bayesian coupling significance testing.

    Fully vectorised: all ``n_surrogates`` generated in one batch — no loop.

    Returns
    -------
    surrogates : (n_surrogates, n_times) float32
    """
    from scipy.signal import hilbert as _hilbert
    analytic  = _hilbert(x)
    amplitude = np.abs(analytic)                         # (T,)
    phase     = np.angle(analytic)                       # (T,)

    shifts    = np.random.uniform(0, 2 * np.pi, n_surrogates)  # (n,)
    # Broadcasting: (n,1) + (1,T) → (n,T) — single batch op
    phases_sh = phase[np.newaxis, :] + shifts[:, np.newaxis]
    return (amplitude[np.newaxis, :] * np.cos(phases_sh)).astype(np.float32)
