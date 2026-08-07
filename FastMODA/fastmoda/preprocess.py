"""Signal preprocessing: crop (clip / first / final) and integer decimation.

Shared by the ``/preprocess`` endpoints and mirrored by MODA's Preprocessing
module. The crop/resample spec is deliberately small and explicit so the desktop
and web apps stay at parity:

    spec = {
        'mode':    'range' | 'first' | 'final' | 'none',
        'start_s':  float,   # 'range'
        'stop_s':   float,   # 'range'
        'length_s': float,   # 'first' / 'final'
        'decimate_factor': int >= 1,   # 1 = no change
    }

Decimation is **integer-only** (fs -> fs/k) and anti-aliased (scipy.decimate),
never naive subsampling; the new fs travels with the returned signal.
"""

from __future__ import annotations

import numpy as np


def _factorize(q: int):
    """Split an integer decimation factor into small stages (<=13 each) so the
    anti-alias filter stays well conditioned (scipy recommends q<=13 per call)."""
    q = int(q)
    factors = []
    for p in (2, 3, 5, 7, 11, 13):
        while q % p == 0:
            factors.append(p)
            q //= p
    if q > 1:
        factors.append(q)          # leftover prime > 13 (decimate still handles it)
    return factors or [1]


def decimate_integer(x: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased integer decimation by ``factor`` (staged for large factors)."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(x, dtype=float)
    from scipy.signal import decimate
    y = np.asarray(x, dtype=float)
    for f in _factorize(factor):
        if f == 1 or len(y) <= 27:     # too short to filter further
            break
        y = decimate(y, f, ftype="iir", zero_phase=True)
    return y


def crop_indices(n: int, fs: float, mode: str,
                 start_s=None, stop_s=None, length_s=None):
    """Return (i0, i1) sample bounds (half-open) for a crop spec, clamped to
    ``[0, n]``. ``mode='none'`` keeps everything."""
    mode = (mode or "none").lower()
    if mode == "range":
        i0 = 0 if start_s is None else int(round(float(start_s) * fs))
        i1 = n if stop_s is None else int(round(float(stop_s) * fs))
    elif mode == "first":
        i0, i1 = 0, int(round(float(length_s) * fs))
    elif mode == "final":
        i0, i1 = n - int(round(float(length_s) * fs)), n
    else:
        i0, i1 = 0, n
    i0 = max(0, min(i0, n))
    i1 = max(0, min(i1, n))
    if i1 <= i0:
        raise ValueError("Crop produces an empty signal — check the start/stop "
                         "or length values against the signal duration.")
    return i0, i1


def crop_and_decimate(x, fs, mode="none", start_s=None, stop_s=None,
                      length_s=None, decimate_factor=1):
    """Apply a crop then an integer decimation.

    Returns (y, fs_new, info) where ``info`` reports the before/after sizes and
    the effective time window, for UI display.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    fs = float(fs)

    i0, i1 = crop_indices(n, fs, mode, start_s, stop_s, length_s)
    y = x[i0:i1]
    k = max(1, int(decimate_factor))
    y = decimate_integer(y, k)
    fs_new = fs / k

    info = {
        "n_in": n,
        "n_cropped": int(i1 - i0),
        "n_out": int(len(y)),
        "fs_in": fs,
        "fs_out": fs_new,
        "decimate_factor": k,
        "t_start": i0 / fs,
        "t_stop": i1 / fs,
        "dur_in": n / fs,
        "dur_out": len(y) / fs_new if fs_new else 0.0,
    }
    return y, fs_new, info


def integer_rate_options(fs: float, min_rate: float = 0.5, max_factor: int = 32):
    """List valid integer-decimation targets (fs/k) down to ~min_rate, for a UI
    dropdown. Returns list of (factor, target_fs)."""
    fs = float(fs)
    out = []
    for k in range(1, max_factor + 1):
        r = fs / k
        if r < min_rate and k > 1:
            break
        out.append((k, r))
    return out
