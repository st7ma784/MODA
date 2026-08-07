"""Legacy (MODA-faithful) transforms.

FastMODA's default transforms (``analysis_gpu.cwt_gpu``, ``filtering.wft``) are
*re-implementations* tuned for speed, and they differ from Dmytro Iatsenko's
MATLAB originals (``allguis/guis/tfa/Functions/wt.m`` / ``wft.m``) in ways that
are documented in ``docs/validation/algorithmic-differences.md``. When you need
results that line up with the MATLAB desktop app as closely as is practical in
Python, use the ``*_legacy`` functions here.

``wt_legacy`` is a faithful, fully-vectorised port of ``wt.m``'s frequency-domain
algorithm:

* the exact Lognorm / Morlet / Bump wavelet **frequency-domain forms**;
* MODA's **log-voice frequency lattice** ``2^(k/nv)`` with ``nv`` derived from the
  wavelet's 50%-energy support (``'auto'`` = MODA's ``auto-10``);
* the ``p = 1`` amplitude normalisation and frequency-domain convolution
  ``WT = ifft(fx · conj(FW))`` — i.e. **complex** coefficients, not magnitude;
* optional **preprocessing** (cubic-polynomial detrend + band-pass to
  ``[fmin, fmax]``), on by default as in MODA;
* next-power-of-two padding with **zero / symmetric / periodic / predictive**
  modes and a cone-of-influence NaN mask (``cut_edges``).

What is *not* reproduced bit-for-bit: MODA's adaptive ``sqeps``/``quadgk``
support integration (we use the same cumulative-energy method on a fine grid,
which agrees to a few ×1e-3 on nv and COI) and ``fcast`` predictive padding
(approximated; irrelevant when ``cut_edges=True`` since the affected samples are
discarded). See the docs page for the quantified consequences.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


# ── wavelet frequency-domain forms (verbatim from wt.m lines 295-321) ─────────

def moda_wavelet(name: str = "Lognorm", f0: float = 1.0):
    """Return (fwt, ompeak, xi1, xi2) for a built-in MODA wavelet.

    ``fwt(xi)`` is the wavelet's Fourier transform (cyclic frequency ``xi``);
    ``ompeak`` its peak frequency; ``[xi1, xi2]`` its frequency support.
    """
    name = name.lower()
    if name in ("lognorm", "lognormal"):
        q = TWO_PI * f0
        fwt = lambda xi: np.exp(-(q ** 2 / 2.0) * np.log(np.maximum(xi, 1e-300)) ** 2)
        return fwt, 1.0, 0.0, np.inf
    if name == "morlet":
        om0 = TWO_PI * f0
        # includes the admissibility correction term MODA keeps (the second exp)
        fwt = lambda xi: (np.exp(-0.5 * (om0 - xi) ** 2)
                          - np.exp(-0.5 * (om0 ** 2 + xi ** 2)))
        return fwt, om0, 0.0, np.inf
    if name == "bump":
        q = 2.5 * f0
        if q < 1:
            raise ValueError("For Bump wavelet f0 cannot be lower than 0.4")
        def fwt(xi):
            xi = np.asarray(xi, float)
            inside = (xi > 1 - 1 / q) & (xi < 1 + 1 / q)
            out = np.zeros_like(xi)
            u = 1.0 - (q ** 2) * (1.0 - xi[inside]) ** 2
            out[inside] = np.exp(1.0 - np.abs(1.0 / u))
            return out
        return fwt, 1.0, max(0.0, 1 - 1 / q), 1 + 1 / q
    raise ValueError(f"Unknown wavelet '{name}'. Choose Lognorm, Morlet, or Bump.")


# ── MODA-method wavelet parameters (nv, ε- and 50%-supports, COI) ─────────────

def _wavelet_params(fwt, ompeak, xi1, xi2, racc=0.01, ngrid=1 << 16):
    """Estimate the frequency/time supports exactly the way wt.m's parcalc /
    sqeps do (see wt.m lines 862-864, 1375).

    * 50%-frequency support: cumulative of ``fwt`` over log-frequency, taken at
      the 25% and 75% points (sqeps ``s1h=fz(0.25)``, ``s2h=fz(0.75)``). Drives
      the auto ``nv``.
    * ε-time support: the *demodulated* time-domain wavelet envelope's
      cumulative, at ``racc/2`` and ``1-racc/2`` (sqeps ε-support). Drives the
      cone of influence and the default ``fmin``.
    """
    # --- 50%-support in log-frequency: cumulate fwt(exp(u)) over u = log(xi) ---
    hi = ompeak * 1e3 if not np.isfinite(xi2) else min(xi2, ompeak * (1 + 1e3))
    lo = xi1 if xi1 > 0 else ompeak * 1e-3
    u = np.linspace(np.log(lo), np.log(hi), ngrid)
    fu = np.real(fwt(np.exp(u)))
    fu[~np.isfinite(fu)] = 0.0
    c = np.cumsum(fu)
    c = (c - c[0]) / (c[-1] - c[0])
    xi1h = np.exp(np.interp(0.25, c, u))
    xi2h = np.exp(np.interp(0.75, c, u))

    # --- ε-time support: invert fwt → twf(t), demodulate, cumulate (as MODA) ---
    Nf = 1 << 15
    xg_max = ompeak * 12.0 if not np.isfinite(xi2) else min(xi2, ompeak * 12.0)
    xg = np.linspace(0.0, xg_max, Nf)
    F = fwt(xg)
    F[~np.isfinite(F)] = 0.0
    twf = np.fft.fftshift(np.fft.ifft(F))
    dt = TWO_PI / (xg[1] - xg[0]) / Nf
    t = (np.arange(Nf) - Nf // 2) * dt
    env = twf * np.exp(-1j * ompeak * t)          # demodulate (wt.m line 932)
    cs = np.cumsum(env)
    cs = np.abs(cs) / np.abs(cs[-1])              # wt.m lines 954-955
    t1e = np.interp(racc / 2.0, cs, t)
    t2e = np.interp(1.0 - racc / 2.0, cs, t)
    return dict(xi1h=xi1h, xi2h=xi2h, t1e=t1e, t2e=t2e)


def _fcast_predictive(sig, fs, n, fmin, fmax, side):
    """Approximate MODA's fcast predictive padding: extrapolate the signal with
    a small set of decaying sinusoids fitted in the [fmin, fmax] band.

    This is a best-effort stand-in for wt.m's ``fcast.m``; when ``cut_edges`` is
    True the padded region is discarded anyway, so exactness here does not affect
    reported coefficients.
    """
    L = len(sig)
    if n <= 0:
        return np.zeros(0)
    # detrended, band-limited copy for a stable AR-ish harmonic fit
    x = sig - np.mean(sig)
    # dominant in-band frequencies from the periodogram
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(L, 1.0 / fs)
    band = (fr >= fmin) & (fr <= fmax)
    if not band.any():
        return np.full(n, sig[-1] if side == "right" else sig[0])
    k = np.argsort(np.abs(X) * band)[-min(5, band.sum()):]
    t_future = (np.arange(1, n + 1)) / fs
    pad = np.zeros(n)
    for ki in k:
        amp = 2 * np.abs(X[ki]) / L
        ph = np.angle(X[ki])
        w = TWO_PI * fr[ki]
        if side == "right":
            pad += amp * np.cos(w * (t_future + L / fs) + ph)
        else:
            pad += amp * np.cos(-w * t_future + ph)
    pad += np.mean(sig)
    return pad


def _detrend_poly(sig, fs, order=3):
    """Subtract an order-3 polynomial fit (standardised columns), as in wt.m."""
    n = len(sig)
    X = (np.arange(1, n + 1) / fs).reshape(-1, 1)
    XM = [np.ones(n)]
    for p in range(1, order + 1):
        c = (X[:, 0]) ** p
        XM.append((c - c.mean()) / (c.std() + 1e-300))
    XM = np.vstack(XM).T
    coef, *_ = np.linalg.lstsq(XM, sig, rcond=None)
    return sig - XM @ coef


# ── WFT window frequency-domain forms (verbatim from wft.m lines 294-326) ─────

def moda_window(name: str = "Gaussian", f0: float = 1.0):
    """Return (fwt, twf, xi1, xi2) for a built-in MODA WFT window.

    All windows are centred at zero frequency (``ompeak = 0``); ``fwt`` is the
    window's Fourier transform and ``twf`` its time-domain form.
    """
    name = name.lower()
    if name in ("gaussian", "wft"):
        fwt = lambda xi: np.exp(-(f0 ** 2 / 2.0) * xi ** 2)
        twf = lambda t: (1.0 / np.sqrt(2 * np.pi) / f0) * np.exp(-t ** 2 / (2 * f0 ** 2))
        return fwt, twf, -np.inf, np.inf
    if name == "hann":
        q = 4.4 * f0
        twf = lambda t: (1 + np.cos(2 * np.pi * t / q)) / 2
        fwt = lambda xi: (-(2 * np.pi / q) ** 2) * np.sin(xi * q / 2) / (
            xi * (xi ** 2 - (2 * np.pi / q) ** 2))
        return fwt, twf, -np.inf, np.inf
    if name == "blackman":
        q, alpha = 5.6 * f0, 0.16
        twf = lambda t: (1 + np.cos(2*np.pi*t/q))/2 - alpha*(1 + np.cos(4*np.pi*t/q))/2
        fwt = lambda xi: ((-(2*np.pi/q)**2) * np.sin(xi*q/2) / xi) * (
            1.0/(xi**2 - (2*np.pi/q)**2) - 4*alpha/(xi**2 - (4*np.pi/q)**2))
        return fwt, twf, -np.inf, np.inf
    if name in ("exp", "exponential"):
        q = 6.5 * f0
        twf = lambda t: np.exp(-np.abs(t) / q)
        fwt = lambda xi: 2 * q / (1 + (q ** 2) * xi ** 2)
        return fwt, twf, -np.inf, np.inf
    if name in ("rect", "rectangular", "boxcar"):
        q = 10 * f0
        twf = lambda t: np.ones_like(np.asarray(t, float))
        fwt = lambda xi: 2 * np.sin(q * xi / 2) / xi
        return fwt, twf, -np.inf, np.inf
    if "kaiser" in name:
        from scipy.special import i0
        a = 3.0 if len(name) <= 6 else float(name.split("-")[1])
        q = 3 * np.sqrt(1 + abs(a - 1 / a)) * f0
        B = i0(np.pi * a)
        def twf(t):
            t = np.asarray(t, float)
            inside = np.abs(2 * t / q) < 1
            out = np.zeros_like(t)
            out[inside] = i0(np.pi * a * np.sqrt(1 - (2 * t[inside] / q) ** 2)) / B
            return out
        return None, twf, -q / 2, q / 2   # fwt derived numerically from twf
    raise ValueError(f"Unknown window '{name}'.")


def _window_params(fwt, twf, racc=0.01, ngrid=1 << 16, tspan=None):
    """WFT analogue of ``_wavelet_params``: 50%-frequency support (for ``fstep``)
    and ε-time support (for the cone of influence). The window is centred at 0,
    so the frequency support is taken on a *linear* symmetric axis.
    """
    # ε-time support from the (given) time-domain window
    span = tspan if tspan is not None else 50.0
    t = np.linspace(-span, span, ngrid)
    w = np.real(twf(t))
    w[~np.isfinite(w)] = 0.0
    cs = np.cumsum(w)
    cs = (cs - cs[0]) / (cs[-1] - cs[0])
    t1e = np.interp(racc / 2.0, cs, t)
    t2e = np.interp(1.0 - racc / 2.0, cs, t)

    # 50%-frequency support: cumulate fwt over a linear ξ axis, 25%/75% points
    if fwt is None:
        # derive fwt numerically from twf (Kaiser): FT of the time window
        F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(w))) * (t[1] - t[0])
        xi = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ngrid, t[1] - t[0]))
        fv = np.abs(F)
    else:
        xi = np.linspace(-200.0, 200.0, ngrid)
        with np.errstate(divide="ignore", invalid="ignore"):
            fv = np.real(fwt(xi))
            bad = ~np.isfinite(fv)
            if bad.any():
                fv[bad] = np.real(fwt(xi[bad] + 1e-12))
        fv[~np.isfinite(fv)] = 0.0
    c = np.cumsum(fv)
    c = (c - c[0]) / (c[-1] - c[0])
    xi1h = np.interp(0.25, c, xi)
    xi2h = np.interp(0.75, c, xi)
    return dict(xi1h=xi1h, xi2h=xi2h, t1e=t1e, t2e=t2e)


def wft_legacy(signal, fs, fmin=None, fmax=None, window="Gaussian", f0=1.0,
               fstep="auto", padding="predictive", preprocess=True,
               cut_edges=False, return_freq=True):
    """MODA-faithful windowed Fourier transform (port of ``wft.m``).

    Unlike the CWT the window is **shifted** (not dilated) to each frequency, the
    frequency grid is **linear** (``fstep``), the cone of influence is constant
    across frequency, and the filter is applied **without** conjugation (see
    wft.m line 523). Returns ``(WFT, freq)`` with complex ``WFT`` of shape
    ``(n_freq, len(signal))``.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    L = len(sig)
    if fmax is None:
        fmax = fs / 2.0
    fwt, twf, xi1, xi2 = moda_window(window, f0)
    # time span for support estimation: a few window widths
    tspan = 40.0 * f0 if not np.isfinite(xi2) else 1.5 * (xi2 - xi1)
    wp = _window_params(fwt, twf, tspan=tspan)

    if isinstance(fstep, str) and "auto" in fstep:
        Nb = 10 if len(fstep) <= 4 else float(fstep.split("-")[1])
        fs_raw = (wp["xi2h"] - wp["xi1h"]) / (2 * np.pi * Nb)
        c10 = np.floor(np.log10(fs_raw))
        fstep = np.floor(fs_raw / 10 ** c10) * 10 ** c10      # 1 significant figure
    fstep = float(fstep)
    if fmin is None:
        fmin = fstep

    freq = np.arange(np.ceil(fmin / fstep), np.floor(fmax / fstep) + 1) * fstep
    SN = len(freq)
    coib1 = int(np.ceil(abs(wp["t1e"] * fs)))
    coib2 = int(np.ceil(abs(wp["t2e"] * fs)))

    if preprocess:
        sig = _detrend_poly(sig, fs)
        fx0 = np.fft.fft(sig)
        ff0 = np.fft.fftfreq(L, 1.0 / fs)
        sig = np.real(np.fft.ifft(np.where(
            (np.abs(ff0) <= max(fmin, fs / L)) | (np.abs(ff0) >= fmax), 0.0, fx0)))

    NL = 1 << int(np.ceil(np.log2(L + coib1 + coib2)))
    if coib1 == 0 and coib2 == 0:
        n1 = (NL - L) // 2
    else:
        n1 = int(np.floor((NL - L) * coib1 / (coib1 + coib2)))
    n2 = NL - L - n1

    if padding == "predictive":
        padleft = _fcast_predictive(sig, fs, n1, max(fmin, fs / L), fmax, "left")
        padright = _fcast_predictive(sig, fs, n2, max(fmin, fs / L), fmax, "right")
    elif padding in ("zero", "zeros", 0):
        padleft, padright = np.zeros(n1), np.zeros(n2)
    elif padding == "symmetric":
        padleft = sig[:n1][::-1] if n1 <= L else np.r_[np.zeros(n1 - L), sig[::-1]]
        padright = sig[-n2:][::-1] if n2 <= L else np.r_[sig[::-1], np.zeros(n2 - L)]
    else:
        padleft, padright = np.zeros(n1), np.zeros(n2)
    sigp = np.concatenate([padleft, sig, padright])

    ff = np.fft.fftfreq(NL, 1.0 / fs)
    fx = np.fft.fft(sigp)
    if preprocess:
        fx = np.where((ff <= max(fmin, fs / L)) | (ff >= fmax), 0.0, fx)

    # numeric fwt for Kaiser (from twf), else the closed form
    if fwt is None:
        def fwt(xi):
            tt = np.linspace(-tspan, tspan, 1 << 14)
            W = twf(tt)
            F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(W))) * (tt[1] - tt[0])
            xg = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(tt), tt[1] - tt[0]))
            return np.interp(xi, xg, np.real(F), left=0.0, right=0.0)

    # shifted (not dilated) frequency axis; NO conjugation (wft.m line 518-523)
    freqwf = freq[None, :] - ff[:, None]                       # NL x SN
    in_supp = (freqwf > xi1 / TWO_PI) & (freqwf < xi2 / TWO_PI)
    FW = np.zeros((NL, SN), dtype=np.float64)
    arg = TWO_PI * freqwf[in_supp]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.real(fwt(arg))
        bad = ~np.isfinite(vals)          # removable singularities (sinc-type)
        if bad.any():
            vals[bad] = np.real(fwt(arg[bad] + 1e-14))
    vals[~np.isfinite(vals)] = 0.0
    FW[in_supp] = vals
    CC = fx[:, None] * FW
    WTfull = np.fft.ifft(CC, axis=0)
    WFT = WTfull[n1:NL - n2, :].T.astype(np.complex128)

    if cut_edges and coib1 + coib2 < L:
        if coib1 > 0:
            WFT[:, :coib1] = np.nan
        if coib2 > 0:
            WFT[:, L - coib2:] = np.nan

    return (WFT, freq) if return_freq else WFT


def wt_legacy(signal, fs, fmin=None, fmax=None, wavelet="Lognorm", f0=1.0,
              nv="auto", padding="predictive", preprocess=True,
              cut_edges=True, return_freq=True):
    """MODA-faithful continuous wavelet transform (port of ``wt.m``).

    Parameters mirror ``wt.m``. Returns ``(WT, freq)`` where ``WT`` is a complex
    ``(n_freq, len(signal))`` array (rows = frequencies, cols = time) and
    ``freq`` the log-voice frequency lattice. With ``cut_edges=True`` (MODA
    default), coefficients outside the cone of influence are ``NaN``.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    L = len(sig)
    if fmax is None:
        fmax = fs / 2.0
    fwt, ompeak, xi1, xi2 = moda_wavelet(wavelet, f0)
    wp = _wavelet_params(fwt, ompeak, xi1, xi2)

    # number of voices (MODA 'auto' == 'auto-10')
    if isinstance(nv, str) and "auto" in nv:
        Nb = 10 if len(nv) <= 4 else float(nv.split("-")[1])
        nv_real = Nb * np.log(2) / np.log(wp["xi2h"] / wp["xi1h"])
        nv = int(np.ceil(nv_real))
    nv = int(nv)

    if fmin is None:
        fmin = (ompeak / TWO_PI) * (wp["t2e"] - wp["t1e"]) * fs / L
    if fmin > fmax:
        raise ValueError(f"fmin {fmin:.3g} exceeds fmax {fmax:.3g}")

    # MODA log-voice lattice: freq = 2^(k/nv), k integer
    k0 = int(np.ceil(nv * np.log2(fmin)))
    k1 = int(np.floor(nv * np.log2(fmax)))
    freq = 2.0 ** (np.arange(k0, k1 + 1) / nv)
    SN = len(freq)

    coib1 = np.ceil(np.abs(wp["t1e"] * fs * (ompeak / (TWO_PI * freq)))).astype(int)
    coib2 = np.ceil(np.abs(wp["t2e"] * fs * (ompeak / (TWO_PI * freq)))).astype(int)

    # preprocessing before padding (detrend + band-pass), as in wt.m default
    if preprocess:
        sig = _detrend_poly(sig, fs)
        fx0 = np.fft.fft(sig)
        ff0 = np.fft.fftfreq(L, 1.0 / fs)
        sig = np.real(np.fft.ifft(np.where(
            (np.abs(ff0) <= max(fmin, fs / L)) | (np.abs(ff0) >= fmax), 0.0, fx0)))

    # padding to next power of two, split by COI ratio
    NL = 1 << int(np.ceil(np.log2(L + coib1[0] + coib2[0])))
    if coib1[0] == 0 and coib2[0] == 0:
        n1 = (NL - L) // 2
        n2 = NL - L - n1
    else:
        n1 = int(np.floor((NL - L) * coib1[0] / (coib1[0] + coib2[0])))
        n2 = NL - L - n1

    if padding == "predictive":
        padleft = _fcast_predictive(sig, fs, n1, max(fmin, fs / L), fmax, "left")
        padright = _fcast_predictive(sig, fs, n2, max(fmin, fs / L), fmax, "right")
    elif padding in ("zero", "zeros", 0):
        padleft, padright = np.zeros(n1), np.zeros(n2)
    elif padding == "symmetric":
        padleft = sig[:n1][::-1] if n1 <= L else np.r_[np.zeros(n1 - L), sig[::-1]]
        padright = sig[-n2:][::-1] if n2 <= L else np.r_[sig[::-1], np.zeros(n2 - L)]
    elif padding == "periodic":
        padleft = sig[L - n1:] if n1 <= L else np.r_[np.zeros(n1 - L), sig]
        padright = sig[:n2] if n2 <= L else np.r_[sig, np.zeros(n2 - L)]
    else:
        raise ValueError(f"Unknown padding '{padding}'")
    sigp = np.concatenate([padleft, sig, padright])

    # fft of padded signal, band-pass again if preprocessing
    ff = np.fft.fftfreq(NL, 1.0 / fs)
    fx = np.fft.fft(sigp)
    if preprocess:
        fx = np.where((ff <= max(fmin, fs / L)) | (ff >= fmax), 0.0, fx)

    # vectorised frequency-domain convolution (wt.m lines 543-615, p = 1)
    freqwf = ff[:, None] * (ompeak / (TWO_PI * freq[None, :]))   # NL x SN
    in_supp = (freqwf > xi1 / TWO_PI) & (freqwf < xi2 / TWO_PI)
    FW = np.zeros((NL, SN), dtype=np.float64)
    arg = TWO_PI * freqwf[in_supp]
    vals = np.conj(fwt(arg))
    vals[~np.isfinite(vals)] = 0.0
    FW[in_supp] = np.real(vals)
    CC = fx[:, None] * FW
    WTfull = np.fft.ifft(CC, axis=0)                            # (NL, SN)
    WT = WTfull[n1:NL - n2, :].T.astype(np.complex128)          # (SN, L)

    if cut_edges:
        for i in range(SN):
            c1, c2 = coib1[i], coib2[i]
            if c1 + c2 >= L:
                WT[i, :] = np.nan
            else:
                if c1 > 0:
                    WT[i, :c1] = np.nan
                if c2 > 0:
                    WT[i, L - c2:] = np.nan

    return (WT, freq) if return_freq else WT
