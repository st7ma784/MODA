"""Legacy (MODA-faithful) ridge extraction.

Ports of Dmytro Iatsenko's ``ecurve.m`` and ``rectfr.m``
(``allguis/guis/filtering/Functions/``), in the configuration MODA's ridge
extraction uses (``MODAridge_filter.m``)::

    tfsupp = ecurve(WT, freq, wopt);
    [iamp, iphi, ifreq] = rectfr(tfsupp, WT, freq, wopt, 'direct');

``ecurve_legacy`` is ``ecurve.m``'s default scheme: method II with parameters
``[1, 1]``, path optimisation on, ``AmpFunc = log``, no normalisation. The
amplitude peaks of the transform are located and refined by a three-point
parabola, the ridge is the path through them that maximises the amplitude
functional penalised for jumps in (log-)frequency and for distance from the
median (log-)frequency, found by dynamic programming and iterated until the
path stops changing. It returns the ridge with the lower and upper frequency
bounds of its time-frequency support.

``rectfr_legacy`` is ``rectfr.m``'s ``'direct'`` method: the component is
rebuilt by integrating the transform across that support and dividing by the
wavelet's reconstruction constant, so the amplitude is the component's own
amplitude ``A`` (not ``|WT| = A/2`` at one bin) and the frequency is a
support-weighted estimate between bins.

Indices are kept 1-based internally wherever ``ecurve.m`` rounds or compares
them, so every rounding lands where MATLAB's does.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _mround(x):
    """MATLAB ``round``: halves away from zero (numpy rounds halves to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _mstd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _nanmax_first(M, axis):
    """MATLAB ``max`` along an axis: NaNs ignored, first index on ties, and a
    NaN value with index 0 where every entry is NaN."""
    Mf = np.where(np.isnan(M), -np.inf, M)
    idx = np.argmax(Mf, axis=axis)
    val = np.take_along_axis(M, np.expand_dims(idx, axis), axis).squeeze(axis)
    allnan = np.all(np.isnan(M), axis=axis)
    return np.where(allnan, np.nan, val), np.where(allnan, 0, idx)


def _log_resolution(freq):
    """ecurve.m's test for a logarithmic frequency axis (line 247)."""
    return not (freq.min() <= 0 or _mstd(np.diff(freq)) < _mstd(np.diff(np.log(freq))))


def _interp1_extrap(x, y, xi):
    """``interp1(x, y, xi, 'linear', 'extrap')``."""
    x = np.asarray(x, float); y = np.asarray(y, float); xi = np.asarray(xi, float)
    if len(x) == 1:
        return np.full(xi.shape, y[0])
    out = np.interp(xi, x, y)
    lo, hi = xi < x[0], xi > x[-1]
    out[lo] = y[0] + (xi[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    out[hi] = y[-1] + (xi[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return out


def _median_range(pf):
    """``[median(dpf), range(dpf), median(pf), range(pf)]`` with ecurve.m's
    quartile-by-rounded-index definition of range (lines 544-547)."""
    dpf = np.diff(pf)
    mv = [np.median(dpf), 0.0, np.median(pf), 0.0]
    for slot, v in ((1, dpf), (3, pf)):
        ss = np.sort(v)
        cl = len(ss)
        hi = int(_mround(0.75 * cl)); lo = int(_mround(0.25 * cl))
        mv[slot] = ss[hi - 1] - ss[lo - 1]
    return mv


def _find_peaks(A, freq, log_res, fstep, tn1, tn2):
    """Amplitude peaks at each time, refined by a three-point parabola
    (ecurve.m lines 280-343). Returns ``Np`` (peaks per time) and the dense
    ``Mp x L`` matrices ``Ip`` (fractional 1-based bin), ``Fp`` (frequency) and
    ``Qp`` (amplitude), NaN where a time has fewer peaks."""
    NF, L = A.shape
    P = np.vstack([np.zeros((1, L)), A, np.zeros((1, L))])        # zero-padded
    with np.errstate(invalid="ignore"):
        pk = (P[1:-1] >= P[:-2]) & (P[1:-1] > P[2:])              # (NF, L)
    pk[0, :] = False                                              # border peaks
    pk[-1, :] = False
    Np = pk.sum(axis=0)
    Mp = int(max(Np.max() if L else 0, 2))
    tcol, frow = np.nonzero(pk.T)                                 # time-major, as MATLAB
    rank = (np.cumsum(pk, axis=0) - 1)[frow, tcol]                # row within its column

    a1, a2, a3 = P[frow, tcol], P[frow + 1, tcol], P[frow + 2, tcol]
    idf = frow + 1.0                                              # 1-based bin
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = 0.5 * (a1 - a3) / (a1 - 2 * a2 + a3)
        ip = idf + dp
        qp = a2 - 0.25 * (a1 - a3) * dp
        fp = freq[frow] * np.exp(dp * fstep) if log_res else freq[frow] + dp * fstep
    bad = np.isnan(dp) | (np.abs(dp) > 1)
    ip[bad], fp[bad], qp[bad] = idf[bad], freq[frow[bad]], a2[bad]

    Ip = np.full((Mp, L), np.nan); Fp = Ip.copy(); Qp = Ip.copy()
    Ip[rank, tcol], Fp[rank, tcol], Qp[rank, tcol] = ip, fp, qp

    # times with no peak inside the valid span take the border points instead
    for tn in tn1 + np.flatnonzero(Np[tn1:tn2 + 1] == 0):
        cg = A[[0, 1, NF - 2, NF - 1], tn]
        cn = 0
        if cg[0] > cg[1] or cg[3] > cg[2]:
            if cg[0] > cg[1]:
                Ip[cn, tn], Qp[cn, tn], Fp[cn, tn] = 1, cg[0], freq[0]; cn += 1
            if cg[3] > cg[2]:
                Ip[cn, tn], Qp[cn, tn], Fp[cn, tn] = NF, cg[3], freq[-1]; cn += 1
        else:
            Ip[0:2, tn] = [1, NF]; Qp[0:2, tn] = [cg[0], cg[3]]
            Fp[0:2, tn] = [freq[0], freq[-1]]; cn = 2
        Np[tn] = cn
    return Np, Ip, Fp, Qp


def _pathopt(Np, Fp, Wp, logw1, logw2, log_res):
    """ecurve.m's ``pathopt``: the peak sequence maximising the summed
    functional, by dynamic programming over time. Returns the 0-based row of
    the chosen peak at each time (-1 outside the span)."""
    Mp, L = Fp.shape
    has = np.flatnonzero(Np > 0)
    tn1, tn2 = has[0], has[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.log(Fp) if log_res else Fp
        W2 = Wp + logw2(F)

    U = np.full((Mp, L), np.nan)
    q = np.zeros((Mp, L), dtype=int)
    U[:Np[tn1], tn1] = W2[:Np[tn1], tn1]
    for tn in range(tn1 + 1, tn2 + 1):
        n, m = Np[tn], Np[tn - 1]
        cf = F[:n, tn][:, None] - F[:m, tn - 1][None, :]
        # same association as MATLAB: (W2 + CW1) + U
        M = (W2[:n, tn][:, None] + logw1(cf)) + U[:m, tn - 1][None, :]
        U[:n, tn], q[:n, tn] = _nanmax_first(M, axis=1)

    idid = np.full(L, -1, dtype=int)
    idid[tn2] = _nanmax_first(U[:, tn2], axis=0)[1]
    for tn in range(tn2 - 1, tn1 - 1, -1):
        idid[tn] = q[idid[tn + 1], tn + 1]
    return idid


def _support_bounds(A, pind1, tn1, tn2):
    """Lower and upper bins (1-based) of the time-frequency support around the
    ridge: the nearest amplitude minima either side (ecurve.m lines 665-672)."""
    NF, L = A.shape
    cols = np.arange(tn1, tn2 + 1)
    S = A[:, cols]
    # local minima, with the non-strict inequality on the side facing the peak
    mn_up = np.zeros(S.shape, dtype=bool); mn_dn = mn_up.copy()
    with np.errstate(invalid="ignore"):
        mn_up[1:-1] = (S[1:-1] <= S[:-2]) & (S[1:-1] < S[2:])
        mn_dn[1:-1] = (S[1:-1] <= S[2:]) & (S[1:-1] < S[:-2])
    rows1 = np.arange(1, NF + 1)[:, None]
    cp = pind1[cols][None, :]

    up = mn_up & (rows1 > cp)
    iup = np.where(up.any(axis=0), np.argmax(up, axis=0) + 1, NF)
    iup = np.where(cp[0] < NF - 1, iup, NF)
    dn = mn_dn & (rows1 < cp)
    idn = np.where(dn.any(axis=0), NF - np.argmax(dn[::-1], axis=0), 1)
    idn = np.where(cp[0] > 2, idn, 1)
    return idn, iup


def ecurve_legacy(TFR, freq, fs, pars=(1.0, 1.0), max_iter=20, return_info=False):
    """MODA-faithful ridge curve (port of ``ecurve.m``, default scheme).

    Parameters
    ----------
    TFR : (NF, L) array, complex or real
        Wavelet or windowed Fourier transform; NaN outside the cone of
        influence is allowed.
    freq : (NF,) array
        Its frequencies (logarithmic for a WT, linear for a WFT).
    fs : float
        Sampling frequency.
    pars : (2,) sequence
        ecurve's ``Param`` for method II: penalty weights for the frequency
        jump and for the distance from the median frequency. Default [1, 1].

    Returns
    -------
    tfsupp : (3, L) array
        Ridge frequency and the lower and upper frequency bounds of its
        time-frequency support, NaN outside the valid span.
    """
    A = np.abs(np.asarray(TFR))
    freq = np.asarray(freq, dtype=float).ravel()
    NF, L = A.shape
    tfsupp = np.full((3, L), np.nan)
    info = {"pind": np.full(L, np.nan), "pamp": np.full(L, np.nan), "iterations": 0}

    valid = np.flatnonzero(~np.isnan(A[-1, :]))
    if len(valid) == 0:
        return (tfsupp, info) if return_info else tfsupp
    tn1, tn2 = int(valid[0]), int(valid[-1])
    span = slice(tn1, tn2 + 1)
    tt = np.arange(tn1, tn2 + 1)

    log_res = _log_resolution(freq)
    fstep = float(np.mean(np.diff(np.log(freq)))) if log_res else float(np.mean(np.diff(freq)))

    Np, Ip, Fp, Qp = _find_peaks(A, freq, log_res, fstep, tn1, tn2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Wp = np.log(Qp)                                           # AmpFunc

    pind = np.full(L, np.nan); pamp = np.full(L, np.nan)

    def assign(rows):
        tfsupp[0, span] = Fp[rows, tt]
        pind[span] = _mround(Ip[rows, tt])
        pamp[span] = Qp[rows, tt]

    # global maximum of the functional at each time: the starting curve
    idr = np.zeros(len(tt), dtype=int)
    for i, tn in enumerate(tt):
        if Np[tn] > 0:
            idr[i] = _nanmax_first(Wp[:Np[tn], tn], axis=0)[1]
    assign(idr)
    idz = np.flatnonzero((pamp[span] == 0) | np.isnan(pamp[span]))
    if len(idz) and len(idz) < len(tt):
        keep = np.setdiff1d(np.arange(len(tt)), idz)
        pind[tt[idz]] = _mround(_interp1_extrap(tt[keep], pind[tt[keep]], tt[idz]))
        tfsupp[0, tt[idz]] = _interp1_extrap(tt[keep], tfsupp[0, tt[keep]], tt[idz])

    # ── method II: iterate the path optimisation to a fixed point ──────────
    def stats():
        pf = tfsupp[0, span]
        return _median_range(np.log(pf) if log_res else pf)

    mv = stats()
    history = [pind.copy()]
    itn, rdiff = 0, np.nan
    while rdiff != 0:
        s1 = mv[1] if mv[1] > 0 else 1e-32 / fs
        s2 = mv[3] if mv[3] > 0 else 1e-16
        m1, m3, p1, p2 = mv[0], mv[2], float(pars[0]), float(pars[1])
        logw1 = lambda x: -p1 * np.abs((x - m1) / s1)
        logw2 = lambda x: -p2 * np.abs((x - m3) / s2)

        pind0 = pind.copy()
        rows = _pathopt(Np, Fp, Wp, logw1, logw2, log_res)[span]
        assign(rows)
        rdiff = np.count_nonzero(pind[span] != pind0[span]) / (tn2 - tn1 + 1)
        itn += 1
        mv = stats()
        if itn > max_iter and rdiff != 0:
            break
        history.append(pind.copy())
        if rdiff != 0 and itn > 2:                                # cycling guard
            if any(np.array_equal(pind[span], h[span]) for h in history[1:itn - 1]):
                break

    idn, iup = _support_bounds(A, np.nan_to_num(pind, nan=1).astype(int), tn1, tn2)
    tfsupp[1, span] = freq[idn - 1]
    tfsupp[2, span] = freq[iup - 1]

    info.update(pind=pind, pamp=pamp, iterations=itn)
    return (tfsupp, info) if return_info else tfsupp


def rectfr_legacy(tfsupp, TFR, freq, fs, C, D=None, omg=None, chunk=4096):
    """MODA-faithful component reconstruction (port of ``rectfr.m``, 'direct').

    Parameters
    ----------
    tfsupp : (3, L) array
        Output of :func:`ecurve_legacy`.
    TFR, freq, fs
        The complex transform the ridge was extracted from.
    C : float
        The wavelet's (or window's) reconstruction constant, ``wopt.wp.C``.
    D : float, optional
        ``wopt.wp.D`` for a wavelet transform. ``inf`` (e.g. Morlet) selects
        rectfr's hybrid frequency estimate from the phase derivative.
    omg : float, optional
        ``wopt.wp.omg`` for a windowed Fourier transform.

    Returns
    -------
    iamp, iphi, ifreq : (L,) arrays
        Instantaneous amplitude, unwrapped phase and frequency; NaN where the
        support holds a non-finite coefficient or outside the ridge's span.
    """
    W = np.asarray(TFR)
    freq = np.asarray(freq, dtype=float).ravel()
    NF, L = W.shape
    asig = np.full(L, np.nan, dtype=complex)
    ifreq = np.full(L, np.nan)

    valid = np.flatnonzero(~np.isnan(tfsupp[0]))
    if len(valid) == 0:
        return np.abs(asig), np.angle(asig), ifreq
    tn1, tn2 = int(valid[0]), int(valid[-1])

    # rectfr.m's own resolution test (line 119), then frequencies -> bins
    linear = freq[0] <= 0 or _mstd(np.diff(freq)) < _mstd(freq[1:] / freq[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        if linear:
            fstep = float(np.mean(np.diff(freq)))
            bins = 1 + np.floor(0.5 + (tfsupp - freq[0]) / fstep)
            wgt, K = TWO_PI * fstep, omg
        else:
            fstep = float(np.mean(np.diff(np.log(freq))))
            bins = 1 + np.floor(0.5 + (np.log(tfsupp) - np.log(freq[0])) / fstep)
            wgt, K = fstep, D
    if K is None:
        raise ValueError("rectfr_legacy needs D for a wavelet transform and "
                         "omg for a windowed Fourier transform")
    bins = np.clip(bins, 1, NF)
    hybrid = not np.isfinite(K)

    rows1 = np.arange(1, NF + 1)[:, None]
    for c0 in range(tn1, tn2 + 1, chunk):
        cols = np.arange(c0, min(c0 + chunk, tn2 + 1))
        lo, hi = bins[1, cols], bins[2, cols]
        ok_col = ~np.isnan(lo) & ~np.isnan(hi)
        mask = (rows1 >= lo[None, :]) & (rows1 <= hi[None, :])    # (NF, n)
        cs = W[:, cols]
        finite = np.isfinite(cs)
        good = ok_col & ~(mask & ~finite).any(axis=0)
        csm = np.where(mask & finite, cs, 0.0)

        casig = (1.0 / C) * np.sum(csm * wgt, axis=0)
        if not hybrid:
            num = np.sum(freq[:, None] * csm * wgt, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = ((1.0 / C) * num / casig - K) if linear else ((1.0 / K) * num / casig)
        else:
            # phase derivative across time, per bin (rectfr.m lines 241-246)
            # central difference inside the span, forward at the first sample
            # of the record, backward everywhere else
            central = (cols > tn1) & (cols < tn2)
            forward = ~central & (cols == 0)
            a = np.where(central | forward, np.minimum(cols + 1, L - 1), cols)
            b = np.where(forward, cols, np.maximum(cols - 1, 0))
            with np.errstate(invalid="ignore"):
                d = np.angle(W[:, a]) - np.angle(W[:, b])
                d = np.where(d < 0, d + TWO_PI, d)
            cw = d * np.where(central, fs / 2, fs)[None, :] / TWO_PI
            cwm = np.where(mask & finite, cw, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = (1.0 / C) * np.sum(cwm * csm * wgt, axis=0) / casig

        asig[cols[good]] = casig[good]
        ifreq[cols[good]] = np.real(fr[good])

    iamp = np.abs(asig)
    iphi = np.angle(asig)
    fin = np.isfinite(iphi)
    if fin.any():
        iphi[fin] = np.unwrap(iphi[fin])
    return iamp, iphi, ifreq


def wavelet_constants(wavelet="Lognorm", f0=1.0):
    """``(C, D)`` for a built-in MODA wavelet, as ``wt.m`` sets them:
    ``C = (1/2) ∫ conj(ψ̂(ξ)) dξ/ξ`` and ``D = (ω_peak/2) ∫ conj(ψ̂(ξ)) dξ/ξ²``.

    Lognorm has both in closed form (wt.m line 299); Morlet's ``D`` is infinite
    by definition there (line 304), which sends ``rectfr`` down its hybrid
    branch; the rest are integrated numerically, as wt.m does with ``quadgk``.
    """
    from scipy.integrate import quad
    from .legacy_moda import moda_wavelet

    name = wavelet.lower()
    fwt, ompeak, xi1, xi2 = moda_wavelet(wavelet, f0)
    if name in ("lognorm", "lognormal"):
        q = TWO_PI * f0
        C = np.sqrt(np.pi / 2) / q
        return C, C * np.exp(1.0 / (2 * q ** 2))

    lo = np.log(xi1) if xi1 > 0 else -np.inf
    hi = np.log(xi2) if np.isfinite(xi2) else np.inf
    up = np.log(ompeak)
    def f(u):
        with np.errstate(over="ignore"):
            return float(np.real(fwt(np.array([np.exp(u)]))[0]))
    kw = dict(epsabs=0, epsrel=1e-13, limit=500)
    C = 0.5 * (quad(f, lo, up, **kw)[0] + quad(f, up, hi, **kw)[0])
    if name == "morlet":
        return C, np.inf
    g = lambda u: f(u) * np.exp(-u)
    D = (ompeak / 2.0) * (quad(g, lo, up, **kw)[0] + quad(g, up, hi, **kw)[0])
    return C, D


def extract_ridge_legacy(TFR, freq, fs, C, D=None, omg=None):
    """``ecurve`` then ``rectfr(...,'direct')``, as MODA's ridge extraction
    runs them. Returns a dict with ``ifreq``, ``iamp``, ``iphi``, ``recon``
    (``iamp·cos(iphi)``) and ``tfsupp``."""
    tfsupp = ecurve_legacy(TFR, freq, fs)
    iamp, iphi, ifreq = rectfr_legacy(tfsupp, TFR, freq, fs, C, D=D, omg=omg)
    return {"ifreq": ifreq, "iamp": iamp, "iphi": iphi,
            "recon": iamp * np.cos(iphi), "tfsupp": tfsupp}
