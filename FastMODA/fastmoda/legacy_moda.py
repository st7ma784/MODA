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
* MODA's **frequency discretization** (``wt.m``'s ``freq`` vector, ``2^(k/nv)``) with ``nv`` derived from the
  wavelet's 50%-energy support (``'auto'`` = MODA's ``auto-10``);
* the ``p = 1`` amplitude normalisation and frequency-domain convolution
  ``WT = ifft(fx · conj(FW))`` — i.e. **complex** coefficients, not magnitude;
* optional **preprocessing** (cubic-polynomial detrend + band-pass to
  ``[fmin, fmax]``), on by default as in MODA;
* next-power-of-two padding with **zero / symmetric / periodic / predictive**
  modes and a cone-of-influence NaN mask (``cut_edges``);

* **predictive padding** by a port of wt.m's ``fcast``, and the wavelet's time
  supports read off the same grid wt.m inverts the wavelet on, so the cone of
  influence and the default ``fmin`` land where MODA's do.

Against real MODA (docs/validation/changelog-vs-moda.md) every wavelet
transform case agrees to better than 3e-9. ``wt_at_freqs`` (``wtAtf2.m``) and
``bispec_wav_legacy`` (``bispecWavNew.m``) build MODA's wavelet bispectrum on
top of it. ``wft_legacy`` is the same for ``wft.m``, with all six of its
windows, and agrees to 1e-14.
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

def _moda_wavelet_twf(name, f0):
    """Time-domain form of a wavelet where wt.m defines one (Morlet, f0 >= 1,
    its line 303); ``None`` where wt.m derives it numerically."""
    if name.lower() == "morlet" and f0 >= 1:
        om0 = TWO_PI * f0
        return lambda t: ((1 / np.sqrt(TWO_PI)) * (np.exp(1j * om0 * t) - np.exp(-om0 ** 2 / 2))
                          * np.exp(-t ** 2 / 2))
    return None


def _wavelet_params(fwt, ompeak, xi1, xi2, racc=0.01, ngrid=1 << 16, L=None,
                    twf=None):
    """Estimate the frequency/time supports the way wt.m's parcalc / sqeps do.

    * 50%-frequency support: cumulative of ``fwt`` over log-frequency, taken at
      the 25% and 75% points (sqeps ``s1h=fz(0.25)``, ``s2h=fz(0.75)``). Drives
      the auto ``nv``.
    * ε- and 50%-time supports: cumulative of the *demodulated* time-domain
      wavelet. The ε-support drives the cone of influence and the default
      ``fmin``; the 50%-support the weighting of predictive padding.

    ``L`` is the signal length. wt.m inverts the wavelet on a grid whose size
    follows it, and reads the supports off that grid, so they carry its
    discretisation (in the sixth figure). The cone of influence is a ``ceil``
    of the support, so the same grid is used here to land on the same sample.
    ``twf`` is the time-domain form for wavelets where wt.m has one in closed
    form (Morlet with f0 >= 1): there it integrates that form directly, with
    no grid of its own, so a fine one is used here.
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

    # --- time supports: invert fwt -> twf(t), demodulate, cumulate (as MODA) ---
    # wt.m lines 914-1010. The cumulative integral is taken from the left for
    # the lower bound and from the right for the upper one, by the midpoint
    # rule, and each bound is the linear interpolation of its crossing.
    lo_x = max(xi1, 0.0)

    def invert(CL, CT, Etot=None):
        CNq = int(np.ceil((CL + 1) / 2))
        cxi = (TWO_PI / CT) * np.arange(CNq - CL, CNq)
        Cf = np.zeros(CL)
        inside = (cxi > lo_x) & (cxi < xi2)
        with np.errstate(over="ignore", invalid="ignore"):
            Cf[inside] = np.real(fwt(cxi[inside]))
        Cf[~np.isfinite(Cf)] = 0.0
        Ct = np.fft.ifft((CL / CT) * np.concatenate([Cf[CL - CNq:], Cf[:CL - CNq]]))
        Ct = np.concatenate([Ct[CNq:], Ct[:CNq]])
        err = None
        if Etot is not None:                      # wt.m's energy-mismatch estimate
            Et, Ef = np.abs(Ct) ** 2, Cf ** 2
            i1 = (CT / CL) * np.sum(np.abs(Et[2:] - 2 * Et[1:-1] + Et[:-2])) / 24
            i2 = (1 / CT) * np.sum(np.abs(Ef[2:] - 2 * Ef[1:-1] + Ef[:-2])) / 24
            err = (abs(Etot - (CT / CL) * np.sum(Et)) + i1 + i2) / Etot
        return Ct, CNq, err

    if twf is not None:
        CL, CT = 1 << 21, 24.0                    # +-12: unit-width Gaussian envelope
        ct = (CT / CL) * (np.arange(CL + 1) - CL // 2)
        Ct = twf(ct)
    else:
        # energy of the wavelet and the ε-support of |fwt|^2 over xi, which
        # sets the starting span of wt.m's grid
        ue = np.linspace(np.log(xi1 if xi1 > 0 else ompeak * 1e-4),
                         np.log(xi2 if np.isfinite(xi2) else ompeak * 1e4), 1 << 18)
        xe = np.exp(ue)
        with np.errstate(over="ignore", invalid="ignore"):
            ge = np.real(fwt(xe)) ** 2 * xe
        ge[~np.isfinite(ge)] = 0.0
        ce = np.concatenate([[0.0], np.cumsum((ge[1:] + ge[:-1]) / 2 * np.diff(ue))])
        Etot = ce[-1] / TWO_PI
        s1e = np.exp(np.interp(racc / 2.0, ce / ce[-1], ue))
        s2e = np.exp(np.interp(1.0 - racc / 2.0, ce / ce[-1], ue))

        MIC = max(10000, 10 * (L or 0))
        CL = 1 << int(np.ceil(np.log2(MIC / 8)))
        CT = CL / (2 * abs(s2e - s1e))
        Ct, CNq, err = invert(CL, CT, Etot)
        perr = np.inf
        while err <= perr:                        # halve the span while it helps
            CT /= 2
            perr = err
            Ct, CNq, err = invert(CL, CT, Etot)
        CL *= 16
        CT *= 2
        Ct, CNq, _ = invert(CL, CT)
        Ct = Ct[:2 * CNq - 3]
        ct = (CT / CL) * np.arange(-(CNq - 2), CNq - 1)
    Ct = Ct * np.exp(-1j * ompeak * ct)           # demodulate (wt.m line 964)

    dt = CT / CL
    edges = np.concatenate([ct - dt / 2, [ct[-1] + dt / 2]])
    CS = np.concatenate([[0.0], dt * np.cumsum(Ct)]); CS = np.abs(CS / CS[-1])
    ICS = np.concatenate([(dt * np.cumsum(Ct[::-1]))[::-1], [0.0]]); ICS = np.abs(ICS / ICS[0])

    def lower(level):
        k = np.flatnonzero((CS[:-1] < level) & (CS[1:] >= level))
        if not len(k):
            return edges[0]
        k = k[0]; b1, b2 = CS[k] - level, CS[k + 1] - level
        return edges[k] - b1 * (edges[k + 1] - edges[k]) / (b2 - b1)

    def upper(level):
        k = np.flatnonzero((ICS[:-1] >= level) & (ICS[1:] < level))
        if not len(k):
            return edges[-1]
        k = k[-1]; b1, b2 = ICS[k] - level, ICS[k + 1] - level
        return edges[k] - b1 * (edges[k + 1] - edges[k]) / (b2 - b1)

    return dict(xi1h=xi1h, xi2h=xi2h, t1e=lower(racc / 2.0), t2e=upper(racc / 2.0),
                t1h=lower(0.25), t2h=upper(0.25))


def _moda_ff(N, fs):
    """FFT frequencies as wt.m builds them: the Nyquist bin of an even-length
    transform is +fs/2 (numpy's ``fftfreq`` makes it -fs/2)."""
    Nq = int(np.ceil((N + 1) / 2))
    return np.concatenate([np.arange(Nq), -np.arange(1, N - Nq + 1)[::-1]]) * fs / N


def _fit3(Y, t, f, rw):
    """Weighted least-squares fit of ``b0 + b1 cos + b2 sin`` at frequency f,
    with fcast's error measure, the standard deviation of the residual."""
    FM = np.column_stack([np.ones(len(t)), np.cos(TWO_PI * f * t), np.sin(TWO_PI * f * t)])
    FM *= rw[:, None]
    b = np.linalg.lstsq(FM, Y, rcond=None)[0]
    return b, float(np.std(Y - FM @ b, ddof=1)), FM


def _golden(cf, cerr, cb, FTol, fit):
    """fcast's golden-section refinement of a bracketed minimum."""
    cf, cerr, cb = list(cf), list(cerr), list(cb)
    while cf[1] - cf[0] > FTol and cf[2] - cf[1] > FTol:
        tf = cf[0] + cf[2] - cf[1]
        tb, terr = fit(tf)
        # four sequential tests, evaluated in MATLAB's order on the updated bracket
        if terr < cerr[1] and tf < cf[1]:
            cf = [cf[0], tf, cf[1]]; cerr = [cerr[0], terr, cerr[1]]; cb = [cb[0], tb, cb[1]]
        if terr < cerr[1] and tf > cf[1]:
            cf = [cf[1], tf, cf[2]]; cerr = [cerr[1], terr, cerr[2]]; cb = [cb[1], tb, cb[2]]
        if terr > cerr[1] and tf < cf[1]:
            cf = [tf, cf[1], cf[2]]; cerr = [terr, cerr[1], cerr[2]]; cb = [tb, cb[1], cb[2]]
        if terr > cerr[1] and tf > cf[1]:
            cf = [cf[0], cf[1], tf]; cerr = [cerr[0], cerr[1], terr]; cb = [cb[0], cb[1], tb]
    return cf[1], cb[1], cerr[1]


def _fcast(sig, fs, NP, fint, max_order, w):
    """MODA's predictive padding (port of ``fcast`` in wt.m / wft.m).

    Forecasts ``NP`` samples past the end of ``sig`` by fitting sinusoids one
    at a time: take the largest peak of the residual's spectrum, refine its
    frequency by a bracketing-then-golden-section search on the weighted
    least-squares residual (once upward, once downward, keeping the better),
    subtract the fit, and repeat until the Bayesian information criterion has
    risen twice or ``max_order`` components are found. Components inside
    ``fint`` are continued in time; the rest are held at their final value.
    ``w`` weights recent samples more heavily.
    """
    if NP <= 0:
        return np.zeros(0)
    sig = np.asarray(sig, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    rw = np.sqrt(w)
    Y = rw * sig
    ok = np.flatnonzero(rw[::-1] / rw.max() >= 1e-8)
    L = int(ok[-1]) + 1
    T = L / fs
    t = np.arange(L) / fs
    rw, Y = rw[-L:], Y[-L:].copy()
    max_order = int(min(max_order, L // 3))

    FTol = 1.0 / T / 100.0
    rr = (1 + np.sqrt(5)) / 2
    Nq = int(np.ceil((L + 1) / 2))
    ftfr = _moda_ff(L, fs)
    orstd = float(np.std(Y, ddof=1))
    eps = np.finfo(float).eps

    frq = [0.0]; amp = [0.0]; phi = [0.0]; ic = []
    itn = 0
    while itn < max_order:
        itn += 1
        imax = int(np.argmax(np.abs(np.fft.fft(Y))[1:Nq - 1])) + 1
        fit = lambda f: _fit3(Y, t, f, rw)[:2]
        results = []
        for direction in (+1, -1):                 # forward, then backward search
            nf = ftfr[imax]
            nb, nerr = fit(nf)
            df, perr, pf, pb = FTol, np.inf, nf, nb
            while nerr < perr:
                if direction > 0 and abs(nf - fs / 2 + FTol) < eps:
                    break
                if direction < 0 and abs(nf - FTol) < eps:
                    break
                pf, perr, pb = nf, nerr, nb
                nf = min(pf + df, fs / 2 - FTol) if direction > 0 else max(pf - df, FTol)
                nb, nerr = fit(nf)
                df *= rr
            if nerr < perr:
                cf, cerr, cb = [nf] * 3, [nerr] * 3, [nb] * 3
            elif abs(nf - pf - FTol) < eps:
                cf, cerr, cb = [pf] * 3, [perr] * 3, [pb] * 3
            elif direction > 0:
                f0 = pf - df / rr / rr
                b0, e0 = fit(f0)
                cf, cerr, cb = [f0, pf, nf], [e0, perr, nerr], [b0, pb, nb]
            else:
                # fcast fits the third bracket point at cf(1), the first
                # point's frequency, not its own; kept as it is in MODA
                f2 = pf + df / rr / rr
                b2, e2 = fit(nf)
                cf, cerr, cb = [nf, pf, f2], [nerr, perr, e2], [nb, pb, b2]
            results.append(_golden(cf, cerr, cb, FTol, fit))
        (fcf, fcb, fcerr), (bcf, bcb, bcerr) = results
        cf, cb, cerr = (fcf, fcb, fcerr) if fcerr < bcerr else (bcf, bcb, bcerr)

        frq.append(cf); amp.append(float(np.hypot(cb[1], cb[2])))
        phi.append(float(np.arctan2(-cb[2], cb[1])))
        amp[0] += cb[0]
        Y -= _fit3(Y, t, cf, rw)[2] @ cb

        with np.errstate(divide="ignore"):
            ic.append(L * np.log(cerr) + (3 * itn + 1) * np.log(L))
        if cerr / orstd < 2 * eps:
            break
        if itn > 2 and ic[-1] > ic[-2] and ic[-2] > ic[-3]:
            break

    nt = T + np.arange(NP) / fs
    fsig = np.zeros(NP)
    for f, a, ph in zip(frq, amp, phi):
        if fint[0] < f < fint[1]:
            fsig += a * np.cos(TWO_PI * f * nt + ph)
        else:
            fsig += a * np.cos(TWO_PI * f * (T - 1 / fs) + ph)
    return fsig


def _predictive_pads(sig, fs, n1, n2, fmin, fmax, SN, t1h, t2h):
    """Left and right predictive padding as wt.m / wft.m request it."""
    L = len(sig)
    w = 2.0 ** (-(L / fs - np.arange(1, L + 1) / fs) / (t2h - t1h))
    fint = (max(fmin, fs / L), fmax)
    order = min(int(np.ceil(SN / 2)) + 5, int(np.floor(L / 3 + 0.5)))
    left = _fcast(sig[::-1], fs, n1, fint, order, w)[::-1]
    right = _fcast(sig, fs, n2, fint, order, w)
    return left, right


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
        return None, twf, -np.inf, np.inf   # known in time only; see moda_window_support
    raise ValueError(f"Unknown window '{name}'.")


def moda_window_support(name: str = "Gaussian", f0: float = 1.0):
    """Time support ``(t1, t2)`` of a built-in MODA window, as wft.m sets
    ``wp.t1``/``wp.t2``: finite for Hann, Blackman, Rect and Kaiser, infinite
    for Gaussian and Exp."""
    name = name.lower()
    if name == "hann":
        q = 4.4 * f0
    elif name == "blackman":
        q = 5.6 * f0
    elif name in ("rect", "rectangular", "boxcar"):
        q = 10.0 * f0
    elif "kaiser" in name:
        a = 3.0 if len(name) <= 6 else float(name.split("-")[1])
        q = 3 * np.sqrt(1 + abs(a - 1 / a)) * f0
    else:
        return -np.inf, np.inf
    return -q / 2, q / 2


_WINDOW_PARAMS = {}


def _window_params(name, f0, fwt, twf, racc=0.01):
    """WFT analogue of ``_wavelet_params``: the window's ε- and 50%-supports in
    time (cone of influence, predictive-padding weights) and its 50%-support
    in frequency (the automatic frequency step).

    wft.m knows every built-in window in the time domain, so it gets the time
    supports by integrating ``twf`` over the window's own support with
    ``quadgk``: they are quantiles of the window's area, ``racc/2`` and
    ``1 - racc/2`` for the ε-support and 25%/75% for the 50%-support. Here
    they are the same quantiles, in closed form where there is one and by
    quadrature and root-finding otherwise. The window is zero outside
    ``moda_window_support``; integrating a periodic formula such as Hann's
    past that would count its repeats.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import ndtri

    lname = name.lower()
    key = (lname, float(f0), float(racc))
    if key in _WINDOW_PARAMS:                       # depends on nothing else
        return dict(_WINDOW_PARAMS[key])
    t1, t2 = moda_window_support(name, f0)

    if lname in ("gaussian", "wft"):
        te, th = f0 * ndtri(1 - racc / 2.0), f0 * ndtri(0.75)
    elif lname in ("exp", "exponential"):
        q = 6.5 * f0
        te, th = -q * np.log(racc), -q * np.log(0.5)
    elif lname in ("rect", "rectangular", "boxcar"):
        te, th = t2 * (1 - racc), t2 * 0.5
    else:
        # symmetric compact window: the point where the area left outside it
        # first reaches the given fraction, coming in from the edge. "First"
        # matters for MODA's Blackman, which dips below zero near its edges:
        # the magnitude of that small negative area reaches racc/2 well before
        # the positive bulk does, and that is the crossing sqeps reports.
        f = lambda x: float(np.real(twf(np.array([x]))[0]))
        kw = dict(epsabs=0, epsrel=1e-13, limit=200)
        total = 2 * quad(f, 0.0, t2, **kw)[0]
        xs = np.linspace(t2, 0.0, 20001)
        ys = np.real(twf(xs))
        tail = np.abs(np.concatenate([[0.0], np.cumsum((ys[1:] + ys[:-1]) / 2 * np.diff(xs))]))

        def upper(frac):
            i = int(np.argmax(tail / abs(total) >= frac))
            g = lambda x: abs(quad(f, x, t2, **kw)[0] / total) - frac
            return brentq(g, xs[min(i + 1, len(xs) - 1)], xs[max(i - 2, 0)],
                          xtol=1e-15, rtol=1e-15)

        te, th = upper(racc / 2.0), upper(0.25)

    # 50%-frequency support: cumulate fwt over a linear xi axis, 25%/75% points
    if lname in ("gaussian", "wft"):
        xi2h = ndtri(0.75) / f0
        xi1h = -xi2h
    elif lname in ("exp", "exponential"):
        xi2h = 1.0 / (6.5 * f0)
        xi1h = -xi2h
    else:
        # The point where the area of the window's Fourier transform, counted
        # outward from its peak at zero, first reaches half of one side's.
        kw = dict(epsabs=0, epsrel=1e-12, limit=2000)
        ft = lambda x: float(np.real(twf(np.array([x]))[0]))
        if fwt is None:
            # known in time only (Kaiser): the transform is its cosine integral
            g = lambda x: 2 * quad(lambda t: ft(t) * np.cos(x * t), 0.0, t2,
                                   epsabs=1e-13, epsrel=1e-11, limit=200)[0]
        else:
            def g(x):
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = float(np.real(fwt(np.array([x]))[0]))
                    if not np.isfinite(v):          # removable singularity
                        v = float(np.real(fwt(np.array([x + 1e-12]))[0]))
                return v
        if fwt is None or lname in ("rect", "rectangular", "boxcar"):
            side = np.pi * ft(0.0)                  # the transform's area is 2 pi twf(0)
        else:
            side = quad(g, 0.0, 400.0 / f0, **kw)[0]
        xs = np.linspace(0.0, 8.0 / f0, 801)
        gs = np.array([g(x) for x in xs])
        cum = np.concatenate([[0.0], np.cumsum((gs[1:] + gs[:-1]) / 2 * np.diff(xs))])
        i = int(np.argmax(cum >= 0.5 * side))
        xi2h = brentq(lambda x: quad(g, 0.0, x, **kw)[0] - 0.5 * side,
                      xs[max(i - 2, 0)], xs[min(i + 1, len(xs) - 1)], xtol=1e-14)
        xi1h = -xi2h
    _WINDOW_PARAMS[key] = dict(xi1h=xi1h, xi2h=xi2h, t1e=-te, t2e=te, t1h=-th, t2h=th,
                               t1=t1, t2=t2)
    return dict(_WINDOW_PARAMS[key])


def wft_legacy(signal, fs, fmin=None, fmax=None, window="Gaussian", f0=1.0,
               fstep="auto", padding="predictive", preprocess=True,
               cut_edges=False, return_freq=True, block_elems=4_000_000):
    """MODA-faithful windowed Fourier transform (port of ``wft.m``).

    Unlike the CWT the window is **shifted** (not dilated) to each frequency, the
    frequency grid is **linear** (``fstep``), the cone of influence is constant
    across frequency, and the filter is applied **without** conjugation (see
    wft.m line 523). Returns ``(WFT, freq)`` with complex ``WFT`` of shape
    ``(n_freq, len(signal))``.

    ``window`` is one of MODA's: Gaussian, Hann, Blackman, Exp, Rect or
    Kaiser-a (``a`` = 3 if omitted). Defaults follow wft.m: predictive padding,
    preprocessing on, ``cut_edges`` off.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    L = len(sig)
    if fmax is None:
        fmax = fs / 2.0
    fwt, twf, _, _ = moda_window(window, f0)
    wp = _window_params(window, f0, fwt, twf)

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
    if wp["t2e"] - wp["t1e"] > L / fs:              # no cone of influence at all
        cut_edges = False

    predictive = padding == "predictive"
    dflag = predictive and fmin < 5 * fs / L           # as in wt_legacy
    if preprocess and not dflag:
        sig = _preprocess(sig, L, fs, fmin, fmax)

    NL = 1 << int(np.ceil(np.log2(L + coib1 + coib2)))
    if coib1 == 0 and coib2 == 0:
        n1 = (NL - L) // 2
    else:
        n1 = int(np.floor((NL - L) * coib1 / (coib1 + coib2)))
    n2 = NL - L - n1

    if predictive:
        padleft, padright = _predictive_pads(sig, fs, n1, n2, fmin, fmax, SN,
                                             wp["t1h"], wp["t2h"])
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
    if preprocess and predictive:
        sigp = _detrend_poly(sigp, fs)

    ff = _moda_ff(NL, fs)
    fx = np.fft.fft(sigp)
    if preprocess:
        fx = np.where((ff <= max(fmin, fs / L)) | (ff >= fmax), 0.0, fx)

    if fwt is None:
        # window known only in time (Kaiser): wft.m's twf branch. The window
        # is evaluated once on the FFT's own time axis; each frequency is that
        # window modulated, then transformed.
        h = int(np.ceil((NL - 1) / 2))
        timewf = np.concatenate([-np.arange(1, h + 1) + 1.0,
                                 NL + 1.0 - np.arange(h + 1, NL + 1)]) / fs
        tw = np.zeros(NL)
        inside = (timewf > wp["t1"]) & (timewf < wp["t2"])
        tw[inside] = np.real(twf(timewf[inside]))
        tw[~np.isfinite(tw)] = 0.0

    WFT = np.empty((SN, L), dtype=np.complex128)
    blk = max(1, min(SN, block_elems // max(NL, 1)))
    for b0 in range(0, SN, blk):
        fr = freq[b0:b0 + blk]
        if fwt is None:
            FW = np.fft.fft(tw[:, None] * np.exp(-1j * TWO_PI * fr[None, :] * timewf[:, None]),
                            axis=0) / fs
        else:
            # shifted (not dilated) frequency axis; NO conjugation (wft.m 518-523)
            arg = TWO_PI * (fr[None, :] - ff[:, None])              # NL x nb
            with np.errstate(divide="ignore", invalid="ignore"):
                FW = np.real(fwt(arg))
                bad = ~np.isfinite(FW)      # removable singularities (sinc-type)
                if bad.any():
                    FW[bad] = np.real(fwt(arg[bad] + 1e-14))
            FW[~np.isfinite(FW)] = 0.0
        WFT[b0:b0 + blk] = np.fft.ifft(fx[:, None] * FW, axis=0)[n1:NL - n2, :].T

    if cut_edges and coib1 + coib2 < L:
        if coib1 > 0:
            WFT[:, :coib1] = np.nan
        if coib2 > 0:
            WFT[:, L - coib2:] = np.nan

    return (WFT, freq) if return_freq else WFT


def _preprocess(sig, N, fs, fmin, fmax):
    """wt.m's preprocessing: cubic detrend, then zero every Fourier component
    outside ``(max(fmin, fs/N), fmax)``."""
    sig = _detrend_poly(sig, fs)
    fx = np.fft.fft(sig, N)
    ff = _moda_ff(N, fs)
    fx[(np.abs(ff) <= max(fmin, fs / N)) | (np.abs(ff) >= fmax)] = 0.0
    return np.real(np.fft.ifft(fx))


def wt_legacy(signal, fs, fmin=None, fmax=None, wavelet="Lognorm", f0=1.0,
              nv="auto", padding="predictive", preprocess=True,
              cut_edges=True, return_freq=True, return_opt=False):
    """MODA-faithful continuous wavelet transform (port of ``wt.m``).

    Parameters mirror ``wt.m``. Returns ``(WT, freq)`` where ``WT`` is a complex
    ``(n_freq, len(signal))`` array (rows = frequencies, cols = time) and
    ``freq`` MODA's frequency discretization (``2^(k/nv)``). With ``cut_edges=True`` (MODA
    default), coefficients outside the cone of influence are ``NaN``.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    L = len(sig)
    if fmax is None:
        fmax = fs / 2.0
    fwt, ompeak, xi1, xi2 = moda_wavelet(wavelet, f0)
    wp = _wavelet_params(fwt, ompeak, xi1, xi2, L=L,
                         twf=_moda_wavelet_twf(wavelet, f0))

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

    # MODA's frequency discretization (wt.m line 371): freq = 2^(k/nv), k integer,
    # i.e. each bin is the previous one times 2^(1/nv), nv = "number of voices"
    k0 = int(np.ceil(nv * np.log2(fmin)))
    k1 = int(np.floor(nv * np.log2(fmax)))
    freq = 2.0 ** (np.arange(k0, k1 + 1) / nv)
    SN = len(freq)

    coib1 = np.ceil(np.abs(wp["t1e"] * fs * (ompeak / (TWO_PI * freq)))).astype(int)
    coib2 = np.ceil(np.abs(wp["t2e"] * fs * (ompeak / (TWO_PI * freq)))).astype(int)

    sig_orig = sig.copy()
    predictive = padding == "predictive"
    # wt.m's dflag: detrend and filter before padding (0) or after it (1).
    # Predictive padding at a low fmin forecasts from the raw signal.
    dflag = predictive and fmin < 5 * fs / L

    # preprocessing before padding (detrend + band-pass), as in wt.m default
    if preprocess and not dflag:
        sig = _preprocess(sig, L, fs, fmin, fmax)

    # padding to next power of two, split by COI ratio
    NL = 1 << int(np.ceil(np.log2(L + coib1[0] + coib2[0])))
    if coib1[0] == 0 and coib2[0] == 0:
        n1 = (NL - L) // 2
        n2 = NL - L - n1
    else:
        n1 = int(np.floor((NL - L) * coib1[0] / (coib1[0] + coib2[0])))
        n2 = NL - L - n1

    if predictive:
        padleft, padright = _predictive_pads(sig, fs, n1, n2, fmin, fmax, SN,
                                             wp["t1h"], wp["t2h"])
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
    # wt.m detrends once more after predictive padding (its line 449)
    if preprocess and predictive:
        sigp = _detrend_poly(sigp, fs)

    # fft of padded signal, band-pass again if preprocessing
    ff = _moda_ff(NL, fs)
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

    if return_opt:
        # what wt.m returns as wopt, as far as the routines downstream of the
        # transform (wt_at_freqs, the bispectrum, ridge reconstruction) need it
        opt = dict(signal=sig_orig, fs=fs, fmin=fmin, fmax=fmax, nv=nv, f0=f0,
                   wavelet=wavelet, padding=padding, preprocess=preprocess,
                   cut_edges=cut_edges, padleft=padleft, padright=padright,
                   fwt=fwt, ompeak=ompeak, xi1=xi1, xi2=xi2,
                   t1e=wp["t1e"], t2e=wp["t2e"], t1h=wp["t1h"], t2h=wp["t2h"])
        return WT, freq, opt
    return (WT, freq) if return_freq else WT


def wt_at_freqs(signal, fs, fr, opt, block_elems=4_000_000):
    """Wavelet transform of ``signal`` at arbitrary frequencies ``fr``.

    Port of MODA's ``wtAtf2.m`` (and its batched form ``wtAtf2_batch.m``),
    which the bispectrum uses to evaluate the transform at exactly
    ``f1 + f2``, a frequency that in general lies between two bins of the
    grid. ``opt`` is the dict :func:`wt_legacy` returns with
    ``return_opt=True`` for the same signal, so the padding is the one that
    transform used. Returns an ``(len(fr), len(signal))`` complex array.

    Two details are MODA's and are kept: the split of the padding between the
    two ends is recomputed from each frequency's own cone of influence, and
    with ``cut_edges`` the upper mask covers one sample more than
    :func:`wt_legacy`'s does.
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    fr = np.atleast_1d(np.asarray(fr, dtype=np.float64))
    L = len(sig)
    fmin, fmax = opt["fmin"], opt["fmax"]
    pre = opt["preprocess"]
    fwt, ompeak, xi1, xi2 = opt["fwt"], opt["ompeak"], opt["xi1"], opt["xi2"]
    dflag = opt["padding"] == "predictive" and fmin < 5 * fs / L

    if pre and not dflag:
        sig = _preprocess(sig, L, fs, fmin, fmax)
    sigp = np.concatenate([opt["padleft"], sig, opt["padright"]])
    NL = len(sigp)
    if pre and dflag:
        sigp = _preprocess(sigp, NL, fs, 0.0, fs / 2.0)

    ff = _moda_ff(NL, fs)
    fx = np.fft.fft(sigp)
    fx[ff <= 0] = 0.0
    if pre:
        fx[(ff <= max(fmin, fs / L)) | (ff >= fmax)] = 0.0

    scale = ompeak / (TWO_PI * fr)
    coib1 = np.ceil(np.abs(opt["t1e"] * fs * scale)).astype(int)
    coib2 = np.ceil(np.abs(opt["t2e"] * fs * scale)).astype(int)
    if (opt["t2e"] - opt["t1e"]) * ompeak / (TWO_PI * fmax) > L / fs:
        coib1[:] = 0; coib2[:] = 0
    zero = (coib1 == 0) & (coib2 == 0)
    tot = np.where(zero, 1, coib1 + coib2)
    n1 = np.where(zero, (NL - L) // 2, np.floor((NL - L) * coib1 / tot)).astype(int)

    M = len(fr)
    out = np.full((M, L), np.nan, dtype=np.complex128)
    blk = max(1, min(M, block_elems // max(NL, 1)))
    for b0 in range(0, M, blk):
        sl = slice(b0, min(b0 + blk, M))
        freqwf = ff[:, None] * scale[None, sl]
        in_supp = (freqwf > xi1 / TWO_PI) & (freqwf < xi2 / TWO_PI)
        FW = np.zeros(freqwf.shape)
        vals = np.real(np.conj(fwt(TWO_PI * freqwf[in_supp])))
        vals[~np.isfinite(vals)] = 0.0
        FW[in_supp] = vals
        full = np.fft.ifft(fx[:, None] * FW, axis=0)              # (NL, nb)
        for i, m in enumerate(range(sl.start, sl.stop)):
            out[m] = full[n1[m]:n1[m] + L, i]
            if opt["cut_edges"]:
                out[m, :coib1[m]] = np.nan
                out[m, max(L - 1 - coib2[m], 0):] = np.nan
    return out


def bispec_wav_legacy(sig1, sig2, fs, progress=None, **wt_kwargs):
    """MODA-faithful wavelet bispectrum (port of ``bispecWavNew.m``).

    ``Bisp[j, k] = < WT1(f_j) · WT2(f_k) · conj(WT2(f_j + f_k)) >_t``, with the
    third transform evaluated at exactly ``f_j + f_k`` by :func:`wt_at_freqs`,
    on MODA's full frequency grid. Cells whose sum frequency is above the top
    of the grid, or does not clear the next bin above the larger of the two,
    are NaN, as are the cells below the diagonal when both signals are the
    same.

    ``wt_kwargs`` go to :func:`wt_legacy` (``fmin``, ``fmax``, ``f0``,
    ``wavelet``, ``padding``, ``preprocess``, ``cut_edges``, ``nv``).
    Returns ``(Bisp, freq, opt, WT1, WT2)`` as ``bispecWavNew`` does.
    """
    wt1, freq, _ = wt_legacy(sig1, fs, return_opt=True, **wt_kwargs)
    wt2, freq, opt = wt_legacy(sig2, fs, return_opt=True, **wt_kwargs)
    nfreq = len(freq)
    Bisp = np.full((nfreq, nfreq), np.nan, dtype=np.complex128)
    auto = np.array_equal(wt1, wt2, equal_nan=True)

    for j in range(nfreq):
        k = np.arange(j if auto else 0, nfreq)
        f3 = freq[j] + freq[k]
        idx3 = np.searchsorted(freq, f3, side="left")             # first freq >= f3
        below = freq[np.maximum(idx3 - 1, 0)]
        valid = (f3 <= freq[-1]) & (below > freq[np.maximum(j, k)])
        k = k[valid]
        if len(k):
            W3 = wt_at_freqs(sig2, fs, f3[valid], opt)
            with np.errstate(invalid="ignore"):
                xx = wt1[j][None, :] * wt2[k] * np.conj(W3)
                n_ok = np.sum(~np.isnan(xx), axis=1)
                Bisp[j, k] = np.where(n_ok > 0, np.nansum(xx, axis=1) / np.maximum(n_ok, 1), np.nan)
        if progress is not None:
            progress(j + 1, nfreq)
    return Bisp, freq, opt, wt1, wt2
