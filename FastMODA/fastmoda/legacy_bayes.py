"""MODA-faithful port of the Bayesian module and its filters.

Ports of the MATLAB files in ``allguis/guis/bayesian/Functions`` (and the
Filtering module's ``loop_butter``), kept line for line where the order of
operations matters:

========================  ==============================================
``full_bayesian``         ``full_bayesian.m`` - the whole chain
``bayes_main``            ``bayes_main.m`` - window loop and propagation
``bayes_phs``             ``bayesPhs.m`` - inference in one window
``dirc``                  ``dirc.m`` - coupling strengths and direction
``cf_print``              ``CFprint.m`` - coupling functions on a grid
``surrogate_cpp``         ``surrcalc.m``, method ``'CPP'``
``loop_butter_bayes``     ``bayesian/Functions/loop_butter.m``
``loop_butter_filtering`` ``filtering/Functions/loop_butter.m``
========================  ==============================================

MODA ships two functions called ``loop_butter`` with different stopping
rules. ``full_bayesian.m`` reaches the Bayesian one (it is first on MODA's
path); the Filtering module's GUI uses its own.

Both raise the Butterworth order until the filter's output blows up and then
step back, so the order they settle on is wherever the transfer-function form
stops being numerically stable. That makes the filter stage, in MODA itself,
sensitive to rounding: at the order MODA picks, changing one coefficient by
one unit in the last place moves MATLAB's own output by 1e-4 to 1e-2, and one
order higher it can diverge (to 1e23) or not depending on rounding alone. No
port can match MATLAB more closely than MATLAB matches itself there, and on
some signals this one settles on an order one higher or lower than MATLAB
does. Everything after the filter (``bayes_main``, ``dirc``, ``CFprint``)
agrees with MODA to 1e-8 when given MODA's phases.

To stay as close as possible the design follows MATLAB's own route
(``butter_matlab``) and ``filtfilt`` pads ``3*(nfilt-1)`` samples as MATLAB
does, where SciPy's default is ``3*nfilt``. ``tests/parity/moda_diff``
checks all of this against MATLAB.
"""
import numpy as np
from scipy.signal import filtfilt, hilbert

from fastmoda.legacy_moda import unwrap_matlab

_EPS = np.finfo(float).eps


# ── filters ──────────────────────────────────────────────────────────────────
#
# MATLAB designs [b,a] through state space: Butterworth prototype poles ->
# zp2ss -> lp2bp -> bilinear -> den = poly(eig(A)). SciPy goes through
# zeros/poles/gain instead. The two agree in exact arithmetic, but the
# transfer-function form is ill-conditioned at the orders loop_butter climbs
# to, and there the difference decides which order it settles on. So the
# design below follows MATLAB's route step by step (R2026a's butter.m,
# buttap.m, zp2ss.m, cplxpair.m, lp2bp.m, bilinear.m, poly.m).

def _poly(e):
    """MATLAB ``poly`` of a vector of roots (left-to-right products)."""
    e = np.asarray(e)
    e = e[np.isfinite(e)]
    c = np.zeros(len(e) + 1, dtype=complex)
    c[0] = 1
    for j in range(len(e)):
        c[1:j + 2] = c[1:j + 2] - e[j] * c[:j + 1]
    pos = np.sort_complex(e[e.imag > 0]); neg = np.sort_complex(np.conj(e[e.imag < 0]))
    if len(pos) == len(neg) and np.array_equal(pos, neg):
        return c.real
    return c


def _cplxpair0(x):
    """MATLAB ``cplxpair(x, 0)`` for vectors of exact conjugate pairs and reals."""
    x = np.asarray(x, dtype=complex)
    is_real = np.abs(x.imag) <= 0
    reals = np.sort(x[is_real].real)
    xc = x[~is_real]
    xc = xc[np.argsort(xc.real, kind="stable")]
    out = []
    while len(xc):
        grp = np.flatnonzero(np.abs(xc.real - xc[0].real) <= 0)
        xq = xc[grp][np.argsort(xc[grp].imag, kind="stable")]
        nn = len(xq)
        top = xq[nn - 1:nn // 2 - 1:-1] if nn // 2 > 0 else xq[::-1]
        for v in top:
            out += [np.conj(v), v]
        xc = np.delete(xc, grp)
    return np.concatenate([np.array(out, dtype=complex), reals.astype(complex)])


def _ldiv_diag(t, X):
    """``t\\X`` for diagonal t: MATLAB divides each row by t's diagonal."""
    return X / np.diag(t)[:, None]


def butter_matlab(n, Wn):
    """``[b,a] = butter(n, Wn)`` for a digital band-pass, as MATLAB computes it."""
    n = int(n)
    # buttap
    ptemp = np.exp(1j * (np.pi * np.arange(1, n, 2) / (2 * n) + np.pi / 2))
    m = len(ptemp)
    p = np.empty(2 * m + (n % 2), dtype=complex)
    p[0:2 * m:2] = ptemp
    p[1:2 * m:2] = np.conj(ptemp)
    if n % 2:
        p[-1] = -1
    kp = np.prod(-p).real
    # zp2ss (no zeros)
    pf = _cplxpair0(p)
    npl = len(pf)
    a = np.zeros((npl, npl)); bvec = np.zeros(npl); c = np.zeros(npl); d = 1.0
    odd = False
    if npl % 2:
        a[0, 0] = pf[npl - 1].real; bvec[0] = 1; c[0] = 1; d = 0.0
        npl -= 1; odd = True
    i = 1
    while i < npl:
        pr_ = pf[i - 1:i + 1]
        den = _poly(pr_).real
        wn = np.sqrt(np.prod(np.abs(pr_)))
        if wn == 0:
            wn = 1.0
        t = np.diag([1.0, 1.0 / wn])
        a1 = _ldiv_diag(t, np.array([[-den[1], -den[2]], [1.0, 0.0]])) @ t
        b1 = _ldiv_diag(t, np.array([[1.0], [0.0]]))[:, 0]
        c1 = np.array([0.0, 1.0]) @ t
        d1 = 0.0
        j = i - 1 if odd else i - 2
        if j == -1:
            a[0:2, 0:2] = a1; c[0:2] = c1
        else:
            a[j + 1:j + 3, 0:j + 1] = np.outer(b1, c[0:j + 1])
            a[j + 1:j + 3, j + 1:j + 3] = a1
            c[0:j + 1] = d1 * c[0:j + 1]
            c[j + 1:j + 3] = c1
        bvec[j + 1:j + 3] = b1 * d
        d = d * d1
        i += 2
    # (b, c, d only feed the state-space outputs; [b,a] needs A, poles and k)
    # pre-warp, lp2bp
    fs = 2.0
    u = 2 * fs * np.tan(np.pi * np.asarray(Wn, float) / fs)
    Bw = u[1] - u[0]
    Wn1 = np.sqrt(u[0] * u[1])
    q = Wn1 / Bw
    ma = a.shape[0]
    at = Wn1 * np.block([[a / q, np.eye(ma)], [-np.eye(ma), np.zeros((ma, ma))]])
    # bilinear
    tt = 1 / fs
    I = np.eye(2 * ma)
    ad = np.linalg.solve(I - at * tt / 2, I + at * tt / 2)
    pz = np.linalg.eigvals(ad) + 0j
    # buttzeros (digital band-pass)
    wd = 2 * np.arctan2(Wn1, 4)
    z = np.concatenate([np.ones(n), -np.ones(n)])
    zWn = np.exp(1j * wd)
    k = (np.prod(zWn - pz) / np.prod(zWn - z)).real
    den = _poly(pz).real
    num = k * _poly(z).real
    return num, den


def bandpass_butter(x, n, flp, fhi, fs):
    """``bandpass_butter.m``: n-th order Butterworth band-pass, zero phase."""
    fnq = fs / 2.0
    b, a = butter_matlab(n, [flp / fnq, fhi / fnq])
    nfilt = max(len(a), len(b))
    return filtfilt(b, a, np.asarray(x, float), padtype="odd",
                    padlen=3 * (nfilt - 1))


def _mmax(x):
    """MATLAB ``max`` of a vector: NaNs are ignored unless all are NaN."""
    x = np.asarray(x, float)
    return np.nan if np.all(np.isnan(x)) else float(np.nanmax(x))


def loop_butter_bayes(s_in, band, fs, max_order=200):
    """Bayesian module's ``loop_butter.m``. Returns ``(s_out, order)``.

    Raises the order until ``max(s_out) >= 10*max(s_in)`` (signed maxima, as
    MODA has it), then uses one order below that.
    """
    s_in = np.asarray(s_in, float)
    order = 1
    s_out = s_in
    max_out = _mmax(s_in)
    ref = 10 * _mmax(s_in)
    while max_out < ref:
        order += 1
        if order > max_order:
            raise RuntimeError("loop_butter: no order up to %d made the filter "
                               "blow up; MODA would loop forever here" % max_order)
        s_out = bandpass_butter(s_in, order, band[0], band[1], fs)
        max_out = _mmax(s_out)
    order -= 1
    return bandpass_butter(s_in, order, band[0], band[1], fs), order


def loop_butter_filtering(s_in, band, fs, max_order=200):
    """Filtering module's ``loop_butter.m``. Returns ``(s_out, order)``.

    Raises the order until ``max|s_out|`` reaches ten times the first-order
    output's, then uses two orders below that (at least 1).
    """
    s_in = np.asarray(s_in, float)
    order = 1
    s_out = bandpass_butter(s_in, order, band[0], band[1], fs)
    max_out = _mmax(np.abs(s_out))
    while _mmax(np.abs(s_out)) < 10 * max_out:
        order += 1
        if order > max_order:
            raise RuntimeError("loop_butter: no order up to %d made the filter "
                               "blow up; MODA would loop forever here" % max_order)
        s_out = bandpass_butter(s_in, order, band[0], band[1], fs)
    order = max(order - 2, 1)
    return bandpass_butter(s_in, order, band[0], band[1], fs), order


# ── linear algebra as MATLAB does it ─────────────────────────────────────────

def _mldivide(A, B):
    """``A\\B``: Cholesky when A is exactly symmetric with a positive diagonal
    (MATLAB tries that first), LU otherwise."""
    if np.array_equal(A, A.T) and np.all(np.diag(A) > 0):
        try:
            Lc = np.linalg.cholesky(A)
            y = np.linalg.solve(Lc, B)
            return np.linalg.solve(Lc.T, y)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.solve(A, B)


def _check_conditioning(E):
    """bayesPhs turns MATLAB's singular-matrix warning on inv(E) into an error."""
    with np.errstate(all="ignore"):
        rc = 1.0 / np.linalg.cond(E, 1)
    if not np.isfinite(rc) or rc < _EPS:
        raise ValueError(
            "One step of the calculations produced a singular matrix, which can "
            "lead to wrong and imprecise results. This is probably due to the "
            "wrong parameters being used. Please check the input parameters, "
            "signals or base functions.")


# ── inference ────────────────────────────────────────────────────────────────

def _base_functions(phi1, phi2, K, bn):
    """``calculateP``: the Fourier base functions, K x N."""
    p = np.zeros((K, len(phi1)))
    p[0] = 1
    br = 0
    for i in range(1, bn + 1):
        p[br + 1] = np.sin(i * phi1); p[br + 2] = np.cos(i * phi1); br += 2
    for i in range(1, bn + 1):
        p[br + 1] = np.sin(i * phi2); p[br + 2] = np.cos(i * phi2); br += 2
    for i in range(1, bn + 1):
        for j in range(1, bn + 1):
            p[br + 1] = np.sin(i * phi1 + j * phi2); p[br + 2] = np.cos(i * phi1 + j * phi2); br += 2
            p[br + 1] = np.sin(i * phi1 - j * phi2); p[br + 2] = np.cos(i * phi1 - j * phi2); br += 2
    return p


def _base_derivatives(phi1, phi2, K, bn, mr):
    """``calculateV``: derivatives of the base functions in variable ``mr``."""
    v = np.zeros((K, len(phi1)))
    br = 0
    if mr == 1:
        for i in range(1, bn + 1):
            v[br + 1] = i * np.cos(i * phi1); v[br + 2] = -i * np.sin(i * phi1); br += 2
        br += 2 * bn
        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                v[br + 1] = i * np.cos(i * phi1 + j * phi2); v[br + 2] = -i * np.sin(i * phi1 + j * phi2); br += 2
                v[br + 1] = i * np.cos(i * phi1 - j * phi2); v[br + 2] = -i * np.sin(i * phi1 - j * phi2); br += 2
    else:
        br += 2 * bn
        for i in range(1, bn + 1):
            v[br + 1] = i * np.cos(i * phi2); v[br + 2] = -i * np.sin(i * phi2); br += 2
        for i in range(1, bn + 1):
            for j in range(1, bn + 1):
                v[br + 1] = j * np.cos(i * phi1 + j * phi2); v[br + 2] = -j * np.sin(i * phi1 + j * phi2); br += 2
                v[br + 1] = -j * np.cos(i * phi1 - j * phi2); v[br + 2] = j * np.sin(i * phi1 - j * phi2); br += 2
    return v


def bayes_phs(Cpr, XIpr, h, max_loops, eps, phi1, phi2, bn):
    """``bayesPhs.m``: posterior parameters, concentration and noise for one window.

    ``Cpr`` is K x 2, ``XIpr`` M x M. Returns ``(Cpt, XIpt, E)``.
    """
    phi1 = np.asarray(phi1, float); phi2 = np.asarray(phi2, float)
    phi1S = (phi1[1:] + phi1[:-1]) / 2; phi2S = (phi2[1:] + phi2[:-1]) / 2
    phiT = np.vstack([(phi1[1:] - phi1[:-1]) / h, (phi2[1:] - phi2[:-1]) / h])
    L = 2
    M = 2 + 2 * ((2 * bn + 1) ** 2 - 1)
    K = M // L

    p = _base_functions(phi1S, phi2S, K, bn)
    v1 = _base_derivatives(phi1S, phi2S, K, bn, 1)
    v2 = _base_derivatives(phi1S, phi2S, K, bn, 2)
    ppT = p @ p.T
    sv1 = v1.sum(axis=1); sv2 = v2.sum(axis=1)
    N = phiT.shape[1]

    C_old = Cpr.copy()
    Cpt = Cpr.copy()
    XIpt = None
    for _ in range(max_loops):
        # calculateE
        D = phiT - Cpt.T @ p
        E = (h / N) * (D @ D.T)
        # calculateC
        _check_conditioning(E)
        invr = np.linalg.inv(E)
        XIpt = np.empty((M, M))
        XIpt[:K, :K] = XIpr[:K, :K] + h * invr[0, 0] * ppT
        XIpt[:K, K:] = XIpr[:K, K:] + h * invr[0, 1] * ppT
        XIpt[K:, :K] = XIpr[K:, :K] + h * invr[1, 0] * ppT
        XIpt[K:, K:] = XIpr[K:, K:] + h * invr[1, 1] * ppT
        ED = _mldivide(E, phiT)
        r1 = XIpr[:K, :K] @ Cpr[:, 0] + XIpr[:K, K:] @ Cpr[:, 1] + h * (p @ ED[0] - 0.5 * sv1)
        r2 = XIpr[K:, :K] @ Cpr[:, 0] + XIpr[K:, K:] @ Cpr[:, 1] + h * (p @ ED[1] - 0.5 * sv2)
        C = _mldivide(XIpt, np.concatenate([r1, r2]))
        Cpt = np.column_stack([C[:K], C[K:]])
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.sum((C_old - Cpt) ** 2 / (Cpt * Cpt)) < eps:
                return Cpt, XIpt, E
        C_old = Cpt
    return Cpt, XIpt, E


def _propagate(Cpt, XIpt, p):
    """``Propagation_function_XIpt``: diffuse the posterior into the next prior."""
    invXIpt = np.linalg.inv(XIpt)
    inv_diff = np.diag(p * p * np.diag(invXIpt))
    return Cpt.copy(), np.linalg.inv(invXIpt + inv_diff)


def _matlab_colon(a, d, b):
    n = int(np.floor((b - a) / d + 1e-10))
    return a + d * np.arange(n + 1) if n >= 0 else np.zeros(0)


def _as_index(x, what):
    xi = int(round(x))
    if abs(x - xi) > 1e-9 * max(1.0, abs(x)):
        raise ValueError(f"{what} is {x:g} samples; MODA indexes with it, so it must "
                         "be a whole number of samples (adjust the window, overlap or fs)")
    return xi


def bayes_main(ph1, ph2, win, h, ovr, pr, bn):
    """``bayes_main.m``. Returns ``(tm, cc, e)``.

    ``win`` in seconds, ``h = 1/fs``, ``ovr`` the step as a fraction of the
    window, ``pr`` the propagation constant. ``cc`` is (windows x M), one row
    per window holding ``Cpt(:)`` (column-major, as MATLAB stores it); ``e``
    is (windows x 2 x 2).
    """
    ph1 = np.asarray(ph1, float).ravel(); ph2 = np.asarray(ph2, float).ravel()
    win_s = win
    win = win / h
    w = ovr * win
    pw = win * h * pr
    win_i = _as_index(win, "The window")
    w_i = _as_index(w, "The window step (overlap x window)")

    M = 2 + 2 * ((2 * bn + 1) ** 2 - 1)
    L = 2
    Cpr = np.zeros((M // L, L)); XIpr = np.zeros((M, M))
    if np.max(ph1) < 2 * np.pi + 0.1:
        ph1 = unwrap_matlab(ph1); ph2 = unwrap_matlab(ph2)

    n_win = int(np.floor((len(ph1) - win) / w)) + 1
    cc = np.zeros((n_win, M)); e = np.zeros((n_win, L, L))
    for i in range(n_win):
        s = i * w_i
        phi1 = ph1[s:s + win_i]; phi2 = ph2[s:s + win_i]
        Cpt, XIpt, E = bayes_phs(Cpr, XIpr, h, 500, 0.00001, phi1, phi2, bn)
        Cpr, XIpr = _propagate(Cpt, XIpt, pw)
        e[i] = E
        cc[i] = Cpt.ravel(order="F")
    tm = _matlab_colon(win / 2, w, len(ph1) - win / 2) * h
    return tm, cc, e


def dirc(c, bn):
    """``dirc.m``: (cpl1, cpl2, drc) for one row of ``cc``."""
    c = np.asarray(c, float).ravel()
    K = len(c) // 2
    cpl1 = float(np.linalg.norm(c[1:K]))
    cpl2 = float(np.linalg.norm(c[K + 1:2 * K]))
    return cpl1, cpl2, (cpl2 - cpl1) / (cpl1 + cpl2)


def cf_print(cc, bn):
    """``CFprint.m``: coupling functions q1, q2 on MODA's 0:0.13:2*pi grid.

    Returns ``(t1, t2, q1, q2)``; ``q[i, j]`` is at ``(t1[i], t2[j])``.
    """
    u = np.asarray(cc, float).ravel()
    K = len(u) // 2
    t1 = _matlab_colon(0.0, 0.13, 2 * np.pi); t2 = t1.copy()
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    q1 = np.zeros(T1.shape); q2 = np.zeros(T1.shape)
    br = 1
    for ii in range(1, bn + 1):
        q1 = q1 + u[br] * np.sin(ii * T1) + u[br + 1] * np.cos(ii * T1)
        q2 = q2 + u[K + br] * np.sin(ii * T2) + u[K + br + 1] * np.cos(ii * T2)
        br += 2
    for ii in range(1, bn + 1):
        q1 = q1 + u[br] * np.sin(ii * T2) + u[br + 1] * np.cos(ii * T2)
        q2 = q2 + u[K + br] * np.sin(ii * T1) + u[K + br + 1] * np.cos(ii * T1)
        br += 2
    for ii in range(1, bn + 1):
        for jj in range(1, bn + 1):
            q1 = q1 + u[br] * np.sin(ii * T1 + jj * T2) + u[br + 1] * np.cos(ii * T1 + jj * T2)
            q2 = q2 + u[K + br] * np.sin(ii * T1 + jj * T2) + u[K + br + 1] * np.cos(ii * T1 + jj * T2)
            br += 2
            q1 = q1 + u[br] * np.sin(ii * T1 - jj * T2) + u[br + 1] * np.cos(ii * T1 - jj * T2)
            q2 = q2 + u[K + br] * np.sin(ii * T1 - jj * T2) + u[K + br + 1] * np.cos(ii * T1 - jj * T2)
            br += 2
    return t1, t2, q1, q2


def surrogate_cpp(phase, n_surr, rng=None):
    """``surrcalc(phase, N, 'CPP', 0, fs)``: cyclic phase permutation.

    Cuts the wrapped phase into its cycles and shuffles the whole cycles
    (keeping the partial first and last ones in place). Returns an
    (n_surr x len(phase)) array of unwrapped phases.
    """
    rng = np.random.default_rng() if rng is None else rng
    signal = np.mod(np.asarray(phase, float).ravel(), 2 * np.pi)
    dc = np.flatnonzero(signal[1:] - signal[:-1] < -np.pi)   # 0-based: cut after dc
    NC = len(dc) - 1
    out = np.empty((n_surr, len(signal)))
    if NC > 0:
        cycles = [signal[dc[k] + 1:dc[k + 1] + 1] for k in range(NC)]
        st = signal[:dc[0] + 1]; en = signal[dc[NC] + 1:]
        for sn in range(n_surr):
            perm = rng.permutation(NC)
            out[sn] = unwrap_matlab(np.concatenate([st] + [cycles[k] for k in perm] + [en]))
    else:
        out[:] = unwrap_matlab(signal)
    return out


def full_bayesian(sig1, sig2, int1, int2, fs, win, pr, ovr, bn, ns, signif,
                  rng=None, progress=None):
    """``full_bayesian.m``: MODA's Bayesian coupling analysis of two signals.

    Band-passes each signal (Bayesian ``loop_butter``), takes the Hilbert
    phase, infers the coupling window by window (``bayes_main``) and tests
    it against ``ns`` cyclic-phase-permutation surrogates at ``signif`` %.

    Returns a dict with MODA's outputs ``tm, p1, p2, cpl1, cpl2, cf1, cf2,
    mcf1, mcf2, surr_cpl1, surr_cpl2`` plus ``drc`` (dirc's direction per
    window), ``order1``/``order2`` (the filter orders loop_butter chose) and
    ``cc``.
    """
    bands1, or1 = loop_butter_bayes(np.asarray(sig1, float).ravel(), int1, fs)
    phi1 = np.angle(hilbert(bands1))
    bands2, or2 = loop_butter_bayes(np.asarray(sig2, float).ravel(), int2, fs)
    phi2 = np.angle(hilbert(bands2))

    tm, cc, _ = bayes_main(phi1, phi2, win, 1 / fs, ovr, pr, bn)
    n_win = cc.shape[0]
    cpl1 = np.zeros(n_win); cpl2 = np.zeros(n_win); drc = np.zeros(n_win)
    cfs = [cf_print(cc[m], bn) for m in range(n_win)]
    for m in range(n_win):
        cpl1[m], cpl2[m], drc[m] = dirc(cc[m], bn)
    q21 = np.stack([c[2] for c in cfs], axis=2)
    q12 = np.stack([c[3] for c in cfs], axis=2)

    rng = np.random.default_rng() if rng is None else rng
    surr1 = surrogate_cpp(phi1, ns, rng)
    surr2 = surrogate_cpp(phi2, ns, rng)
    scpl1 = np.zeros((ns, n_win)); scpl2 = np.zeros((ns, n_win))
    for n in range(ns):
        _, cc_s, _ = bayes_main(surr1[n], surr2[n], win, 1 / fs, ovr, pr, bn)
        for idx in range(cc_s.shape[0]):
            scpl1[n, idx], scpl2[n, idx], _ = dirc(cc_s[idx], bn)
        if progress:
            progress(n + 1, ns)

    alph = 1 - signif / 100
    Kq = int(np.floor((ns + 1) * alph))
    if ns == 0:
        surr_cpl1 = surr_cpl2 = np.full(n_win, np.nan)
    elif Kq == 0:
        # MATLAB's max() of a 1-row matrix is the max of that row, a scalar
        surr_cpl1 = np.max(scpl1, axis=0) if ns > 1 else np.full(n_win, scpl1.max())
        surr_cpl2 = np.max(scpl2, axis=0) if ns > 1 else np.full(n_win, scpl2.max())
    else:
        surr_cpl1 = -np.sort(-scpl1, axis=0)[Kq - 1]
        surr_cpl2 = -np.sort(-scpl2, axis=0)[Kq - 1]

    return {"tm": tm, "p1": phi1, "p2": phi2, "cpl1": cpl1, "cpl2": cpl2,
            "drc": drc, "cf1": q21, "cf2": q12,
            "mcf1": q21.mean(axis=2), "mcf2": q12.mean(axis=2),
            "surr_cpl1": surr_cpl1, "surr_cpl2": surr_cpl2,
            "order1": or1, "order2": or2, "cc": cc}
