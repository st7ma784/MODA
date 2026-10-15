"""FastMODA's ports of MODA's other modules, against MODA itself.

Compares ``fastmoda.legacy_bayes`` (Bayesian inference, coupling functions,
both ``loop_butter`` variants), the Hilbert transform and the biphase port
with reference output written by ``gen_moda_other.m`` from real MODA. Each
stage is fed MODA's own input for that stage, so a difference is that
stage's and not inherited from the one before.

    matlab -batch "gen_moda_other('<repo>', '<repo>/tests/parity/moda_diff/reference')"
    pytest tests/parity/moda_diff/test_moda_other.py -v

Same metric as test_moda_diff.py: max|X-Y| / max|X|.
"""
import os
import sys

import numpy as np
import pytest
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "FastMODA"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(REF, "other_bayes_bn1.mat")),
    reason="no MODA references; run gen_moda_other.m")

EXACT = 1e-8
# bayesPhs stops iterating once the relative change in the parameters is
# below 1e-5, so rounding differences come out at about this size.
BAYES = 1e-6
# MODA's loop_butter runs the Butterworth filter up to the order where it
# blows up. At the order it settles on, a one-ulp change to one coefficient
# moves MATLAB's own output by up to a few percent; gen_moda_other.m records
# that sensitivity for every case. Two implementations differ by many ulps
# over many coefficients and samples, so the port is held to a small
# multiple of MATLAB's own sensitivity, and never looser than EXACT.
ULP_FACTOR = 10


def filter_tol(sens):
    return max(EXACT, ULP_FACTOR * float(sens))


def _load(name):
    return loadmat(os.path.join(REF, name + ".mat"), squeeze_me=True)


def rel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    s = np.nanmax(np.abs(a))
    return float(np.nanmax(np.abs(a - b)) / (s if s else 1.0))


# ── Butterworth design and MODA's two loop_butters ──────────────────────────

def test_butter_coefficients():
    """MATLAB's own design route, orders 1 to 10."""
    from fastmoda.legacy_bayes import butter_matlab
    d = _load("other_butter")
    worst = 0.0
    for n, wn, bb, aa in zip(d["BN"], d["BW"], d["BB"], d["BA"]):
        b, a = butter_matlab(int(n), wn)
        k = 2 * int(n) + 1
        worst = max(worst, rel(bb[:k], b), rel(aa[:k], a))
    assert worst < 1e-12, worst


def test_filtfilt_where_well_conditioned():
    """filtfilt with MATLAB's exact coefficients, at the orders where the
    filter is not yet near the edge of stability. Above them the outputs part
    by about 100x per order, in MATLAB as much as here (see ULP_FACTOR)."""
    from scipy.signal import filtfilt
    d = _load("other_filtfilt")
    for i, n in enumerate(d["FN"]):
        n = int(n)
        if n > 3:
            continue
        k = 2 * n + 1
        b = d["FB"][i][:k]; a = d["FA"][i][:k]
        for x, Y in ((d["xS"], d["YS"]), (d["xL"], d["YL"])):
            assert rel(Y[:, i], filtfilt(b, a, x, padlen=3 * (k - 1))) < 1e-8, (n, d["FW"][i])


@pytest.mark.parametrize("variant", ["filtering", "bayes"])
def test_loop_butter(variant):
    from fastmoda.legacy_bayes import loop_butter_filtering, loop_butter_bayes
    d = _load("other_loopbutter")
    fn = loop_butter_filtering if variant == "filtering" else loop_butter_bayes
    out, orders, sens = ((d["outF"], d["orF"], d["sensF"]) if variant == "filtering"
                         else (d["outB"], d["orB"], d["sensB"]))
    for ib, band in enumerate(np.atleast_2d(d["bandsF"])):
        y, order = fn(d["xF"], band, float(d["fsF"]))
        assert order == int(np.atleast_1d(orders)[ib]), (band, order, orders)
        tol = filter_tol(np.atleast_1d(sens)[ib])
        assert rel(out[:, ib], y) < tol, (band, rel(out[:, ib], y), tol)


def test_hilbert():
    from scipy.signal import hilbert
    d = _load("other_hilbert")
    assert rel(d["H1"], hilbert(d["xF"])) < 1e-12
    assert rel(d["H2"], hilbert(d["xH"])) < 1e-12


# ── Bayesian chain ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("bn", [1, 2])
def test_bayes_filter_and_phase(bn):
    from fastmoda.legacy_bayes import loop_butter_bayes
    from scipy.signal import hilbert
    d = _load(f"other_bayes_bn{bn}")
    fs = float(d["fs"])
    b1, o1 = loop_butter_bayes(d["x1"], d["int1"], fs)
    b2, o2 = loop_butter_bayes(d["x2"], d["int2"], fs)
    # Hilbert phase of MODA's own filtered band
    ph = np.angle(hilbert(d["bands1"]))
    assert rel(np.exp(1j * d["phi1"]), np.exp(1j * ph)) < 1e-12
    # The order is decided by where the filter blows up, which rounding
    # moves: one order either way is the most that can be asked.
    assert abs(o1 - int(d["or1"])) <= 1 and abs(o2 - int(d["or2"])) <= 1
    for o, mo, b, mb, sens in ((o1, int(d["or1"]), b1, d["bands1"], d["sens1"]),
                               (o2, int(d["or2"]), b2, d["bands2"], d["sens2"])):
        if o == mo:
            assert rel(mb, b) < filter_tol(sens)


@pytest.mark.parametrize("bn", [1, 2])
def test_bayes_main(bn):
    """bayes_main on MODA's own phases."""
    from fastmoda.legacy_bayes import bayes_main
    d = _load(f"other_bayes_bn{bn}")
    tm, cc, e = bayes_main(d["phi1"], d["phi2"], float(d["win"]), 1 / float(d["fs"]),
                           float(d["ovr"]), float(d["pr"]), bn)
    assert rel(d["tm"], tm) < 1e-12
    assert rel(d["cc"], cc) < BAYES, rel(d["cc"], cc)
    assert rel(d["e"], e) < BAYES


@pytest.mark.parametrize("bn", [1, 2])
def test_dirc_and_cfprint(bn):
    """dirc and CFprint on MODA's own coefficients."""
    from fastmoda.legacy_bayes import dirc, cf_print
    d = _load(f"other_bayes_bn{bn}")
    cc = np.atleast_2d(d["cc"])
    got = np.array([dirc(row, bn) for row in cc])
    assert rel(d["cpl1"], got[:, 0]) < 1e-12
    assert rel(d["cpl2"], got[:, 1]) < 1e-12
    assert rel(d["drc"], got[:, 2]) < 1e-12
    q21 = np.stack([cf_print(row, bn)[2] for row in cc], axis=2)
    q12 = np.stack([cf_print(row, bn)[3] for row in cc], axis=2)
    assert rel(d["q21"], q21) < 1e-12 and rel(d["q12"], q12) < 1e-12


@pytest.mark.parametrize("bn", [1, 2])
def test_full_bayesian_end_to_end(bn):
    """The whole chain from the raw signals, as full_bayesian.m runs it.

    Matches MODA wherever loop_butter settles on MODA's filter orders. When
    rounding puts it one order away (see legacy_bayes' docstring), the phases
    and so the couplings differ; the test reports that as an expected
    failure rather than hiding it.
    """
    from fastmoda.legacy_bayes import full_bayesian
    d = _load(f"other_bayes_bn{bn}")
    r = full_bayesian(d["x1"], d["x2"], d["int1"], d["int2"], float(d["fs"]),
                      float(d["win"]), float(d["pr"]), float(d["ovr"]), bn, 0, 95)
    assert rel(d["fb_tm"], r["tm"]) < 1e-12
    orders = (r["order1"], r["order2"]); moda = (int(d["or1"]), int(d["or2"]))
    if orders != moda:
        pytest.xfail(f"loop_butter chose filter orders {orders}, MODA {moda}: the "
                     "order sits where the Butterworth filter goes unstable, which "
                     "rounding moves (cpl1 differs by "
                     f"{rel(d['fb_cpl1'], r['cpl1']):.2g})")
    tol = filter_tol(max(d["sens1"], d["sens2"]))
    for k in ("cpl1", "cpl2", "mcf1", "mcf2"):
        assert rel(d["fb_" + k], r[k]) < tol, (k, rel(d["fb_" + k], r[k]), tol)


def test_cpp_surrogate_permutes_whole_cycles():
    """CPP is random, so check its defining property: the surrogate is the
    original's cycles in another order, with the first and last in place."""
    from fastmoda.legacy_bayes import surrogate_cpp
    d = _load("other_bayes_bn1")
    ph = d["phi1"]
    s = surrogate_cpp(ph, 3, np.random.default_rng(0))
    w = np.mod(ph, 2 * np.pi)
    cuts = np.flatnonzero(np.diff(w) < -np.pi)

    def same_angle(a, b):
        return np.allclose(np.exp(1j * a), np.exp(1j * b), atol=1e-9)

    for row in s:
        assert row.shape == ph.shape
        assert np.all(np.abs(np.diff(row)) < np.pi)               # unwrapped
        assert same_angle(row[:cuts[0] + 1], w[:cuts[0] + 1])      # first partial cycle kept
        assert same_angle(row[cuts[-1] + 1:], w[cuts[-1] + 1:])    # and the last
        assert np.allclose(np.sort(np.cos(row)), np.sort(np.cos(w)), atol=1e-9)  # same values
        assert not same_angle(row, w)                              # but reordered


# ── biphase ─────────────────────────────────────────────────────────────────

def test_biphase():
    from fastmoda.legacy_moda import biphase_wav_legacy
    d = _load("other_biphase")
    fs = float(d["fsP"]); f1, f2 = d["fpair"]
    kw = dict(fmin=0.3, fmax=8, f0=1.0)
    amp, ph = biphase_wav_legacy(d["s1"], d["s2"], fs, f1, f2, **kw)
    assert rel(d["biamp"], amp) < EXACT and rel(d["biphase"], ph) < EXACT
    amp, ph = biphase_wav_legacy(d["s1"], d["s1"], fs, f1, f2, opt_from=d["s2"], **kw)
    assert rel(d["biampA"], amp) < EXACT and rel(d["biphaseA"], ph) < EXACT
