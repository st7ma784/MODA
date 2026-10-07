# MODA ↔ FastMODA parity suite

Two lightweight, Docker-only test suites. No local Python or MATLAB required —
everything runs inside the FastMODA CPU image.

## Build the image once

```bash
docker build -t fastmoda:cpu -f FastMODA/Dockerfile --target base FastMODA
```

## Run

```bash
bash tests/parity/run_parity.sh
# → 171 passed, 12 skipped, 57 xfailed   (with MODA references generated)
```

Three test modules: `test_ui_parity.py` (Task A), `test_numeric_equivalence.py`
(Task B), and `test_legacy_transforms.py` — which pins the MODA-faithful
`fastmoda.legacy_moda.wt_legacy` port and quantifies how far the fast path
diverges from it. The algorithmic differences behind that port are documented in
`docs/validation/algorithmic-differences.md`.

`moda_diff/` is the direct diff against **real MODA output**: `gen_moda_diff.m`
(local MATLAB) writes the references to `moda_diff/reference/` (gitignored), and
`test_moda_diff.py` runs FastMODA on the same inputs and asserts
`max|X−Y|/max|X| < 1e-8`. Known gaps (predictive padding, compact WFT windows,
default `fmin`, ridge, bispectrum) are strict `xfail`s, so closing one shows up
as a failure to un-mark. Without references the module skips. Print the full
per-case table with `python tests/parity/moda_diff/test_moda_diff.py`; results
are written up in `docs/validation/changelog-vs-moda.md`.

## Task A — UI feature parity  (`test_ui_parity.py`)

Asserts every user-facing MODA desktop capability has a FastMODA equivalent.
The canonical map lives in `moda_inventory.py` (one row per capability, not per
button). Each row is checked two ways:

1. **Route parity** — the mapped FastMODA HTTP route is registered in
   `FastMODA/app.py` (parsed statically).
2. **Backing-symbol parity** — the mapped `fastmoda.<module>:<function>` imports.

Result: **22 / 22 active capabilities covered**, plus 3 *intentional*
MODA-desktop-only gaps (native save dialog, session save/load, PDF report) that
have no meaningful web equivalent. FastMODA additionally exposes capabilities
MODA's GUI doesn't (MODWT, ML feature extraction, condition classification).

Run `pytest ... -s test_ui_parity.py::test_coverage_report` to print the full
matrix.

## Task B — numerical equivalence  (`test_numeric_equivalence.py`)

"Bit-identical to MODA" is not achievable across MATLAB and NumPy (different FFT
libraries, edge handling, float ordering), so correctness is validated two ways:

* **Ground-truth (always runs):** synthetic signals with analytically known
  answers pin FastMODA to the same mathematics MODA implements — a 10 Hz tone
  peaks at 10 Hz (sliding-FFT, WFT, CWT), Hilbert recovers instantaneous
  frequency, a band-pass rejects out-of-band energy, identical signals are
  perfectly phase-coherent, a quadratic triad lights up the bispectrum at the
  interacting pair, and MODWT reconstructs to < 1e-6.

* **Direct MODA diff (opt-in):** if `tests/parity/reference/moda_*.mat` exist,
  each is correlated against the FastMODA result (tol r > 0.9). Generate them in
  MATLAB with `gen_moda_reference.m`; without them these cases are skipped, so
  the suite is green on a MATLAB-less machine but becomes a true
  cross-implementation diff wherever MATLAB is available.
