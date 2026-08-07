# Refactor Notes

Background for developers working on MODA's MATLAB engine or app shell. This
summarizes two ongoing/completed efforts documented in more detail elsewhere in the
repo:

## App Designer migration

MODA's desktop app was migrated from GUIDE (the legacy MATLAB GUI builder, last used
in v2.5, 2017) to App Designer. `MODAApp.m` is now the single shell window
(`uifigure`-based) hosting all five analysis modules as tabs, each of which can also
run standalone (embedded-vs-owns-figure pattern via `RootContainer`/`OwnsFigure`
properties on each module class).

See `docs/REFACTOR_GUIDE.md` in the repository for the full file-by-file migration
notes, common pitfalls (e.g. `uigridlayout` row/column sizing, `Layout.Row` only being
valid when the direct parent is a grid layout, `uibuttongroup` `Title` consuming space
inside its own declared height rather than adding to it), and troubleshooting
checklist.

## Calculation engine vectorization

Several of MODA's core numerical routines had `for`-loops replaced with batched/matrix
operations, following a consistent verification methodology:

1. Save the original function to a scratch copy under a renamed function/file.
2. Write the vectorized version in the production file.
3. Run both versions in real MATLAB (`matlab -batch`) on the same inputs and compare
   outputs directly (`max(abs(a-b))`, and NaN-position matching where relevant) —
   accepting only exact or floating-point-epsilon differences.

Routines vectorized so far:

- `bispecWavNew.m` / `bispecWavPython.m` — the O(n_freq²) nested loop's redundant
  per-`(j,k)` signal preprocessing was batched via new `wtAtf2_batch.m` /
  `wtAtfMod_batch.m` helpers, which run the signal FFT/padding/preprocessing once per
  row instead of once per `(j,k)` pair. See
  [Wavelet Bispectrum](../algorithms/wavelet-bispectrum.md).
- `CFprint.m`, `dirc.m`, `bayes_main.m` (diagonal-matrix construction) — Bayesian
  inference support routines.
- `ecurve.m` — ridge amplitude/index extraction (`max` over columns rather than a
  per-column loop).
- `wphcoh.m` — wavelet phase coherence's per-row averaging loop.
- `MODAread.m` — extended (not a performance change) to support multi-file and CSV
  import.

**Deliberately not vectorized:** IAAFT surrogate generation (inherently sequential
convergence), the main recursive filtering loop in `bayes_main.m`, and the
dynamic-programming ridge-tracing steps in `ecurve.m` (`pathopt`/`onestepopt`) — each
step in these algorithms depends on the previous step's result, so there's no batching
opportunity without changing the algorithm itself.

**Planned but not yet done:** vectorizing `wt.m`/`wft.m`'s main per-scale transform
loop — the single most shared numerical routine in the app (every module depends on it
directly or via ridge extraction / coherence / bispectrum). This is scoped as its own
change, to be verified separately across every built-in wavelet/window type and the
full downstream pipeline before landing, given the blast radius of a regression here.

See also `docs/matlab-memory-optimizations.md` in the repository for related notes on
memory-focused (rather than loop-focused) MATLAB optimizations.
