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
- `wt.m` / `wft.m` — the main per-scale transform loop; see below.

**Deliberately not vectorized:** IAAFT surrogate generation (inherently sequential
convergence), the main recursive filtering loop in `bayes_main.m`, and the
dynamic-programming ridge-tracing steps in `ecurve.m` (`pathopt`/`onestepopt`) — each
step in these algorithms depends on the previous step's result, so there's no batching
opportunity without changing the algorithm itself.

### `wt.m` / `wft.m` — the per-scale transform loop

This is the single most shared numerical routine in the app: every module depends on it
directly or via ridge extraction / coherence / bispectrum. Because of that blast radius
it was scoped as its own change and verified separately, to the bar set out below.

**What changed.** The signal's FFT (`fx`) and the padding that produced it are already
shared across all scales — only the kernel differs per scale. So the per-scale kernels
are built as columns of one `NL x nb` matrix, multiplied against the single `fx` column
by broadcasting, and inverse-transformed in one batched column-wise `ifft` instead of
`nb` separate calls.

Three things guard this:

- **Custom-handle probe.** Built-in wavelets/windows are elementwise-safe on matrix
  input, but a user-supplied `fwt`/`twf` handle need not be. The handle is probed on a
  2x2 input first; anything that errors or returns the wrong shape falls back to the
  original per-scale loop, which is retained verbatim. The whole vectorized path is
  additionally wrapped in `try`/`catch` onto that same fallback.
- **Bounded memory.** Building all `SN` scales at once would need several `NL x SN`
  arrays alive simultaneously — `O(NL*SN)` against the serial loop's `O(NL)`. Since
  `NL = 2^nextpow2(L + cone)`, a ~20-minute 40 Hz recording gives `NL = 131072`, so at a
  few hundred scales that is gigabytes, which would OOM into the fallback precisely on
  the large inputs the batching is meant to speed up. The scales are therefore processed
  in blocks sized to a fixed element budget (`maxElem`, ~64 MB per complex array), and
  intermediates reuse one buffer rather than naming each step. When `NL*SN` already fits
  the budget this is a single block — i.e. bit-identical to computing all scales at once.
- **Optional GPU offload** of the batched multiply + `ifft` above a size threshold, when
  Parallel Computing Toolbox and a device are present. Availability is cached in a
  `persistent` (probing it is slow), and any failure falls through the same `try`/`catch`.

**Verification.** Production `wt.m`/`wft.m` are compared against pre-vectorization
copies of both files extracted from `a58a257`, the commit immediately before the
vectorized path landed in `a9efd94`. The harness is committed at
`tests/transform_parity/` — run it with `bash tests/transform_parity/run_parity.sh
--blocks`. Results below are from the run of 2026-08-15:

| Run | Cases | Matched | Worst rel. error |
|---|---:|---:|---:|
| Transform parity, full case matrix | 966 | 966 | `4.49e-15` |
| Forced `blk=1` (subset) | 46 | 46 | `3.99e-15` |
| Forced `blk=7` (subset) | 46 | 46 | `3.99e-15` |
| Downstream: ridge extraction + coherence | 12 | 12 | `1.42e-14` |
| Bispectrum (`bispecWavNew`) | 1 | 1 | `9.82e-16` |

- The **case matrix** crosses all four built-in wavelets (`Lognorm`, `Morlet`, `Bump`,
  `Morse-3`) at two `f0` values each and all six built-in windows (`Gaussian`, `Hann`,
  `Blackman`, `Exp`, `Rect`, `Kaiser-3`), with `Preprocess` and `CutEdges` toggles,
  three `Padding` modes, default and narrowed `fmin`/`fmax`, and both even and odd
  signal lengths (`NL` parity changes how the time axis is built). NaN positions agreed
  exactly on every case.
- The **non-default code paths** are covered explicitly: a custom `fwt`-only cell
  wavelet (`9.13e-16`), `wft.m`'s custom `twf`-only branch (`2.80e-17`), and a handle
  that deliberately rejects matrix input. That last one returns **exactly zero**
  difference, which is the point — bit-identical output is only possible if the probe
  detected the bad handle and ran the retained serial loop.
- The **blocking loop** is exercised by forced `blk=1` / `blk=7` variants, since the
  short test signals otherwise yield a single block. Block size 7 also puts the final
  block on a non-divisible boundary. Subset rather than the full matrix because `blk=1`
  over 966 cases is impractically slow.
- **Downstream** covers ridge extraction (`ecurve` → `rectfr`, checking ridge support,
  instantaneous amplitude, phase, frequency and the reconstructed signal) and
  `wphcoh`/`tlphcoh` coherence, for three wavelets and three windows. For `wft` the
  results are bit-identical; the small residuals are all on the `wt` side, where the
  per-scale dilation and normalization factor reorder the arithmetic.

**Not covered:**

- **The GPU branch**, which needs Parallel Computing Toolbox and a device. It is
  structurally guarded (any failure falls through to the CPU path and then the serial
  loop), but it has never been numerically compared.
- **The `ouflag` retry path.** A handle producing NaNs on the support fails earlier, in
  parameter estimation (`MATLAB:quadgk:invalidAbsTol`), identically on both
  implementations — so it never reaches the retry. Covering it needs a handle that
  survives `quadgk` but still yields NaNs on the frequencies the transform samples.
  The behavioural note about `ouval` below is therefore **untested**.

One deliberate behavioural difference: on overflow/underflow the reported example
argument (`ouval`) is now the *first* offending one rather than an arbitrary one, so the
warning text doesn't depend on the block size. This affects the warning message only,
never the returned transform. Separately, the original serial loop indexed the
frequency array with an index into the (shorter) in-support kernel when building that
message, so the value it printed was generally the wrong one.

**Known pre-existing bug, unrelated to this change:** calling `wt.m` with a custom
`twf`-only wavelet (empty `fwt`) fails in the parameter estimation with
`Unrecognized function or variable 'xx'`. Confirmed to reproduce identically on both
the old and new code, so it is not a regression — but that documented input form does
not currently work. `wft.m`'s `twf`-only form is fine.

**A trap for anyone extending the parity suite.** `bispecWavNew` obtains its transforms
through `transformCached`, whose `persistent` cache is keyed on the signal and
parameters only — nothing identifies which `wt` implementation produced an entry. A
harness that swaps implementations by `addpath`/`rmpath` mid-session will therefore get
a stale hit on the second run and silently compare the baseline against itself, scoring
a perfect match while testing nothing. `bispec_run.m` calls `clear transformCached`
before each side. A quick tell that this has gone wrong: the serial loop prints
incremental `1%2%3%…100%` progress while the vectorized path prints one `100%` per
block, so if both sides print identically they ran the same code.

See also `docs/matlab-memory-optimizations.md` in the repository for related notes on
memory-focused (rather than loop-focused) MATLAB optimizations.
