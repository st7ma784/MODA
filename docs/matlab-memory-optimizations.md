# MATLAB memory-allocation fixes (Tier 1 & 2)

These changes target memory-scaling problems found in the original MATLAB
algorithms. All changes are allocation-pattern fixes only
— no algorithm, formula, or output value was changed. Loop bounds and
results are identical before and after; only *how* the result arrays
acquire their memory changed.

## Tier 1 — preallocation instead of grow-in-loop

MATLAB arrays grown via `arr(end+1) = ...` or `arr(:,:,k) = ...` past the
current bound force a full reallocation + copy of the array on every
iteration. Each of the following had a loop with a bound known in advance,
so the array is now preallocated with `zeros(...)` before the loop.

- **`allguis/guis/bispectrum/Functions/bispectrum_analysis.m`** — `surrxxx`,
  `surrppp`, `surrxpp`, `surrpxx` were started as `[]` and grown one
  `ns`-sized slice at a time. Now preallocated as
  `zeros(size(bispxxx,1), size(bispxxx,2), ns)` (grid size is already known
  from the non-surrogate bispectrum computed earlier in the same function).
  Fixed in both the `preprocess`/no-`preprocess` branches.

- **`allguis/guis/bayesian/Functions/bayes_main.m`** — `cc` and `e` were
  grown via `cc(i+1,:) = ...` / `e(i+1,:,:) = ...` inside the per-window
  loop. The window count (`numWin = floor((length(ps)-win)/w)+1`) is
  computable up front, and `e`'s per-window shape is always `L`-by-`L`
  (per `bayesPhs.m`), so both are now preallocated before the loop.

- **`allguis/guis/bayesian/Functions/full_bayesian.m`** — `cpl1`, `cpl2`,
  `q21`, `q12` were grown via `(m)` / `(:,:,m)` indexing inside
  `for m=1:size(cc,1)`. `CFprint.m` always returns a fixed
  `numel(0:0.13:2*pi)`-square grid regardless of input, so `q21`/`q12` are
  now preallocated as `zeros(ng,ng,numWin)` and `cpl1`/`cpl2` as
  `zeros(1,numWin)`.

- **`allguis/guis/tfa/Functions/wt.m`** and **`wft.m`** — `WT`/`WFT` were
  built via `zeros(SN,L)*NaN`, which allocates a full zero matrix and then
  multiplies it elementwise by NaN (two full-size allocations for one
  result, since `0*NaN == NaN` everywhere). Replaced with `NaN(SN,L)` /
  `NaN(SN,L)` — one allocation, identical result.

  These two files later gained a vectorized per-scale transform, which
  introduced a memory concern of its own: batching every scale at once needs
  several `NL x SN` arrays alive simultaneously, i.e. `O(NL*SN)` against the
  original loop's `O(NL)`. That is bounded by processing the scales in blocks
  sized to a fixed element budget. See the `wt.m` / `wft.m` section of
  `docs/developer-guide/refactor-notes.md`.

## Tier 2 — stop retaining data that's already dead

- **`allguis/guis/bayesian/Functions/full_bayesian.m`** — the surrogate
  loop (`for n=1:ns`) stored the *entire* per-window coupling-coefficient
  matrix for every surrogate in a growing cell array, `cc_surr{n}`. That
  cell array was never returned by the function and never read outside the
  loop body that immediately follows each assignment — only
  `scpl1(n,idx)`/`scpl2(n,idx)`, extracted via `dirc()` in the same
  iteration, survive past the loop. `cc_surr` has been removed; each
  iteration now uses a single reused local (`cc_s`) that's discarded before
  the next surrogate is processed. `scpl1`/`scpl2` are now preallocated as
  `zeros(ns, numWin)` for the same reason as Tier 1.
  Net effect: removes an `ns`-times-larger-than-necessary allocation
  (previously: `ns` full coupling matrices held simultaneously; now: one
  at a time) with zero change to `surr_cpl1`/`surr_cpl2`, the only outputs
  that depend on this loop.

## What this does NOT fix

These were intentionally left out of scope because removing them changes
the function's *interface* (what it returns) rather than just how memory
for the existing interface is acquired:

- `bispectrum_analysis.m` returns the full `surrxxx`/`surrppp`/`surrxpp`/
  `surrpxx` 3-D arrays (all `ns` surrogate bispectra) to its caller, which
  presumably uses them to compute a significance threshold for plotting.
  Preallocating fixed the reallocation churn, but the `ns × nfreq²` memory
  footprint itself is inherent to the current function signature — shrinking
  it would mean changing what the function returns (e.g. returning only a
  percentile threshold), which is a bigger, riskier change than requested
  here.
- `surrogate.m`'s `zeros(N,L)` ensemble preallocation was already correct
  (it preallocates, just at a large size) — left untouched.
