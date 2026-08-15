# Transform parity suite

Checks that the vectorised, blocked per-scale transform in `wt.m` / `wft.m`
returns the same numbers as the pre-vectorisation implementation it replaced,
and that the routines downstream of it are unaffected.

It compares MODA against its own history: the current implementation against
the one it replaced.

## Run

```bash
bash tests/transform_parity/run_parity.sh             # main + downstream + bispectrum + swaps
bash tests/transform_parity/run_parity.sh --blocks    # additionally force blk=1 / blk=7
```

Needs a local MATLAB (`MATLAB=/path/to/matlab` to override the default).

`make_baseline.sh` extracts the baseline from git history into `.baseline/`
(untracked). It does not pin a commit hash: it walks the history of `wt.m` and
takes the most recent commit whose copy predates the vectorised path, so it
keeps working in any clone that has the history. Set `BASELINE_COMMIT=<sha>` to
override. A shallow clone will not reach back far enough.

## What each part does

| Script | Covers |
|---|---|
| `build_cases.m` | The shared case matrix — 966 cases. Both sides run an identical list in an identical order. |
| `parity_run.m` | Transform-level parity. Optional 5th arg is a regexp over case ids, used to subset for the forced-block runs. |
| `diag_special.m` | Reports the non-default-path cases individually rather than as a total. |
| `downstream_run.m` | Ridge extraction (`ecurve` → `rectfr`) and coherence (`wphcoh`, `tlphcoh`). |
| `bispec_run.m` | `bispecWavNew`. |
| `swap_run.m` | Each rewritten routine against upstream's, one at a time, plus the cases below. |
| `bispec_memory_run.m` | Time and peak memory of the bispectrum on a long recording. |

The case matrix crosses all four built-in wavelets (`Lognorm`, `Morlet`,
`Bump`, `Morse-3`) and all six built-in windows (`Gaussian`, `Hann`,
`Blackman`, `Exp`, `Rect`, `Kaiser-3`) with two `f0` values, `Preprocess`
on/off, `CutEdges` on/off, three `Padding` modes, default and narrowed
frequency ranges, and both even and odd signal lengths — plus six
non-default-path cases (custom `fwt`-only, custom `twf`-only, a handle that
rejects matrix input, and a NaN-producing handle).

## Two things that will silently make results meaningless

**Comparison in one process.** `parity_run` and `downstream_run` expose the
baseline as `wt_base`/`wft_base` — renamed copies — so both implementations
are callable in one session without path games. Nothing large is written to
disk: each pair is compared immediately and only scalars are kept. An earlier
version saved every transform and produced a 5.4 GB file per run.

**The transform cache.** `bispec_run` cannot use renamed functions, because
the bispectrum chain calls `wt()` by name across several files — so it swaps
`addpath`/`rmpath` mid-session instead. `transformCached` keeps a `persistent`
cache keyed on signal and parameters only, with nothing identifying which
implementation produced an entry, so the second run gets a stale hit from the
first and the comparison silently becomes baseline-vs-itself. `bispec_run`
calls `clear transformCached` before each side. Anything else added here that
swaps implementations mid-session must do the same.

A useful tell: the baseline prints incremental `1%2%3%…100%` progress from the
serial loop, while the vectorised path prints one `100%` per block. If both
sides print identically, they probably ran the same code.

## Error metric

`max|X−Y| / max|X|` — the largest absolute difference normalised by the
transform's own magnitude. Not per-element `|X−Y|/|X|`, which explodes on the
near-zero coefficients a transform is full of and reports a huge error for a
numerically perfect result. Phase is compared on the unit circle
(`exp(1i*phi)`) so 2π wrapping is not counted as a difference.

Match threshold is 1e-12 at transform level, 1e-9 downstream.

## Each routine on its own: `swap_run.m`

`make_baseline.sh` also extracts upstream's `wphcoh`, `ecurve`,
`bispecWavNew`, `CFprint`, `dirc`, `bayes_main` and `full_bayesian` into
`.baseline/swap/`, renamed `<name>_base`. Only the primary function is
renamed, so whatever it calls by name is the current code and a difference
can only come from that one file. `swap_run.m` (step 4 of `run_parity.sh`,
about two minutes) runs each pair on one input and also covers:

| Check | Result on R2026a |
|---|---|
| `wphcoh`, CutEdges on and off | $1.4\times10^{-14}$ |
| `ecurve`, methods 1, 2 and `'max'` | identical |
| `bispecWavNew`, cross and auto | $5.9\times10^{-15}$ |
| `CFprint`, `dirc` (bn 1–3), `bayes_main` (bn 1–2), `full_bayesian` with surrogates | identical |
| numeric `'nv'` (wt) and `'fstep'` (wft) against upstream | $1.3\times10^{-15}$, identical |
| overflow retry path in `wt` and `wft` (a wavelet / window that overflows to NaN far out on its support; both sides print the warning) | $5\times10^{-15}$, identical |
| a wavelet given only in the time domain, against the built-in Morlet, batched and serial loop | $4.7\times10^{-8}$ (the numerical inversion of `twf`); batched vs loop $6\times10^{-16}$ |
| GPU branch against upstream's CPU loop | runs when a GPU is present; skipped otherwise |

## Long recordings: `bispec_memory_run.m`

Run once per side, each in a fresh MATLAB, on 20 minutes of a two-signal
recording at 40 Hz (48,000 samples, 186 bins):

```bash
matlab -batch "addpath('tests/transform_parity'); bispec_memory_run(pwd, 'tests/transform_parity/.baseline', 'new')"
matlab -batch "addpath('tests/transform_parity'); bispec_memory_run(pwd, 'tests/transform_parity/.baseline', 'base')"
```

| | Time | Peak memory above start |
|---|---|---|
| upstream (one `wtAtf2` per cell) | 605 s | 843 MB |
| batched, before blocking | 168 s | 1,357 MB |
| batched, blocked to the same budget as `wt.m` | 252 s | 991 MB |

The bispectra agree to $7\times10^{-16}$ with an identical NaN pattern. Peak
memory no longer grows with the number of frequencies in a row.

## An upstream bug this fixes

**Time-domain-only custom wavelets** (`{[],[],twf,[t1 t2]}`) failed with
`Unrecognized function or variable 'xx'` on upstream: the admissibility test
used integration points only the frequency-domain branch computes. It now
computes its own. Behind that error was a second bug nothing could reach:
the time-domain branch never applied the 1/scale factor, so each frequency
came out scaled by `ompeak/(2*pi*f)`. Fixed in `wt.m`, `wtAtf2.m` and
`wtAtf2_batch.m`. There is no upstream result to compare with, so
`swap_run.m` checks it against the same wavelet built in (Morlet).

## Not covered

- **The GPU branch** on this machine: no Parallel Computing Toolbox. The
  check is in `swap_run.m` and runs wherever a GPU is present.
- **`parfor` in `CoherenceMulti.m`**: surrogate thresholds come from worker
  random streams, so they are not comparable with a serial run by
  construction.
