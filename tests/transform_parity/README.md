# Transform parity suite

Checks that the vectorised, blocked per-scale transform in `wt.m` / `wft.m`
returns the same numbers as the pre-vectorisation implementation it replaced,
and that the routines downstream of it are unaffected.

Distinct from `tests/parity/`, which compares MODA against FastMODA. This one
compares MODA against its own history.

## Run

```bash
bash tests/transform_parity/run_parity.sh             # main + downstream + bispectrum
bash tests/transform_parity/run_parity.sh --blocks    # additionally force blk=1 / blk=7
```

Needs a local MATLAB (`MATLAB=/path/to/matlab` to override the default). The
baseline is extracted from git by `make_baseline.sh` into `.baseline/`
(untracked) — `BASELINE_COMMIT` defaults to `a58a257`, the commit immediately
before the vectorised path landed in `a9efd94`.

## What each part does

| Script | Covers |
|---|---|
| `build_cases.m` | The shared case matrix — 966 cases. Both sides run an identical list in an identical order. |
| `parity_run.m` | Transform-level parity. Optional 5th arg is a regexp over case ids, used to subset for the forced-block runs. |
| `diag_special.m` | Reports the non-default-path cases individually rather than as a total. |
| `downstream_run.m` | Ridge extraction (`ecurve` → `rectfr`) and coherence (`wphcoh`, `tlphcoh`). |
| `bispec_run.m` | `bispecWavNew`. |

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

## Not covered

- **The GPU branch.** Needs Parallel Computing Toolbox and a device. It is
  structurally guarded — any failure falls through to the CPU path and then to
  the serial loop — but it has never been numerically compared.
- **The `ouflag` retry path.** The NaN-handle cases fail earlier, in parameter
  estimation (`MATLAB:quadgk:invalidAbsTol`), identically on both sides.
  Covering it needs a handle that survives `quadgk` but still produces NaNs on
  the frequencies the transform samples.
