# CWT feedback — what's been found and what changed

 Short version:

- Your `f0` → `nv` → `n_freqs` reverse-engineering was **correct**, and FastMODA
  already implements exactly that rule — but only on a code path that wasn't
  documented, so you had no way to find it.
- Predictive padding is implemented too, on the same path.
- The raw CWT matrix genuinely wasn't available. It is now, and the quantity you
  were actually after (`total_power`) now comes back directly, so you doesn't need
  the matrix at all.
- The plot round-trip is a large chunk of the error - see §5.

---

## 1. What you were aiming for vs. what the request did

One of the big challenges is that your very used to MATLAB, and I'm not. I've had another look at the UI, and the Matlab side was very different in what it called things to what I'd thought were sensible names. 

```matlab
opts.fmin = 0.01;  opts.fmax = 2;  opts.f0 = 2;
opts.Wavelet = 'Morlet';  opts.Preprocess = 'on';  opts.CutEdges = 'on';
```

The FastMODA API has no equivalent of `f0`, and you actually manages to pick a 
really sensible `n_cycles` to make the numbers line up.  The parameters
you found land on FastMODA's **fast** transform, which is a different algorithm
from `wt.m` in MODA, not a configurable version of it.

| What you set | What you meant (MODA) | What `/analyze_cwt` did with it |
| --- | --- | --- |
| `n_cycles: 12.6` | `f0 = 2` (and 12.6 ≈ 2π·2, so this was the right conversion) | resolution parameter of the fast Morlet filter bank — related to `f0`, but not MODA's `q = 2πf₀` |
| `n_freqs: 490` | the 490 bins MODA printed | `logspace(0.01, 2, 490)` — right count, but anchored on `fmin`, not MODA's `2^(k/nv)` lattice |
| `padding: 'symmetric'` | (MODA was using predictive) | symmetric reflection |
| `cut_edges: 'true'` | `CutEdges = 'on'` | ✅ same intent |
| — | `Preprocess = 'on'` | **not applied** — the fast path does no detrend/band-pass |

This is super cool, as a user sample, as it's how an experienced user behaves, rather than
how me as a developer, assumes you might use the tool. I've done my best to add signposting.
 `/analyze_cwt` had no entry in the API reference, which we talked about, but leads to exactly
 this kind of breakdown and in retrospect, is why the parameter set had to be guessed in the first place.

### The hidden setting.

Because CWT's implementation actually moved substantively away from what I found
in the MATLAB, after fixing a load of memory issues etc... There is a `legacy=true`
 flag on `/analyze_cwt` that maps back to a `fastmoda.legacy_moda.wt_legacy`. Its
 a more faithful port of `wt.m` with the exact Lognorm / Morlet / Bump forms  I'd 
 used for testing, (Morlet including the admissibility correction term), MODA's
log-voice lattice, cubic detrend + band-pass preprocessing, predictive padding
and the cone of influence. On that path you pass `f0` and nothing else;
`n_cycles` and `n_freqs` are ignored, because `f0` determines them.

If you needed direct compatibility with previous work done on the MATLAB 
implementation, then that gives a better, MODA-comparable, value.

## 2. Frequency resolution

From three cases, our `f0` → `nv` code and compared against the reported 
numbers in the MATLAB console output:

| `f0` | MODA printed | FastMODA computes | `nv` | `n_freqs` (0.01–2 Hz) |
| --- | --- | --- | --- | --- |
| 1 | 30.85 | 30.8488 | 31 | 237 |
| 2 | 63.89 | 63.8850 | 64 | 490 |
| 3 | 96.40 | 96.4019 | 97 | 742 |

Exact agreement, bin counts included — so `Nf = nv·No + 1` and the linear
`nv ∝ f0`. 

The "additional rounding code to avoid overlapping wavelets" you suspected is
just a **ceiling**: `nv = ceil(nv_real)`. At `f0 = 2` that takes 63.885 → 64, and
at `f0 = 1` it takes 30.849 → 31, which is why you saw 237 bins rather than the
236 that rounding-to-nearest would give.

For completeness, `nv_real` comes from the wavelet's own 50% frequency support
(MODA's `sqeps`), so it is wavelet-dependent — Lognorm gives 32.28 / 64.57 /
96.85 for the same three `f0` values, not the same proportionality constant.

The lattice difference turned out to be insignificant: MODA's
`2^(k/nv)` grid and your `logspace(0.01, 2, 490)` differ by at most 0.22% per bin
(~0.11% avg), because with 490 bins over the same span they are pretty much the 
same, I'm pretty happy it's not the main source of any errors
---

## 3. Padding

MODA's predictive padding (`fcast.m`) is implemented on the legacy path and is
its default there. FASTMOda extrapolates using a small set of in-band
sinusoids fitted from the periodogram, rather than reproducing `fcast.m`
line-by-line. With `cut_edges = true` the padded region is discarded anyway, so
it should not move the numbers you are computing, but it is a known fidelity
limit rather than a solved problem.

The fast path's `symmetric` / `zero` / `periodic` options are genuinely different
algorithms, and it also pads to ±½-support rather than to the next power of two.

---

## 4. Raw CWT matrix

Two things changed.

`/analyze_cwt` results now include, in raw units with no dB-scaling anywhere:

- `time_avg_power` — one value per frequency bin, computed as
  `mean(abs(WT).^2, 2, 'omitnan')`, i.e. literally the MATLAB
- `total_power` — `sum(time_avg_power, 'omitnan')`
- `freqs`, `nv`, `n_freq_bins`, `n_times`, and `f0` on the legacy path

For your total-power comparison that should be the whole job...

**The full matrix is downloadable when you want it.** Pass `return_matrix=true`
and the response carries a `cwt_matrix_url`; `GET` it for an `.npz` containing:

| Key | Shape | Notes |
| --- | --- | --- |
| `cwt` | `(n_freq, n_time)` complex64 | NaN outside the cone of influence when `cut_edges=true` |
| `freqs` | `(n_freq,)` | Hz |
| `times` | `(n_time,)` | seconds |

Complex, full time resolution, NaNs preserved. It is roughly 10 MB per 5-minute
16 Hz recording, and files expire after an hour, so fetch it in the same script
run that started the job.

---

## 5. Where the 48% probably came from

Your suspicion that the plot round-trip was the main cause was right, and the 
mechanism is narrower than "dB conversion is lossy".

Reconstructing your approach on a 10-minute 16 Hz test signal:

| Method | `total_power` | vs. truth |
| --- | --- | --- |
| From the full matrix | 4.19755 | — |
| From the plot, **NaNs preserved** | 4.19757 | +0.0% |
| From the plot, **NaNs read as a finite floor** | 3.61477 | **−13.9%** |

So, although the dB conversion itself is fine, and even the plot's time downsampling
(9600 columns → 506) is fine, because it was being averaged anyway. The
cone-of-influence mask was the problem. With `CutEdges` on, the low-frequency rows are
partly or entirely NaN; those NaNs become `null` in the plot's JSON, and if they
come back as 0 (or as the plot's `1e-12` floor) those cells get counted as real
zero-power samples instead of being skipped the way `omitnan` skips them.

That I think also accounts for the spread up to 48%.  How much of the low-frequency end the COI
eats depends on recording length relative to `fmin`, so short recordings are hit
hard and long ones barely at all, though this would be an interesting test to see if I
got this diagnosis correct...

Fwiw, I also compared your exact fast-path settings against the
legacy path directly, and they agreed to 0.2% on total power, so your
`n_cycles` tuning was really good! That is consistent with your ~2% average, and points
away from the parameters being the problem.

---

## 6. `mean_ridge_freq`

Removed from `/analyze_cwt`, you were right about what it was
doing, and it was not adding anything `dominant_freq` did not. 

Classic case of "I imagine it's useful" being wrong. I might have to stop 
trying to be helpful...

---

## 7. Suggested config

```python
params = {
    'fs':          16,
    'wavelet':     'Morlet',
    'freq_min':    0.01,
    'freq_max':    2,
    'legacy':      'true',    # the MODA-faithful path
    'f0':          2,         # replaces n_cycles / n_freqs entirely
    'cut_edges':   'true',
    # padding defaults to 'predictive' on the legacy path — no need to set it
}
# then, from the /status/<task_id> results:
#   results['total_power']     -> compare directly against MODA_Powers.csv
#   results['time_avg_power']  -> the per-bin spectrum
#   results['nv'], results['n_freq_bins']  -> should read 64 and 490
```

If `nv` does not come back as 64 and `n_freq_bins` as 490, something has not
taken effect — worth telling us before you run the whole cohort rather than
after.

No need to touch `n_cycles` or `n_freqs` at all; if they are still in your
`parameter_dict` they will simply be ignored on this path.

---

## 8. What else changed

- `/analyze_cwt` is now in the
  [REST API reference](../api-and-ml/rest-api-reference.md#post-analyze_cwt),
  including the `f0` → `nv` → `n_freqs` rule and the table from §2.
- `padding` now defaults to `predictive` when `legacy=true`. It was defaulting to
  `symmetric` regardless, which quietly diverged from MODA.
- The web UI's TFA page gained an "Export coefficients (.npz)" checkbox, and
  shows `total_power`

## 9. Other considerations

- I didn't add a `f0`-only mode to the *fast* implementation. I figured it made sense for people with those labels in mind to be nudged to `legacy=true`.
- The `fcast` predictive-padding approximation in §3 isn't wrong, just different? Feels overkill to change it.
- The 0.005–2 Hz spectrum you mention testing later will push further into the
  COI, so expect the NaN handling in §5 to matter. With the `total_power` field, it should be fine, anything reconstructed from a plot will not be.

Just going to wrap up by saying thanks again. It's super cool to get insights from a real user and make the endpoint more usable than it was, especially to legacy users. As a dev coming in cold, I can only really guess!