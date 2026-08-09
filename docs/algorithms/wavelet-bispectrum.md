# Wavelet Bispectrum

## What it computes

The bispectrum is a higher-order spectral measure that detects **quadratic phase
coupling**: whether two frequency components f₁ and f₂ in a signal (or pair of
signals) combine nonlinearly to produce energy at their sum frequency f₃ = f₁ + f₂, in
a way that's phase-locked rather than coincidental. A strong bispectral peak at
(f₁, f₂) is evidence of a genuine nonlinear interaction between those two oscillatory
components — something an ordinary (linear) power spectrum cannot detect.

MODA computes the bispectrum from wavelet transforms rather than Fourier transforms,
so — like [time-frequency analysis](time-frequency-analysis.md) — it can track how
this coupling changes over time, and works well on signals with time-varying
frequencies (unlike classical FFT-based bispectrum estimators).

## The mathematics

### Definition

The wavelet bispectrum is the time-average of a triple product of wavelet
coefficients — two at the interacting frequencies, one conjugated at their sum:

$$
B(f_1,f_2) \;=\; \left\langle\, W_1(f_1,t)\; W_2(f_2,t)\; W^{*}(f_1+f_2,\,t) \,\right\rangle_t
$$

This is literally the code in `bispecWavNew.m`:
`nanmean(wt1(j,:) .* wt2(k,:) .* conj(WTdat_at_f3), 2)`.

### Why the triple product detects nonlinearity

Write each coefficient in polar form. The product's phase is

$$
\arg B = \phi_1(t) + \phi_2(t) - \phi_3(t)
$$

This combination — the **biphase** — is the crux. If the component at $f_3$ arises from
a genuine quadratic interaction (a term like $x^2$ mixing $f_1$ and $f_2$), then its
phase is *determined* by the other two, the biphase stays constant, and the
time-average accumulates coherently into a large $|B|$.

If instead all three components merely happen to be present but are generated
independently, the biphase drifts uniformly, the complex values cancel, and
$|B| \to 0$ — by exactly the same averaging geometry as
[phase coherence](wavelet-phase-coherence.md).

This is why a power spectrum cannot substitute: it discards phase entirely, so three
independent peaks and three nonlinearly-locked peaks look identical to it.

### Biamplitude and normalisation

The magnitude $|B|$ scales with the amplitudes of all three components, so a large
value can reflect strong oscillations rather than strong coupling. Normalising by the
biamplitude $\langle |W_1 W_2 W_3| \rangle$ gives **bicoherence**, bounded in $[0,1]$,
which isolates phase-locking from sheer power — the same amplitude-vs-phase separation
that makes coherence amplitude-free.

### Bispectrum types

With two input signals there are four ways to assign them to the three slots, which
MODA labels by which signal fills each position:

| Type | $W_1(f_1)$ | $W_2(f_2)$ | $W^*(f_3)$ | Detects |
|---|---|---|---|---|
| `111` | sig 1 | sig 1 | sig 1 | self-coupling within signal 1 (auto-bispectrum) |
| `222` | sig 2 | sig 2 | sig 2 | self-coupling within signal 2 |
| `112` | sig 1 | sig 1 | sig 2 | two components of signal 1 driving signal 2 |
| `122` | sig 1 | sig 2 | sig 2 | signal 1 and 2 interacting, appearing in signal 2 |

### The non-grid frequency problem

$f_3 = f_1 + f_2$ is a *sum* of frequencies, but the wavelet transform is computed on a
**logarithmic** grid — so $f_1 + f_2$ almost never lands on a grid point. The transform
must therefore be evaluated at arbitrary frequencies, which is what
`wtAtf2.m` exists for. Only pairs whose sum falls inside the analysed range are
meaningful; the rest are masked out.

## Source files

- `allguis/guis/bispectrum/Functions/bispecWavNew.m` — main bispectrum computation
  (auto- and cross-bispectrum).
- `allguis/guis/bispectrum/Functions/wtAtf2.m` / `wtAtf2_batch.m` — evaluates the
  wavelet transform at an arbitrary (non-grid) frequency; `_batch` evaluates it at many
  frequencies at once, sharing the signal's FFT/preprocessing across them.
- `allguis/guis/bispectrum/Functions/python/bispecWavPython.m` — Python-packaging
  variant of the same algorithm.
- `allguis/guis/bispectrum/Bispectrum.m` — the app module.

## Performance notes

The bispectrum's O(n_freq²) double loop over frequency pairs (j, k) was recently
optimized: each row's frequencies are now batched into a single `wtAtf2_batch` call
instead of one call per (j, k) pair, since the expensive part (signal FFT/padding/
preprocessing) doesn't depend on which frequency is being evaluated. See
[Refactor Notes](../developer-guide/refactor-notes.md) for the verification
methodology used.

## Key parameters

- **Bispectrum type** (`111`/`112`/`122`/`222`) — as tabulated above; see
  [REST API Reference](../api-and-ml/rest-api-reference.md#bispectrum-analysis) for the
  same typing used by FastMODA.
- **Frequency range and resolution** — inherited from the underlying wavelet
  transform. Note the cost is quadratic in the number of frequencies.

## In the web app

The [Bispectrum page](../using-moda/web-app.md#bispectrum) computes all four types on
1–2 signals, with GPU acceleration where available.

## Related pages

- [Time-Frequency Analysis](time-frequency-analysis.md) — the transform the bispectrum
  is built on.
- [Surrogate Testing](surrogate-testing.md) — for assessing whether a bispectral peak
  is statistically significant.
