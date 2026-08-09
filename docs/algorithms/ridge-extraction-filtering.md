# Ridge Extraction & Filtering

## What it computes

Given a signal's [time-frequency representation](time-frequency-analysis.md), a
**ridge** is the curve of (time, frequency) points along which the representation's
amplitude is locally maximal — it tracks a signal's dominant instantaneous frequency
as it changes over time. **Filtering** then reconstructs a time-domain signal
corresponding to just the oscillation along that ridge (or within a frequency band
around it), separating it from other components/noise.

This is useful when a signal is a mixture of several oscillatory components at
different (possibly time-varying) frequencies — e.g. separating a respiratory
component from a cardiac component in a physiological recording.

## The mathematics

### Ridges as an optimal path

The naive approach — at each time $t_n$, take the frequency with the largest amplitude —
fails on real data: noise makes the peak jump erratically between competing components,
producing a shattered, discontinuous curve.

MODA instead treats ridge extraction as a **global optimisation** over the whole curve.
It finds the sequence of frequency indices $\{k_n\}$ maximising a path functional that
rewards amplitude while penalising implausible behaviour:

$$
\max_{\{k_n\}} \; \sum_{n} \Big[\, \underbrace{A(k_n, t_n)}_{\text{amplitude}}
\;+\; \underbrace{w_1\!\left(f_{k_n} - f_{k_{n-1}}\right)}_{\text{jump penalty}}
\;+\; \underbrace{w_2\!\left(f_{k_n}\right)}_{\text{deviation penalty}} \Big]
$$

The two penalty terms encode the physical prior:

- $w_1$ penalises **frequency jumps between adjacent time points**. Real oscillators
  change frequency continuously; noise-driven peak-hopping does not. By default this is
  linear in the jump size, $-\lambda|\Delta f|$, normalised by the transform's own
  frequency resolution.
- $w_2$ penalises **excursions from the curve's overall mean frequency**, discouraging
  the ridge from wandering off onto an unrelated component.

Because $w_1$ couples each step only to its immediate predecessor, the maximisation has
optimal substructure and is solved exactly by **dynamic programming** rather than by
searching the exponentially many possible paths:

$$
U(k, t_n) \;=\; A(k,t_n) \;+\; \max_{j}\Big[\, U(j, t_{n-1}) + w_1\!\left(f_k - f_j\right) \Big]
$$

accumulating forward through time and then backtracking from the best final state
(`pathopt` in `ecurve.m`). The alternative `onestepopt` mode is greedy — it commits to
the best local step at each time — which is faster but can be trapped by a strong
transient.

!!! note "Why this stage is not a vectorisation target"
    Each DP step depends on the previous one, so the recursion is inherently
    sequential. The per-time-step amplitude extraction (a `max` over frequency) *is*
    vectorised; the recursion itself is not. See
    [Refactor Notes](../developer-guide/refactor-notes.md).

### Reconstruction from a ridge

Once the ridge $f_r(t)$ is known, `rectfr.m` inverts the transform restricted to it,
recovering the component's amplitude $A(t)$, phase $\phi(t)$ and instantaneous
frequency $\nu(t)$. Integrating the wavelet coefficients across a band around the ridge
gives the analytic signal:

$$
\zeta(t) \;=\; \frac{1}{C_\psi} \int_{\text{band}} W_x(f,t)\, \frac{df}{f}
\qquad\Longrightarrow\qquad
A(t) = |\zeta(t)|, \quad \phi(t) = \arg \zeta(t)
$$

The normalisation constant $C_\psi$ depends only on the wavelet — for the lognormal,
$C_\psi = \sqrt{\pi/2}\,/\,q$ with $q = 2\pi f_0$. A second constant $D_\psi$ appears in
the frequency estimator, which recovers instantaneous frequency as an
amplitude-weighted centroid of the band rather than simply the ridge's own frequency:

$$
\nu(t) \;=\; \frac{1}{D_\psi\,\zeta(t)} \int_{\text{band}} f\; W_x(f,t)\, \frac{df}{f}
$$

Note $D_\psi = \infty$ for the Morlet and Morse wavelets, in which case the code falls
back to the direct estimator. Two reconstruction routes exist — **ridge-based**
(integrate a narrow band around the peak) and **band-based** (integrate a fixed
frequency band regardless of where the peak sits); the former follows a moving
component, the latter is closer to a conventional bandpass filter.

## Source files

- `allguis/guis/filtering/Functions/ecurve.m` — ridge extraction (amplitude/index
  tracking, dynamic-programming path optimization).
- `allguis/guis/filtering/Functions/rectfr.m` — reconstruction of the time-domain
  signal from a ridge/band.
- `allguis/guis/filtering/Filtering.m` — the app module.

## Key parameters

- **Ridge-tracking penalty** — the $\lambda$ weighting $w_1$ above: how much the ridge
  may jump in frequency between adjacent time points (trades off following genuine
  frequency changes vs. resisting noise).
- **Band width** — for band-based (rather than pure-ridge) filtering, the frequency
  width around the ridge to include in reconstruction.
- **Path optimisation on/off** — global dynamic programming vs. greedy one-step.

## Related pages

- [Time-Frequency Analysis](time-frequency-analysis.md) — the representation ridges
  are extracted from.
- [Dynamical Bayesian Inference](dynamical-bayesian-inference.md) — a common
  downstream use of filtered/reconstructed phase signals: the $\phi(t)$ recovered here
  is exactly its input.
