# Surrogate Testing

## What it computes

Surrogate testing answers the question: *"is this result (coherence, bispectral peak,
coupling strength...) actually meaningful, or could it have arisen by chance from
unrelated signals with similar basic statistical properties?"*

The method generates many **surrogate signals** — randomized versions of the original
that preserve some property (typically the power spectrum / autocorrelation structure)
but destroy the property being tested for (e.g. phase relationships between signals).
The analysis is then re-run on the surrogates, giving a distribution of "chance"
results to compare the real result against. If the real result falls well outside the
surrogate distribution, it's considered statistically significant.

## The mathematics

### The null hypothesis is the surrogate

The logic runs backwards from usual: rather than assuming a parametric null
distribution, you *construct* data satisfying the null and measure what your statistic
does on it. The choice of surrogate method **is** the null hypothesis — which is the
one decision that matters most here, and the easiest to get wrong.

Given a statistic $S$ (coherence, bicoherence, coupling strength) computed on the real
data as $S_0$, and on $N$ surrogates as $S_1 \ldots S_N$, significance is assessed by
rank. If $S_0$ exceeds all $N$ surrogate values, the one-sided p-value is bounded by:

$$
p \;=\; \frac{1 + \#\{ i : S_i \geq S_0 \}}{N + 1}
$$

This bound is why surrogate counts look arbitrary but are not: $N = 19$ is the smallest
$N$ that can reach $p \le 0.05$ (since $1/20 = 0.05$), and $N = 99$ the smallest for
$p \le 0.01$. **No number of surrogates can produce a p-value smaller than
$1/(N+1)$** — if you need $p < 0.001$, you need at least 999 surrogates, regardless of
how extreme the real value is.

### Phase randomisation

The workhorse method is Fourier-transform (`FT`) surrogates. Take the FFT of the
signal, keep every magnitude exactly, and replace the phases with uniform random values
$\eta_k \sim U(0, 2\pi)$ (conjugate-symmetric, so the inverse transform stays real):

$$
\tilde{x} = \mathcal{F}^{-1}\Big\{ \left|\hat{x}(f)\right| e^{i\eta(f)} \Big\}
$$

Because the power spectrum is $|\hat{x}|^2$ and is untouched, the surrogate has
**exactly** the original's spectrum and autocorrelation — but all phase structure, and
hence any phase coupling, is destroyed. That is precisely the null that
[coherence](wavelet-phase-coherence.md) and
[bispectrum](wavelet-bispectrum.md) results need testing against, since both are
built on phase relationships and both are biased upward on finite records.

### Preserving the amplitude distribution

Phase randomisation forces the surrogate toward a Gaussian amplitude distribution
(a sum of many random-phase components tends to normal). If the original signal is
strongly non-Gaussian, the surrogates differ from it in *two* ways rather than one, and
a rejection can no longer be attributed to phase structure alone.

The **AAFT** and **IAAFT** families fix this. IAAFT alternates between two projections —
impose the original power spectrum, then rank-order the values back onto the original
amplitude distribution — iterating until the ranking stops changing:

$$
x^{(k+1)} \;=\; \text{rank-map}\Big( \mathcal{F}^{-1}\big\{ |\hat{x}_{\text{orig}}| \, e^{i \arg \hat{x}^{(k)}} \big\} \Big)
$$

Convergence is when successive iterations produce an identical rank ordering (the code's
`max(abs(oldrank-irank)) == 0` test), subject to an iteration cap. Each iteration
refines the previous surrogate, so the loop is inherently sequential and is not a
vectorisation target.

### Methods available

`surrcalc.m` implements:

| Method | Preserves | Destroys | Use when |
|---|---|---|---|
| `RP` | amplitude distribution only | all temporal structure | testing against pure noise |
| `FT` | power spectrum exactly | phase relationships | the standard phase-coupling null |
| `AAFT` | spectrum + amplitude distribution (approx.) | phase relationships | signal is non-Gaussian |
| `IAAFT1` / `IAAFT2` | both, iteratively refined | phase relationships | non-Gaussian, spectrum matters |
| `WIAAFT` | as IAAFT, wavelet-based | phase relationships | non-stationary signals |
| `CPP` | cycle structure | inter-cycle relations | phase-based analyses |
| `tshift` | everything within a signal | relative timing between signals | testing bivariate coupling only |

!!! note "`tshift` tests a different null"
    Time-shifted surrogates cyclically rotate one signal against the other. Each signal
    keeps *all* its own structure, linear and nonlinear — only their alignment is
    broken. This isolates genuine inter-signal coupling from the possibility that both
    signals merely have similar internal dynamics.

## Source files

- `surrcalc.m` — all surrogate generation methods listed above.
- Exposed in the desktop app via each module's surrogate-testing controls, and via
  FastMODA's `/tests` diagnostics page and the `n_surrogates` parameter on
  `/analyze_bayesian` (see [REST API Reference](../api-and-ml/rest-api-reference.md#bayesian-inference)).

## Key parameters

- **Number of surrogates** — sets the smallest attainable p-value at $1/(N+1)$, as
  above; more surrogates cost proportionally more computation, since the full analysis
  re-runs on each.
- **Surrogate method** — which properties of the original signal are preserved, i.e.
  which null hypothesis is being tested.

## Related pages

- [Wavelet Phase Coherence](wavelet-phase-coherence.md) and
  [Wavelet Bispectrum](wavelet-bispectrum.md) — the two analyses most commonly paired
  with surrogate testing in MODA, and the two most in need of it.
- [Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md)
  — conceptual background on significance testing.
