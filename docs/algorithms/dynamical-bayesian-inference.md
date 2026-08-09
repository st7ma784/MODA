# Dynamical Bayesian Inference

## What it computes

Given two signals believed to be coupled oscillators (e.g. cardiac and respiratory
rhythms), dynamical Bayesian inference estimates the **coupling functions** describing
how each oscillator's phase dynamics depend on the other's — including the *direction*
and *strength* of coupling, and how these change over time. Unlike coherence (which
only says *whether* two signals are synchronized), this method estimates the actual
functional form of the interaction.

## The mathematics

### The model

Each signal is reduced to a phase $\phi_1(t), \phi_2(t)$ — in practice via
[ridge extraction & filtering](ridge-extraction-filtering.md) — and the pair is modelled
as two coupled phase oscillators driven by noise:

$$
\dot{\phi}_1 = f_1(\phi_1,\phi_2) + \xi_1(t), \qquad
\dot{\phi}_2 = f_2(\phi_1,\phi_2) + \xi_2(t)
$$

The unknowns are the **coupling functions** $f_1, f_2$ — whole functions of two
variables, not scalars. To make them estimable they are expanded in a truncated 2-D
Fourier series of order $b_n$:

$$
f_1(\phi_1,\phi_2) = \omega_1 + \sum_{k=1}^{b_n}\Big[ a_k \sin k\phi_1 + b_k \cos k\phi_1
+ c_k \sin k\phi_2 + d_k \cos k\phi_2 \Big]
+ \sum_{k,l}\Big[ \cdots \sin(k\phi_1 + l\phi_2) + \cdots \cos(k\phi_1 + l\phi_2)\Big]
$$

The terms in $\phi_2$ alone, and the mixed terms, are what carry the influence of
oscillator 2 on oscillator 1. Fitting reduces to estimating the coefficient vector
$\mathbf{c}$, of dimension $M = 2 + 2\left[(2b_n+1)^2 - 1\right]$ — which is why $b_n$
is kept small (a typical $b_n = 2$ already gives 50 parameters).

`CFprint.m` evaluates this series back onto a $(\phi_1,\phi_2)$ grid, which is what the
coupling-function surface plots show.

### Why "dynamical": window-to-window propagation

The signal is divided into windows. Within each, Bayesian inference yields a posterior
over the coefficients — a mean $\mathbf{c}$ and covariance $\Xi$ — from that window's
phase data alone (`bayesPhs`).

The step that makes the method *dynamical* is what happens **between** windows. The
posterior of window $n$ becomes the prior of window $n+1$, but with its covariance
deliberately inflated:

$$
\Xi^{-1}_{\text{prior},\,n+1} \;=\; \Big(\Xi^{-1}_{\text{post},\,n} + \Sigma_{\text{diff}}\Big)^{-1},
\qquad
\Sigma_{\text{diff}} = p^2 \,\mathrm{diag}\!\left(\Xi^{-1}_{\text{post},\,n}\right)
$$

with $p$ the propagation constant scaled by the window duration. This is a random-walk
assumption on the parameters, and it sets the method between two failure modes:

- **No propagation** ($p \to \infty$, prior forgotten) — every window is re-estimated
  from scratch, so the inferred coupling is dominated by estimation noise.
- **No inflation** ($p = 0$) — the posterior hardens after the first few windows and
  the method cannot track genuine time-variation.

$p$ tunes how fast coupling is allowed to drift.

### Directionality

`dirc.m` collapses the fitted coefficients into scalars by taking the norm of the
coefficient block describing each direction of influence:

$$
\kappa_1 = \|\mathbf{q}_1\|, \qquad \kappa_2 = \|\mathbf{q}_2\|, \qquad
D = \frac{\kappa_2 - \kappa_1}{\kappa_1 + \kappa_2}
$$

The **directionality index** $D \in [-1, 1]$ reads directly: $D > 0$ means oscillator 1
predominantly drives oscillator 2, $D < 0$ the reverse, and $D \approx 0$ means
symmetric (mutual) coupling. This asymmetry is precisely what coherence cannot provide —
coherence is symmetric in its two arguments by construction.

!!! warning "Phase estimation quality dominates the result"
    The method infers dynamics *of the phases it is given*. If the ridge extraction
    feeding it picked up the wrong component, or the band was too wide and mixed two
    oscillations, the coupling functions describe that artefact faithfully. Inspect the
    filtered components before trusting the inference.

## Source files

- `allguis/guis/bayesian/Functions/bayes_main.m` — main recursive Bayesian filtering
  loop (propagates a probability distribution over model parameters through time;
  inherently sequential, not a vectorization target).
- `allguis/guis/bayesian/Functions/CFprint.m` — coupling function evaluation on a grid,
  from inferred Fourier coefficients.
- `allguis/guis/bayesian/Functions/dirc.m` — extracts directionality/coupling-strength
  indices from the inferred parameters.
- `allguis/guis/bayesian/Bayesian.m` — the app module.

`CFprint.m` and `dirc.m`, along with `bayes_main.m`'s diagonal-matrix-construction
steps, were vectorized (matrix ops in place of nested loops) without changing the
underlying (necessarily sequential) recursive filter — see
[Refactor Notes](../developer-guide/refactor-notes.md).

## Key parameters

- **Frequency bands** for each signal — the oscillation of interest is first isolated
  via [ridge extraction & filtering](ridge-extraction-filtering.md) into a band around
  its dominant frequency.
- **Window length** and **overlap** — the time window over which coupling parameters
  are estimated before propagating to the next window. Longer windows give more stable
  estimates but blur faster coupling changes.
- **Fourier order** ($b_n$) — how much structure the coupling function may have; costs
  $O(b_n^2)$ parameters.
- **Propagation constant** ($p$) — how quickly the coupling is permitted to drift.
- **Number of surrogates** — for significance testing of the inferred coupling (see
  [Surrogate Testing](surrogate-testing.md)).

## In the web app

The [Bayesian Inference page](../using-moda/web-app.md#bayesian-inference) exposes this
for signal pairs, with GPU acceleration and an `n_surrogates` parameter.

## Related pages

- [Wavelet Phase Coherence](wavelet-phase-coherence.md) — the symmetric,
  functional-form-free counterpart.
- [Maths Primer → Probability & Bayesian Inference](../maths-primer/probability-and-bayesian-inference.md)
  for the conceptual background.
