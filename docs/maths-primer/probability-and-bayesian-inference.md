# Probability & Bayesian Inference

!!! info "Stub — outline of planned content"

Builds on [Foundations](foundations.md). Underpins
[Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) and
[Surrogate Testing](../algorithms/surrogate-testing.md).

## Planned content

### 1. Probability basics

- Probability as a measure of belief/frequency; probability distributions
  (discrete and continuous), mean and variance as summary statistics.
- The normal (Gaussian) distribution — why it shows up everywhere (brief, intuitive
  mention of the Central Limit Theorem), and its role as a noise model.

### 2. Bayes' theorem

- $P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta) \, P(\theta)$ — the
  prior, the likelihood, and the posterior, explained with a simple worked example
  (e.g. a medical-test-style example) before applying the idea to signal parameters.
- Why "updating a belief as new evidence arrives" is exactly what a Bayesian *filter*
  does over time, one data window at a time — the conceptual link to
  `bayes_main.m`'s recursive structure.

### 3. Bayesian inference for dynamical systems

- Treating the coefficients of a coupling function as unknown parameters with a prior
  distribution, and updating that distribution's estimate (mean and covariance) as
  each new segment of data arrives — propagation forward in time
  (["Linear Algebra & Eigenvalues"](linear-algebra-and-eigenvalues.md) covers the
  covariance/eigenvalue side of this).
- Why this approach naturally handles time-varying coupling, unlike a single
  least-squares fit over the whole signal.

### 4. Hypothesis testing and significance

- Null hypothesis, p-values, and what "statistically significant" actually means (and
  its common misinterpretations).
- Surrogate-based significance testing as a distribution-free alternative to classical
  parametric tests: rather than assuming a theoretical null distribution, build one
  empirically by re-running the analysis on randomized surrogate signals — direct
  link to [Surrogate Testing](../algorithms/surrogate-testing.md).

## Next

[Worked Example: Heartbeat Profiling](worked-example-heartbeat.md) — everything in this
primer applied to one signal, end to end.
