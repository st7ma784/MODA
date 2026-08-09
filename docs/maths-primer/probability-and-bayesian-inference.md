# Probability & Bayesian Inference

Builds on [Foundations](foundations.md). Underpins
[Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) and
[Surrogate Testing](../algorithms/surrogate-testing.md).

## 1. Probability basics

A **probability distribution** assigns plausibility across possible values. For a
continuous quantity it is a density $p(x)$, where the area under any interval is the
probability of landing in it, and the total area is 1.

Two summaries carry most of the weight:

$$
\mu = \mathbb{E}[x] = \int x\, p(x)\, dx, \qquad
\sigma^2 = \mathbb{E}\!\left[(x-\mu)^2\right]
$$

The **mean** is the centre; the **variance** is the spread. Its square root, the
standard deviation $\sigma$, is in the same units as $x$ and is usually the more
readable of the two.

### The normal distribution

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

The Gaussian is ubiquitous because of the **Central Limit Theorem**: sums of many
independent contributions tend toward a normal distribution almost regardless of the
individual shapes. Measurement noise is typically such a sum — thermal noise, quantisation,
interference — so a Gaussian noise model is usually a defensible default rather than a
lazy one.

It matters here for a specific reason. Phase-randomised
[surrogates](../algorithms/surrogate-testing.md) are sums of many random-phase
components, so the CLT drags them toward Gaussian amplitude distributions. If your real
signal is strongly non-Gaussian, the surrogates then differ from it in *two* ways, and a
significant result can no longer be attributed to phase structure alone. That is exactly
what the IAAFT family exists to repair.

## 2. Bayes' theorem

$$
P(\theta \mid D) = \frac{P(D \mid \theta)\, P(\theta)}{P(D)}
\;\;\propto\;\; \underbrace{P(D \mid \theta)}_{\text{likelihood}} \; \underbrace{P(\theta)}_{\text{prior}}
$$

- **Prior** $P(\theta)$ — belief about the parameters before seeing this data.
- **Likelihood** $P(D\mid\theta)$ — how well parameters $\theta$ explain the data.
- **Posterior** $P(\theta \mid D)$ — updated belief afterwards.

### Why the base rate bites

A test is 99% accurate for a condition affecting 1 in 10,000. You test positive. The
intuitive answer — 99% — is badly wrong.

Out of 1,000,000 people: 100 have the condition and about 99 test positive. The 999,900
healthy people yield about 9,999 false positives at a 1% error rate. So

$$
P(\text{condition} \mid +) = \frac{99}{99 + 9999} \approx 1\%
$$

The prior dominates. The lesson generalises: **evidence updates a prior, it does not
replace one**, and ignoring the prior makes strong evidence look far more conclusive
than it is.

### Bayes as a filter

Nothing forces the update to happen once. Take the posterior after one batch of data as
the prior for the next:

$$
P(\theta) \;\to\; P(\theta \mid D_1) \;\to\; P(\theta \mid D_1, D_2) \;\to\; \cdots
$$

Belief accumulates. This recursive structure is precisely `bayes_main.m`'s loop over
windows, and it is what "filter" means in this context.

## 3. Bayesian inference for dynamical systems

Applied to coupled oscillators, the unknown $\theta$ is the vector of Fourier
coefficients describing the coupling functions, and the posterior is summarised by a
mean $\mathbf{c}$ and covariance $\Xi$ — the latter analysed in
[Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md).

Straight recursive updating would assume the coupling is *constant*: each window's
evidence accumulates, the posterior narrows, and after enough windows it is effectively
frozen. For a system whose coupling genuinely changes — a body at rest and then
exercising — that is the wrong model.

The fix is to inflate the covariance between windows, encoding "the parameters may have
drifted since we last looked". The estimate can then follow real change while still
pooling evidence across time. Setting the propagation constant $p$ chooses where to sit
between two failure modes:

| $p$ | Behaviour | Failure |
|---|---|---|
| $p = 0$ | posterior never widens | cannot track genuine change |
| small | slow drift permitted | good for gradual coupling change |
| large | prior nearly discarded | estimates dominated by noise |

Compare a single least-squares fit over the whole recording: it returns one coupling
function and no way to express that the coupling changed halfway through. The Bayesian
filter returns a trajectory, and — through the covariance — an honest statement of
confidence at each point.

## 4. Hypothesis testing and significance

### What a p-value is, and is not

Classical testing posits a **null hypothesis** $H_0$ (the effect is absent) and asks how
surprising the observation would be if $H_0$ held. The **p-value** is

$$
p = P(\text{result at least this extreme} \mid H_0)
$$

Note carefully what that is *not*. It is not the probability the null is true, nor the
probability your finding is real — both are $P(H_0 \mid D)$, which by Bayes needs a
prior that a p-value never supplies. A result at $p = 0.04$ studying an implausible
effect remains, in all likelihood, a false positive.

### Surrogates as a distribution-free null

Classical tests need a theoretical null distribution, which usually requires assumptions
— independent samples, Gaussian noise — that time-series with strong autocorrelation
violate flatly. Coherence between two *independent* signals is not centred on zero; it
is biased upward by roughly $1/\sqrt{N}$, so comparing against a nominal zero would
declare almost anything significant.

Surrogate testing sidesteps the problem by **constructing** the null. Generate signals
that keep the properties you are not testing (spectrum, autocorrelation, amplitude
distribution) and destroy the one you are (phase relationships), then re-run the
identical analysis on each. The resulting spread of values is the null distribution —
inheriting whatever bias the estimator has, since the surrogates are the same length and
processed the same way.

Significance is then a rank: if the real value exceeds all $N$ surrogates,

$$
p = \frac{1}{N+1}
$$

which is why $N=19$ is the minimum for $p \le 0.05$ and $N=99$ for $p \le 0.01$. The
choice of surrogate method *is* the choice of null hypothesis — the most consequential
decision in the procedure. See [Surrogate Testing](../algorithms/surrogate-testing.md)
for the methods MODA provides.

## Next

[Worked Example: Heartbeat Profiling](worked-example-heartbeat.md) — everything in this
primer applied to one signal, end to end.
