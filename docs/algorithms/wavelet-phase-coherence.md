# Wavelet Phase Coherence

## What it computes

Given two signals, wavelet phase coherence measures how consistently the *phase
difference* between them is maintained over time, at each frequency — i.e. whether two
oscillators are synchronized, and at which frequencies/time intervals. A coherence
value near 1 means a stable phase relationship; a value near 0 means the phases drift
independently.

This differs from simple correlation: two signals can be strongly phase-coherent at
one frequency band while being essentially uncorrelated overall.

## The mathematics

### Phase from the wavelet transform

Both signals are first passed through the
[wavelet transform](time-frequency-analysis.md), giving complex coefficients whose
argument is the instantaneous phase at each frequency and time:

$$
W_1(f,t) = A_1(f,t)\, e^{i\phi_1(f,t)}, \qquad
W_2(f,t) = A_2(f,t)\, e^{i\phi_2(f,t)}
$$

### Time-averaged coherence

The key quantity is the phase *difference* $\Delta\phi(f,t) = \phi_1(f,t) - \phi_2(f,t)$.
Each time point contributes a unit vector $e^{i\Delta\phi}$ on the complex plane, and
coherence is the length of their average:

$$
C(f) \;=\; \left| \frac{1}{N} \sum_{n=1}^{N} e^{i\left(\phi_1(f,t_n) - \phi_2(f,t_n)\right)} \right|
$$

This is exactly what `wphcoh.m` computes. The geometry is worth pausing on, because it
explains the whole measure:

- If $\Delta\phi$ is **constant**, every unit vector points the same way, they sum
  coherently, and $C \to 1$ — regardless of *what* the constant phase lag is.
- If $\Delta\phi$ is **uniformly random**, the vectors cancel and $C \to 0$.

So $C$ measures the *consistency* of the phase relationship, not its value, and it is
entirely independent of the amplitudes $A_1, A_2$ — only the arguments enter. Two very
weak but tightly-locked oscillations score as highly as two strong ones.

The routine also returns the mean phase difference $\arg\left(\langle e^{i\Delta\phi}\rangle\right)$,
which *is* the lag, and is only meaningful when $C$ is high.

!!! warning "Coherence is biased upward at short records"
    With $N$ effectively-independent samples, even two unrelated signals give
    $C \approx 1/\sqrt{N}$ rather than 0, because a random walk of $N$ unit steps has
    expected length $\sqrt{N}$. The bias is worst at low frequencies, where few cycles
    fit in the record. This is why a raw coherence value means little without
    [surrogate testing](surrogate-testing.md).

### Time-localized coherence

Averaging over the whole record assumes the coupling is stationary. `tlphcoh.m` instead
averages over a sliding window, giving coherence as a function of both time and
frequency:

$$
C(f,t) \;=\; \left| \frac{1}{w_f} \sum_{n \,\in\, \text{window}(t)} e^{i\left(\phi_1(f,t_n)-\phi_2(f,t_n)\right)} \right|
$$

Crucially the window is a fixed number of **cycles**, not a fixed duration — in the code
$w_f = \texttt{wsize} \cdot f_s / f$, with `wsize` defaulting to 10. Each frequency
therefore averages over the same number of oscillations, so low- and high-frequency
coherence estimates carry comparable statistical weight and the same upward bias.
Implementation-wise this is a cumulative-sum trick, giving the whole sliding average in
$O(N)$ rather than $O(N w)$.

## Source files

- `allguis/guis/coherence/Functions/wphcoh.m` — the core coherence averaging routine.
- `allguis/guis/coherence/Functions/tlphcoh.m` — time-localized phase coherence.
- `allguis/guis/coherence/CoherenceMulti.m` — the app module, supporting up to several
  simultaneous signal pairs.

## Key parameters

- **Window length** (`wsize`) — how many cycles to average the phase difference over,
  for time-localized coherence.
- **Frequency range** — inherited from the underlying wavelet transform.

## Statistical significance

Coherence values are compared against a surrogate distribution (see
[Surrogate Testing](surrogate-testing.md)) to determine whether an observed coherence
level is significant, rather than arising by chance from two independent oscillators —
which, per the bias above, it frequently can.

## In the web app

The [Coherence page](../using-moda/web-app.md#wavelet-coherence) runs this on 2–6
signals, computing every pairwise combination, with GPU acceleration where available.

## Related pages

- [Dynamical Bayesian Inference](dynamical-bayesian-inference.md) — goes further,
  estimating the *form* and *direction* of the coupling rather than only its
  consistency.
- [Maths Primer → Worked Example](../maths-primer/worked-example-heartbeat.md)
  for coherence applied to a cardiorespiratory-style signal pair.
