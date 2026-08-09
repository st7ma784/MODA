# Worked Example: Heartbeat Profiling

One running example applying every earlier page in the [Maths Primer](index.md) to a
single realistic case, end to end. The signal is synthetic, so we know the right answer
and can check each step against it — the point of the exercise is to see *why* each
tool is reached for, and what it looks like when it works.

## The scenario

Two coupled physiological rhythms:

- a **respiratory** oscillation near 0.25 Hz (about 15 breaths/minute), itself slowly
  drifting, and
- a **cardiac** oscillation near 1.1 Hz (about 66 bpm) whose rate is pushed up and down
  by where respiration is in its cycle.

That second effect is real and well studied: **respiratory sinus arrhythmia**, the
reason your pulse quickens slightly as you inhale. It gives us genuine coupling to
detect, with a known direction (breathing drives heart rate, not the reverse).

## Generating the signals

```python
import numpy as np
from scipy.io import savemat

fs = 20.0           # Hz
duration = 600.0    # seconds
t = np.arange(0, duration, 1 / fs)
rng = np.random.default_rng(0)

# Respiration: a slow oscillator that itself drifts in rate
f_resp = 0.25 + 0.03 * np.sin(2 * np.pi * t / 300.0)
phi_resp = 2 * np.pi * np.cumsum(f_resp) / fs
resp = np.sin(phi_resp)

# Heart: faster oscillator, rate modulated BY respiration (the coupling)
coupling = 0.08
f_heart = 1.1 + 0.05 * np.sin(2 * np.pi * t / 400.0) + coupling * resp
phi_heart = 2 * np.pi * np.cumsum(f_heart) / fs
heart = (1 + 0.25 * resp) * (np.sin(phi_heart) + 0.3 * np.sin(2 * phi_heart))

# Baseline wander: breathing physically moves the chest/sensor, adding a direct
# respiratory component to the pulse trace.
heart += 0.6 * resp

heart += 0.15 * rng.standard_normal(t.size)
resp  += 0.05 * rng.standard_normal(t.size)

savemat('heart.mat', {'sig': heart})
savemat('resp.mat',  {'sig': resp})
```

Note $f_s = 20$ Hz gives a Nyquist limit of 10 Hz — comfortably above everything of
interest, per [Fourier & Convolution](fourier-and-convolution.md#3-sampling-and-the-nyquist-rate).
The phases are built by **cumulative sum** of a time-varying frequency, which is the
discrete form of $\phi(t) = 2\pi\int f(t')\,dt'$; writing `sin(2*pi*f*t)` with a varying
`f` would be wrong, producing a different signal from the one intended.

## 1. The raw signal

Plotted in the time domain, `heart` is a dense oscillation whose envelope visibly
breathes. What you *cannot* see is the thing we care about: whether the rate speeds and
slows in step with respiration. The rate change is a few percent, spread over minutes,
and the eye has no purchase on it.

A whole-record Fourier transform does not rescue this. It reports energy near 1.1 Hz and
near 0.25 Hz, but — as [Fourier & Convolution](fourier-and-convolution.md#5-limits-of-the-plain-fourier-transform)
argues — it averages over all time and so cannot say the 1.1 Hz component was at 1.05 Hz
during one stretch and 1.18 Hz during another. **We need time resolution.**

## 2. Time-frequency analysis

Apply the [wavelet transform](../algorithms/time-frequency-analysis.md) to `heart.mat`
at $f_s = 20$, over 0.5–3 Hz. The scalogram shows a bright band near 1.1 Hz that visibly
undulates — the heart rate varying over time — plus a fainter band near 2.2 Hz, the
second harmonic from the non-sinusoidal pulse shape.

The [Wavelets](wavelets.md) page explains why the band is sharper here than a WFT would
render it: at ~1 Hz the wavelet spans only a few seconds, so a rate change lasting tens
of seconds is easily resolved, while the log-spaced frequency axis still gives fine
resolution down at the respiratory rate.

Running this through FastMODA's CWT endpoint reports:

```
mean_ridge_freq = 1.11 Hz
dominant_freq   = 1.11 Hz
```

Against a designed mean of 1.1 Hz. The transform recovers the right answer.

## 3. Ridge extraction

The scalogram shows the rate varying but does not hand you the curve. Following the
brightest point through time — the **ridge** — turns the picture into a number per
instant:

$$
f_r(t) \;\approx\; 1.1 + 0.05\sin\!\left(\tfrac{2\pi t}{400}\right) + 0.08\,\text{resp}(t)
$$

This is the instantaneous heart rate, and it is where the coupling lives. Note that
naive per-instant maximum-picking would fail here: the second harmonic at 2.2 Hz is a
competing local maximum, and noise makes the peak hop. The dynamic-programming path
optimisation in [Ridge Extraction & Filtering](../algorithms/ridge-extraction-filtering.md)
penalises those jumps and follows the genuine component.

## 4. Wavelet phase coherence

Now pair the two signals. Running
[coherence](../algorithms/wavelet-phase-coherence.md) on `heart.mat` and `resp.mat` at
$f_s = 20$ over 0.05–3 Hz:

| Frequency | Coherence |
|---|---|
| **0.245 Hz** | **1.000** |
| 0.519 Hz | 0.329 |
| 1.101 Hz | 0.418 |

A sharp peak at exactly the respiratory frequency, and unremarkable values elsewhere.
The two signals maintain a rock-steady phase relationship at 0.245 Hz and drift
independently at every other frequency — which is precisely what coherence is built to
report.

!!! note "Why this one is 1.000, and yours will not be"
    Coherence reaches its ceiling here because `heart` literally contains a scaled copy
    of `resp` via the baseline-wander term, so at that frequency the phase difference is
    *constant by construction*. Real recordings never look like this. Treat it as a
    demonstration that the estimator does what it claims on a case with a known answer —
    not as a target to expect from data.

Notice also that coherence at **1.1 Hz is unremarkable (0.42)** even though the coupling
genuinely acts there. Coherence compares the *same* frequency band in both signals, and
respiration has no energy at 1.1 Hz. The frequency-modulation coupling is invisible to
this measure — a real limitation, and the reason the next step exists.

## 5. Dynamical Bayesian inference

To recover the coupling that coherence missed, work with phases rather than raw signals:
extract $\phi_{\text{heart}}(t)$ and $\phi_{\text{resp}}(t)$ by
[ridge extraction](../algorithms/ridge-extraction-filtering.md) in a band around each
rhythm, then infer the coupling functions with
[dynamical Bayesian inference](../algorithms/dynamical-bayesian-inference.md).

Because we built the signal, we know what should come back. Respiration enters the
heart's phase equation through the `coupling * resp` term, and nothing feeds back the
other way, so the expected findings are:

- a coupling function $f_{\text{heart}}$ with real dependence on $\phi_{\text{resp}}$,
- a coupling function $f_{\text{resp}}$ essentially flat in $\phi_{\text{heart}}$, and
- a **directionality index** $D$ clearly on the respiration → heart side.

That asymmetry is the payoff. Coherence is symmetric in its two arguments and can never
distinguish "A drives B" from "B drives A"; the Bayesian estimate can, because it fits
each oscillator's dynamics separately.

## 6. Significance

Every number so far could in principle be a coincidence of finite-length noise. Coherence
in particular is [biased upward](../algorithms/wavelet-phase-coherence.md) by roughly
$1/\sqrt{N}$, so a moderate value proves little on its own.

Run [surrogate testing](../algorithms/surrogate-testing.md) with, say, 99 IAAFT
surrogates — preserving each signal's spectrum and amplitude distribution while
destroying the phase relationship between them — and re-run the coherence on each. The
0.245 Hz peak should sit far outside that distribution; the 1.1 Hz value probably will
not. With 99 surrogates the strongest available claim is $p \le 0.01$, since
$p = 1/(N+1)$.

For the coupling result, `tshift` surrogates are the better null: they preserve all of
each signal's own structure and break only the alignment between them, isolating genuine
inter-signal coupling.

## 7. Eigenvalues in context

Finally, a callback. Step 5's Bayesian filter carried not just a coefficient estimate but
a covariance matrix $\Xi$, inflated between windows so the estimate could track change.
[Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md) is what that inflation
means geometrically: the eigenvalues of $\Xi$ are the uncertainty along each direction in
parameter space, and growing them says *time has passed, so we know less than we did*.

Directions the data constrain well stay tight; poorly-constrained ones loosen fastest.
That is why the method can follow coupling that genuinely varies — such as the slow
0.05 Hz drift we baked into `f_heart` — instead of freezing on the first window's answer.

## Related pages

This example touches every algorithm page under
[Algorithms](../algorithms/time-frequency-analysis.md) — see those for the formal
parameter reference for each step.
