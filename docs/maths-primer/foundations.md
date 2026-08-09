# Foundations

Prerequisite reading for the rest of the [Maths Primer](index.md). Assumes GCSE maths;
builds up to what A-level Further Maths / first-year undergraduate courses cover.

## 1. Periodic and trigonometric functions

Picture a point travelling anticlockwise around a circle of radius $A$. Its height
above the centre traces a **sine wave**; its horizontal offset traces a **cosine wave**.
Every oscillation in this primer is a version of that picture.

A single-frequency oscillation is written

$$
x(t) = A\sin(\omega t + \phi)
$$

with three independent knobs:

| Symbol | Name | What it controls | Physical example |
|---|---|---|---|
| $A$ | amplitude | how far the signal swings | strength of a heartbeat |
| $\omega$ | angular frequency | how fast it oscillates | heart rate |
| $\phi$ | phase | where in the cycle it starts | timing relative to breathing |

Amplitude and frequency are intuitive. **Phase is the one that matters most in MODA**,
and it is the least familiar: it says *where in its cycle* an oscillation is at a given
instant. Two signals can have identical amplitudes and frequencies and still differ
entirely in how they relate to each other — that relationship lives in the phase, and
it is what [coherence](../algorithms/wavelet-phase-coherence.md),
[bispectrum](../algorithms/wavelet-bispectrum.md) and
[Bayesian inference](../algorithms/dynamical-bayesian-inference.md) all measure.

### Radians, and why

Angular frequency $\omega$ is in **radians per second**, related to the ordinary
frequency $f$ in Hz (cycles per second) by

$$
\omega = 2\pi f
$$

One full turn around the circle is $2\pi$ radians. Radians are not a stylistic
preference: they are the unit in which $\frac{d}{d\theta}\sin\theta = \cos\theta$ holds
without a stray conversion factor, which is why every formula later in this primer —
and every line of MODA's source — uses them. When you see $2\pi f$ in the code, it is
converting a frequency you specified in Hz into the units the maths needs.

### Period

The **period** $T = 1/f$ is the time for one complete cycle. A 1 Hz oscillation has a
period of 1 second. This matters practically: to see $N$ cycles of an oscillation you
need at least $N/f$ seconds of recording, which is why low-frequency analysis needs
long records.

## 2. Complex numbers

### The idea

A complex number is a pair of real numbers treated as one object,
$z = a + ib$, where $i = \sqrt{-1}$. Plot $a$ horizontally and $b$ vertically and $z$
is a point in the plane. That point can equally be described by:

- its **modulus** $|z| = \sqrt{a^2+b^2}$ — distance from the origin, and
- its **argument** $\arg z = \arctan(b/a)$ — the angle it makes.

### Euler's formula

The bridge that makes everything downstream work:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

So $e^{i\theta}$ is the point on the unit circle at angle $\theta$. It follows that a
general oscillation is compactly

$$
A e^{i(\omega t + \phi)}
$$

whose modulus is the amplitude $A$ and whose argument is the phase $\omega t + \phi$.

### Why signal processing insists on this

**One complex number carries amplitude and phase simultaneously.** This is not a
notational nicety — it is the reason MODA's transforms return complex arrays. In
[Time-Frequency Analysis](../algorithms/time-frequency-analysis.md) the wavelet
transform $W(f,t)$ is complex at every point, and the two pieces are used for different
things:

$$
A(f,t) = |W(f,t)| \quad\text{(amplitude — how much)}, \qquad
\phi(f,t) = \arg W(f,t) \quad\text{(phase — where in the cycle)}
$$

Coherence throws away $|W|$ and keeps only $\arg W$. Scalograms plot only $|W|$. Had we
tracked sine and cosine components separately, every one of those operations would be
clumsier.

One more property earns its keep: multiplying complex numbers **adds their arguments**,
and conjugating one **negates** its argument. So

$$
W_1 W_2^{*} \;\text{has argument}\; \phi_1 - \phi_2
$$

That single fact is why phase *differences* — the quantity coherence and the bispectrum
are built on — fall out of a plain multiplication rather than needing explicit angle
subtraction.

## 3. Vectors and basic matrix notation

A **vector** is an ordered list of numbers, $\mathbf{v} = (v_1, v_2, \ldots, v_n)$,
equivalently a point or arrow in $n$-dimensional space. In MODA a vector might be a
signal's samples, or the coefficients describing a coupling function.

The **dot product** combines two vectors into one number:

$$
\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i = |\mathbf{a}||\mathbf{b}|\cos\theta
$$

Geometrically it measures **alignment**: largest when the vectors point the same way,
zero when perpendicular. Keep this in view — every transform in this primer is at heart
a dot product of the signal against a template (a sine wave, or a wavelet), asking "how
much does the signal look like this?"

The **norm** $\|\mathbf{v}\| = \sqrt{\mathbf{v}\cdot\mathbf{v}}$ is the vector's length;
it appears directly in `dirc.m`, where the norm of a block of coefficients becomes a
coupling strength.

A **matrix** is a rectangular array that maps vectors to vectors. Matrix-vector
multiplication $A\mathbf{v}$ takes each row's dot product with $\mathbf{v}$.
[Linear Algebra & Eigenvalues](linear-algebra-and-eigenvalues.md) develops this
properly.

## 4. Rates of change and sampling

### Derivatives, briefly

The derivative $\frac{dx}{dt}$ is the instantaneous rate of change — the slope of the
curve. You need only the intuition here: when
[Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) writes
$\dot\phi_1 = f_1(\phi_1,\phi_2)$, it is saying *the rate at which oscillator 1's phase
advances depends on where both oscillators currently are*. That is what a coupled
system means.

### Sampling

Real recordings are not continuous. A sensor measures at a fixed **sampling frequency**
$f_s$, producing values at times $0, 1/f_s, 2/f_s, \ldots$ — so a signal is a list of
numbers plus the single number $f_s$ that says how fast they were taken.

Everything frequency-related depends on $f_s$ being right. It is not stored in a bare
`.mat` array of samples, which is why both MODA and FastMODA make you type it, and why
getting it wrong rescales every frequency axis without any error appearing. The deeper
consequence — that $f_s$ imposes a hard ceiling on the frequencies you can recover at
all — is the Nyquist limit, covered next.

## Next

[Fourier & Convolution](fourier-and-convolution.md) — building any signal out of sine
waves.
