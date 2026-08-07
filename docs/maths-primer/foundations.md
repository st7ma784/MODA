# Foundations

!!! info "Stub — outline of planned content"

Prerequisite reading for the rest of the [Maths Primer](index.md). Assumes GCSE maths;
builds up to what A-level Further Maths / first-year undergraduate courses cover.

## Planned content

### 1. Periodic and trigonometric functions

- Sine, cosine, and the unit circle; amplitude, frequency, phase, period.
- Radians vs degrees, and why radians are the natural unit for the maths later on.
- Why $A\sin(\omega t + \phi)$ is the general form of a single-frequency oscillation,
  and how amplitude/frequency/phase map onto real signal properties (e.g. a
  heartbeat's strength, rate, and timing).

### 2. Complex numbers

- $i = \sqrt{-1}$, the complex plane, modulus and argument.
- Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ — the bridge between
  oscillations and exponentials that the rest of the primer relies on.
- Why signal processing represents a real oscillation as a complex exponential
  (amplitude *and* phase in one number) rather than tracking sine and cosine
  separately.

### 3. Vectors and basic matrix notation

- Vectors as ordered lists of numbers / points in space.
- Dot product, and its geometric meaning (projection, "how aligned are two vectors").
- Matrices as linear maps; matrix-vector and matrix-matrix multiplication.

### 4. Rates of change and sampling

- Derivatives as instantaneous rate of change (brief, intuition-only refresher).
- What it means to *sample* a continuous signal at a fixed rate, and why the sampling
  frequency matters (forward pointer to the Nyquist rate, covered in
  [Fourier & Convolution](fourier-and-convolution.md)).

## Next

[Fourier & Convolution](fourier-and-convolution.md) — building any signal out of sine
waves.
