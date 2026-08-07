# Linear Algebra & Eigenvalues

!!! info "Stub — outline of planned content"

Builds on [Foundations](foundations.md)'s brief vector/matrix introduction. Written to
be accessible starting from GCSE/A-level maths, up through what's needed to understand
eigenvalues at the level MODA uses them.

## Planned content

### 1. Matrices as transformations

- A matrix as something that takes a vector in and gives a (possibly rotated,
  stretched, or skewed) vector out.
- Worked geometric examples: scaling, rotation, shear, in 2D, with diagrams.

### 2. Eigenvectors and eigenvalues, built from intuition

- The question: "are there any vectors a given matrix *doesn't* rotate — only
  stretches or shrinks?" Those are the eigenvectors; the stretch factor is the
  eigenvalue.
- Formal definition: $A\mathbf{v} = \lambda\mathbf{v}$.
- Worked 2×2 example by hand (characteristic polynomial, solving for $\lambda$, then
  $\mathbf{v}$), building intuition before generalizing.
- Why eigenvalues/eigenvectors are basis-independent, meaningful properties of the
  transformation itself, not of how it happens to be written down.

### 3. Covariance matrices and their eigenvalues

- A covariance matrix as a description of how a set of variables vary together.
- Its eigenvectors are the directions of greatest (and least) joint variation; its
  eigenvalues are how much variation lies along each direction — the basis of
  Principal Component Analysis (PCA), mentioned as a widely-known application to
  anchor the intuition.

### 4. Where this shows up in MODA

- `bayes_main.m`'s propagation-function routines build and manipulate covariance-like
  matrices (`Inv_Diffusion`, built via `diag()` in the vectorized version — see
  [Refactor Notes](../developer-guide/refactor-notes.md)) as part of propagating
  uncertainty through the Bayesian filter over time; eigen-structure of these matrices
  relates to how uncertainty grows/shrinks along different directions in parameter
  space between observations.
- Forward pointer to [Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md)
  for the full algorithm this supports.

### 5. Numerically finding eigenvalues (brief, conceptual)

- Why MATLAB/NumPy compute eigenvalues iteratively rather than via the characteristic
  polynomial for anything but small matrices (numerical stability) — `eig()` as a
  black box, with a one-paragraph intuition for what it's doing.

## Next

[Probability & Bayesian Inference](probability-and-bayesian-inference.md) — priors,
posteriors, and testing whether a result is real or chance.
