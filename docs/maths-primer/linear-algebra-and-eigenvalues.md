# Linear Algebra & Eigenvalues

Builds on [Foundations](foundations.md)'s brief vector/matrix introduction. Written to
be accessible starting from GCSE/A-level maths, up through what's needed to understand
eigenvalues at the level MODA uses them.

## 1. Matrices as transformations

The productive way to read a matrix is not "a grid of numbers" but **a machine that
takes a vector in and returns a transformed vector out**. For a $2\times2$ matrix,

$$
A\mathbf{v} =
\begin{pmatrix} a & b \\ c & d \end{pmatrix}
\begin{pmatrix} v_1 \\ v_2 \end{pmatrix}
= \begin{pmatrix} a v_1 + b v_2 \\ c v_1 + d v_2 \end{pmatrix}
$$

Each output component is a dot product of one row with the input. Familiar
transformations are just particular entries:

| Matrix | Effect |
|---|---|
| $\begin{pmatrix} 2 & 0 \\ 0 & 2\end{pmatrix}$ | scale everything by 2 |
| $\begin{pmatrix} 3 & 0 \\ 0 & 1\end{pmatrix}$ | stretch horizontally, leave vertical alone |
| $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ | rotate by $\theta$ |
| $\begin{pmatrix} 1 & 1 \\ 0 & 1\end{pmatrix}$ | shear — slide horizontally by height |

Matrix multiplication $AB$ means "apply $B$, then $A$", which is why it does not
commute: rotating then stretching differs from stretching then rotating.

## 2. Eigenvectors and eigenvalues

### The question they answer

Most transformations both rotate and stretch a vector. But look at the horizontal
stretch $\begin{pmatrix}3&0\\0&1\end{pmatrix}$: a vector along the $x$-axis comes back
pointing the *same way*, merely three times longer. A vector along the $y$-axis comes
back unchanged. Every other vector gets tilted.

So ask: **for a given matrix, which directions survive untilted?** Those are the
**eigenvectors**, and the factor each is scaled by is its **eigenvalue**:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

They expose what the transformation *does*, stripped of the coordinate system it
happens to be written in. For the stretch above, the eigenvectors are the two axes with
eigenvalues 3 and 1 — the description "stretches threefold this way, leaves that way
alone" is complete.

### Working one by hand

Take $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. Rearranged,
$(A - \lambda I)\mathbf{v} = \mathbf{0}$ has a non-zero solution only where the matrix
is singular, i.e. its determinant vanishes:

$$
\det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix}
= (2-\lambda)^2 - 1 = 0
$$

giving $\lambda = 3$ and $\lambda = 1$. Substituting $\lambda = 3$ yields
$\mathbf{v} = (1,1)$; $\lambda = 1$ yields $\mathbf{v} = (1,-1)$. So this matrix
stretches threefold along the diagonal and leaves the anti-diagonal untouched.

Note both eigenvalues are real and the eigenvectors perpendicular. That is guaranteed
here because $A$ is **symmetric** ($A = A^{T}$) — a fact that matters below, since
covariance matrices are always symmetric.

## 3. Covariance matrices and their eigenvalues

Given variables measured together, the **covariance matrix** records how they vary
jointly:

$$
\Sigma = \begin{pmatrix}
\operatorname{Var}(x) & \operatorname{Cov}(x,y) \\
\operatorname{Cov}(x,y) & \operatorname{Var}(y)
\end{pmatrix}
$$

Diagonal entries are each variable's spread; off-diagonals say whether they rise and
fall together. Picture a cloud of data points: $\Sigma$ describes the ellipse enclosing
it.

Its **eigenvectors are the ellipse's axes** and its **eigenvalues the spread along
each**. The largest eigenvalue's eigenvector is the direction of greatest joint
variation. That is exactly Principal Component Analysis — and it is why PCA is an
eigenvalue problem rather than a separate technique.

Two readings worth carrying forward:

- A **large** eigenvalue means much variance along that direction — in an estimation
  context, large uncertainty.
- A **near-zero** eigenvalue means the data are essentially flat there — the variables
  are redundant in that combination.

## 4. Where this shows up in MODA

[Dynamical Bayesian Inference](../algorithms/dynamical-bayesian-inference.md) estimates
the coefficients of a coupling function. It tracks not just a best estimate
$\mathbf{c}$ but a full covariance matrix $\Xi$ describing how uncertain that estimate
is, and in which directions.

The **propagation step** in `bayes_main.m` is where the eigen-picture earns its place.
Between windows the covariance is deliberately inflated:

$$
\Xi_{\text{prior}} = \left(\Xi_{\text{post}}^{-1} + \Sigma_{\text{diff}}\right)^{-1},
\qquad
\Sigma_{\text{diff}} = p^2\,\mathrm{diag}\!\left(\Xi_{\text{post}}^{-1}\right)
$$

Adding to the inverse covariance and re-inverting *increases* the covariance — the
eigenvalues grow, the uncertainty ellipse swells. In plain terms: **time has passed, so
we know less than we did.** The parameter $p$ sets how fast that forgetting happens, and
the eigenstructure decides *where* — directions already well-determined stay relatively
tight, poorly-determined ones loosen fastest.

This is what lets the method track coupling that genuinely changes, instead of either
freezing on the first window's estimate or re-fitting each window from scratch.

## 5. Computing eigenvalues numerically

The characteristic-polynomial route works by hand for $2\times2$ but is a poor
algorithm: polynomial root-finding is numerically unstable, and tiny coefficient errors
can move roots wildly. For a 50-parameter Bayesian model the polynomial is degree 50 and
the approach is hopeless.

Real implementations — MATLAB's `eig()`, NumPy's `numpy.linalg.eig` — iterate instead,
repeatedly applying similarity transformations that preserve eigenvalues while nudging
the matrix toward triangular form, where the eigenvalues sit on the diagonal and can be
read off. Treat it as a black box, but a well-understood one: it is stable, and for
symmetric matrices such as covariances it is both faster and more accurate.

## Next

[Probability & Bayesian Inference](probability-and-bayesian-inference.md) — priors,
posteriors, and testing whether a result is real or chance.
