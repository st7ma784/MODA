# MODA

[![DOI](https://zenodo.org/badge/194114858.svg)](https://zenodo.org/badge/latestdoi/194114858)

**MODA** (Multiscale Oscillatory Dynamics Analysis) is a numerical toolbox for analysing
real-life time-series that are assumed to be the output of some *a priori* unknown
non-autonomous dynamical system, and deriving important properties about that system from
the time-series alone.

It was developed by the [Nonlinear & Biomedical Physics group](https://www.lancaster.ac.uk/physics/research/experimental-condensed-matter/nonlinear-and-biomedical-physics/)
at [Lancaster University](https://www.lancaster.ac.uk/physics/) and the Nonlinear Dynamics
and Synergetics Group at the Faculty of Electrical Engineering, University of Ljubljana,
Slovenia, under the supervision of Aneta Stefanovska.

!!! note
    A Python implementation of MODA, [PyMODA](https://github.com/luphysics/PyMODA), is
    also in development and does not require a MATLAB license.

## What MODA does

MODA includes methods for analysing recordings of a single signal over time, and for
analysing sets of recordings of multiple signals over time. In particular, it has tools
for analysing bivariate time-series — the simultaneous recordings of two different
signals — with a view to examining possible connections between them.

Two ways to use MODA:

- **The MATLAB desktop app** (`MODA.m`) — a windowed application with five analysis
  modules (Time-Frequency Analysis, Wavelet Phase Coherence, Ridge Extraction &
  Filtering, Wavelet Bispectrum, Dynamical Bayesian Inference). See
  [Using MODA → The Desktop App](using-moda/desktop-app.md).
- **FastMODA**, a Flask-based web app and REST API exposing the same underlying
  algorithms for browser-based and programmatic use, including machine-learning feature
  extraction. See [Using MODA → The Web App](using-moda/web-app.md) and
  [API & Machine Learning](api-and-ml/rest-api-reference.md).

## Where to go next

| I want to... | Go to |
|---|---|
| Install MODA and run my first analysis | [Getting Started](getting-started/installation.md) |
| Understand what an algorithm actually computes | [Algorithms](algorithms/time-frequency-analysis.md) |
| Brush up on the maths behind an algorithm | [Maths Primer](maths-primer/index.md) |
| Call MODA/FastMODA from code, or train a classifier on its output | [API & Machine Learning](api-and-ml/rest-api-reference.md) |
| Contribute to MODA itself | [Developer Guide](developer-guide/index.md) |

## Citing MODA

If you use MODA in your research, please see [Reference → Citations](reference/citations.md)
for the relevant papers to cite for each method.
