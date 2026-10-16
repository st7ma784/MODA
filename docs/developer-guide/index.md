# Developer Guide

This guide is aimed at developers wishing to modify or contribute to MODA.

## Additional requirements

In addition to the [requirements](../getting-started/installation.md#requirements)
listed for users, you'll also need to
[install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Downloading MODA

- Open a terminal in a desired folder and run `git clone https://github.com/luphysics/MODA.git`.
- The code will download as a folder named `MODA`.

## Installing Git hooks

Git hooks are used to automatically perform tasks when a commit is made. MODA uses
`doctoc` to add a table of contents to markdown files, and `mkdocs` for this
documentation site.

Commit your current work, if there are changes. Then open a terminal in the `MODA`
folder and run:

```bash
pip install pre-commit --user   # Installs the pre-commit tool.
python -m pre-commit install    # Adds the Git hooks to the repository.
```

On Windows, also run `git config core.safecrlf false` in the `MODA` folder. This
prevents a circular problem where Git cannot commit because it converts line endings to
CRLF but `doctoc` converts line endings back to LF.

Once installed, the hooks automatically run every time a commit changes relevant files.

!!! warning
    When a pre-commit hook changes files, you'll need to `git add` and commit again.

## Building the documentation locally

This site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material theme](https://squidfunk.github.io/mkdocs-material/). To preview it locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000`. The site is deployed automatically to GitHub Pages
on every push to `main` — see `.github/workflows/docs.yml`.

## Releasing the Python package

The `fastmoda` library (see [The Python Library](../api-and-ml/python-library.md)) is
published to PyPI by `.github/workflows/pypi-publish.yml`. It authenticates with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so the repository
stores no PyPI token.

**One-off setup**, done once per index by the PyPI project owner:

1. Make an account on [pypi.org](https://pypi.org) and on
   [test.pypi.org](https://test.pypi.org), with two-factor authentication.
2. On each index, open *Your account → Publishing → Add a new pending publisher* and
   enter: project `fastmoda`, owner `st7ma784`, repository `MODA`, workflow
   `pypi-publish.yml`. The environment is `pypi` on PyPI and `testpypi` on TestPyPI.
   The first upload creates the project.
3. In the GitHub repository, go to *Settings → Environments* and create `pypi` and
   `testpypi`. Add yourself as a required reviewer on `pypi` if releases should wait
   for your approval.

**Each release:**

1. Bump `__version__` in `FastMODA/fastmoda/__init__.py`. The scheme is the date,
   e.g. `2026.10.16`, with `.1`, `.2` for a second release that day. PyPI never
   accepts the same version twice.
2. Check that the package builds and works:
   `bash FastMODA/scripts/build_package.sh`. It runs locally, or in Docker if there
   is no Python.
3. Optional dry run: *Actions → Publish FastMODA to PyPI → Run workflow*, with
   target `testpypi`. Then check
   [test.pypi.org/project/fastmoda](https://test.pypi.org/project/fastmoda/) and
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fastmoda`.
4. Commit, then publish a GitHub release tagged `fastmoda-v<version>`, e.g.
   `fastmoda-v2026.10.16`. The workflow checks that the tag matches `__version__`,
   builds, smoke-tests and uploads.

The PyPI page is `FastMODA/PYPI_README.md`, not `FastMODA/README.md`. Its links must
be absolute URLs, because relative links don't resolve on PyPI.

## See also

- [Refactor Notes](refactor-notes.md) — background on recent MATLAB engine
  vectorization work and the verification methodology used for it.
