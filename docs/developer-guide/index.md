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

## See also

- [Refactor Notes](refactor-notes.md) — background on recent MATLAB engine
  vectorization work and the verification methodology used for it.
