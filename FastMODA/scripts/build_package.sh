#!/usr/bin/env bash
# Build the fastmoda library as a pip-installable package and check the wheel.
#
#   bash FastMODA/scripts/build_package.sh            # local Python if there is one, else Docker
#   bash FastMODA/scripts/build_package.sh --docker   # always build inside python:3.11-slim
#
# Writes FastMODA/dist/fastmoda-<version>.tar.gz and fastmoda-<version>-py3-none-any.whl,
# then installs the wheel into a fresh virtual environment (outside the source
# tree, so nothing is imported from it by accident) and runs a smoke test:
# the version, a MODA-faithful wavelet transform of a tone, and the Bayesian
# port's filter design.
#
# Install the result anywhere with
#   pip install FastMODA/dist/fastmoda-<version>-py3-none-any.whl
#   pip install "FastMODA/dist/fastmoda-<version>-py3-none-any.whl[ml,mat73]"   # with extras
# or straight from the repository
#   pip install "git+https://github.com/st7ma784/MODA.git#subdirectory=FastMODA"
# Publishing to PyPI is .github/workflows/pypi-publish.yml, which runs this
# script first; the release steps are in docs/developer-guide/index.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"          # FastMODA/

PY=""
if [ "${1:-}" != "--docker" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi

if [ -z "$PY" ]; then
  command -v docker >/dev/null 2>&1 || { echo "Need Python >= 3.10 or Docker." >&2; exit 1; }
  echo "No local Python >= 3.10; building in python:3.11-slim"
  # Copied in rather than bind-mounted: bind mounts are not always shared
  # (Docker Desktop on Windows), and only the package sources are needed.
  export MSYS_NO_PATHCONV=1
  CID="$(docker create -w /FastMODA python:3.11-slim bash scripts/build_package.sh --in-container)"
  trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
  tar -C "$PROJ/.." --exclude=__pycache__ -cf - \
      FastMODA/pyproject.toml FastMODA/PYPI_README.md FastMODA/LICENSE FastMODA/fastmoda FastMODA/scripts \
    | docker cp - "$CID:/"
  docker start -a "$CID"
  rm -rf "$PROJ/dist"
  DEST="$PROJ/dist"
  command -v cygpath >/dev/null 2>&1 && DEST="$(cygpath -m "$DEST")"   # Windows docker.exe
  docker cp "$CID:/FastMODA/dist" "$DEST"
  echo "copied to $PROJ/dist"
  exit 0
fi
[ "${1:-}" = "--in-container" ] && echo "building in $(python3 -V)"

cd "$PROJ"
rm -rf build dist fastmoda.egg-info
"$PY" -m pip install -q --upgrade pip build
"$PY" -m build --sdist --wheel --outdir dist .

WHEEL="$(ls dist/fastmoda-*-py3-none-any.whl)"
echo "built: $(ls dist)"

# Check the wheel in isolation
VENV="$(mktemp -d)/venv"
"$PY" -m venv "$VENV"
VPY="$VENV/bin/python"; [ -x "$VPY" ] || VPY="$VENV/Scripts/python.exe"
"$VPY" -m pip install -q "$WHEEL"
( cd "$(dirname "$VENV")" && "$VPY" - <<'EOF'
import numpy as np
import fastmoda
from fastmoda.legacy_moda import wt_legacy, wphcoh_legacy
from fastmoda.legacy_bayes import butter_matlab

assert fastmoda.__file__.find("site-packages") >= 0, fastmoda.__file__
fs = 40.0
t = np.arange(2000) / fs
WT, freq = wt_legacy(np.cos(2 * np.pi * 1.5 * t), fs, fmin=0.5, fmax=5, f0=1.0)
peak = freq[np.nanargmax(np.nanmean(np.abs(WT), axis=1))]
assert abs(peak - 1.5) < 0.05, peak
W, _ = wt_legacy(np.cos(2 * np.pi * 1.5 * t), fs, fmin=0.5, fmax=5, f0=1.0, cut_edges=False)
phcoh, _ = wphcoh_legacy(W, W)
assert np.allclose(phcoh, 1), phcoh.min()
b, a = butter_matlab(4, [0.1, 0.2])
assert len(a) == 9 and abs(a[0] - 1) < 1e-12
print(f"fastmoda {fastmoda.__version__} installed from the wheel: "
      f"WT peak at {peak:.3f} Hz for a 1.5 Hz tone; Butterworth design OK")
EOF
)
rm -rf "$(dirname "$VENV")" build fastmoda.egg-info
echo "OK: $WHEEL"
