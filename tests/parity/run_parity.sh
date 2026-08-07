#!/usr/bin/env bash
# Run the MODA↔FastMODA parity + numerical-equivalence suites inside the
# FastMODA CPU image. No local Python needed — only Docker.
#
#   bash tests/parity/run_parity.sh
#
# Prereq (one-time): build the image from the repo root:
#   docker build -t fastmoda:cpu -f FastMODA/Dockerfile --target base FastMODA
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${FASTMODA_IMAGE:-fastmoda:cpu}"

docker run --rm \
  -v "${REPO}:/repo:ro" \
  "${IMAGE}" bash -c "
    pip install -q pytest >/dev/null 2>&1
    cd /repo/FastMODA   # so 'import fastmoda' resolves against the *mounted*
                        # source (incl. legacy_moda), not the baked-in copy,
                        # and app.py is found by the UI-parity test
    python -m pytest -q -p no:cacheprovider /repo/tests/parity/ \"\$@\"
  " _ "$@"
