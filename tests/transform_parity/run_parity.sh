#!/usr/bin/env bash
# Numerical parity of the vectorised wt.m / wft.m against the pre-vectorisation
# implementation. See README.md.
#
#   bash tests/transform_parity/run_parity.sh            # main + downstream + bispectrum
#   bash tests/transform_parity/run_parity.sh --blocks   # additionally force blk=1 / blk=7
#
# Requires a local MATLAB. Override with MATLAB=/path/to/matlab.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BASE="$HERE/.baseline"
MATLAB="${MATLAB:-/c/Program Files/MATLAB/R2026a/bin/matlab}"

[ -x "$MATLAB" ] || { echo "MATLAB not found at '$MATLAB'; set MATLAB=..." >&2; exit 1; }

bash "$HERE/make_baseline.sh" "$BASE"

run() { "$MATLAB" -batch "$1"; }

echo
echo "### 1/3  transform parity (full case matrix)"
run "addpath('$HERE'); parity_run('$REPO','$BASE/ref','','PRODUCTION vs BASELINE')"

echo
echo "### 2/3  downstream: ridge extraction + coherence"
run "addpath('$HERE'); downstream_run('$REPO','$BASE/ref')"

echo
echo "### 3/3  bispectrum"
run "addpath('$HERE'); bispec_run('$REPO','$BASE/baseline')"

if [ "${1:-}" = "--blocks" ]; then
  # The blocking loop only runs when NL*SN exceeds the element budget, which
  # short test signals never do — so force small block sizes. Restricted to a
  # subset because blk=1 over the full matrix is very slow.
  PAT='(pre=on\|cut=off\|pad=predictive\|band=default)|(custom-)|(handle-)'
  for n in 1 7; do
    echo
    echo "### forced blk=$n"
    VAR="$BASE/blk$n"; mkdir -p "$VAR"
    for f in wt wft; do
      sed "s|blk=max(1,min(SN,floor(maxElem/max(NL,1))));|blk=max(1,min(SN,$n)); % FORCED|" \
        "$REPO/allguis/guis/tfa/Functions/$f.m" > "$VAR/$f.m"
      grep -q "FORCED" "$VAR/$f.m" || { echo "ERROR: could not force blk in $f.m" >&2; exit 1; }
    done
    run "addpath('$HERE'); parity_run('$REPO','$BASE/ref','$VAR','FORCED blk=$n','$PAT')"
  done
fi

echo
echo "done."
