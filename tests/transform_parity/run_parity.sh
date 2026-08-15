#!/usr/bin/env bash
# Numerical parity of the vectorised wt.m / wft.m against the pre-vectorisation
# implementation, plus the checks in swap_run.m. See README.md.
#
#   bash tests/transform_parity/run_parity.sh            # main + downstream + bispectrum + swaps
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

# MATLAB on Windows needs C:/... paths, not Git Bash's /c/...
mpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
H="$(mpath "$HERE")"; R="$(mpath "$REPO")"; B="$(mpath "$BASE")"

run() { "$MATLAB" -batch "$1"; }

echo
echo "### 1/4  transform parity (full case matrix)"
run "addpath('$H'); parity_run('$R','$B/ref','','PRODUCTION vs BASELINE')"

echo
echo "### 2/4  downstream: ridge extraction + coherence"
run "addpath('$H'); downstream_run('$R','$B/ref')"

echo
echo "### 3/4  bispectrum"
run "addpath('$H'); bispec_run('$R','$B/baseline')"

echo
echo "### 4/4  each rewritten routine against upstream's, numeric nv/fstep, overflow retry, twf-only wavelet, GPU"
run "addpath('$H'); swap_run('$R','$B')"

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
    run "addpath('$H'); parity_run('$R','$B/ref','$(mpath "$VAR")','FORCED blk=$n','$PAT')"
  done
fi

echo
echo "done."
