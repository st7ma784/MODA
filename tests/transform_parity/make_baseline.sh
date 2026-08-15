#!/usr/bin/env bash
# Extract the pre-vectorisation wt.m / wft.m that this suite compares against.
#
# BASELINE_COMMIT is the commit immediately BEFORE the vectorised per-scale
# transform landed (it landed in a9efd94). Both forms are produced:
#   ref/wt_base.m, ref/wft_base.m  - renamed, so they can be called alongside
#                                    the production wt/wft in one session
#   baseline/wt.m, baseline/wft.m  - original names, for path shadowing (the
#                                    bispectrum chain calls wt() by name)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/.baseline}"

WT=allguis/guis/tfa/Functions/wt.m

# Find the baseline rather than pinning a hash: walk the history of wt.m and
# take the most recent commit whose copy predates the vectorised path. A
# pinned hash would be a commit in the contributor's fork, which does not
# survive the merge into a repository that history never entered.
if [ -z "${BASELINE_COMMIT:-}" ]; then
  for c in $(git -C "$REPO" log --format=%H -- "$WT"); do
    if ! git -C "$REPO" show "$c:$WT" 2>/dev/null | grep -q useVectorized; then
      BASELINE_COMMIT="$c"; break
    fi
  done
fi

if [ -z "${BASELINE_COMMIT:-}" ]; then
  echo "ERROR: no pre-vectorisation commit found in the history of $WT." >&2
  echo "       Set BASELINE_COMMIT=<sha> explicitly, or run in a clone with" >&2
  echo "       enough history (a shallow clone will not have it)." >&2
  exit 1
fi

mkdir -p "$OUT/ref" "$OUT/baseline"

git -C "$REPO" show "$BASELINE_COMMIT:allguis/guis/tfa/Functions/wt.m"  > "$OUT/baseline/wt.m"
git -C "$REPO" show "$BASELINE_COMMIT:allguis/guis/tfa/Functions/wft.m" > "$OUT/baseline/wft.m"

if grep -q useVectorized "$OUT/baseline/wt.m" || grep -q useVectorized "$OUT/baseline/wft.m"; then
  echo "ERROR: $BASELINE_COMMIT already contains the vectorised path - wrong baseline." >&2
  exit 1
fi

# Rename only the primary function; the subfunctions are file-local and
# neither file calls itself recursively, so this is safe.
sed 's/^function \[WT,freq,varargout\] = wt(/function [WT,freq,varargout] = wt_base(/' \
    "$OUT/baseline/wt.m"  > "$OUT/ref/wt_base.m"
sed 's/^function \[WFT,freq,varargout\] = wft(/function [WFT,freq,varargout] = wft_base(/' \
    "$OUT/baseline/wft.m" > "$OUT/ref/wft_base.m"

grep -q "function \[WT,freq,varargout\] = wt_base(" "$OUT/ref/wt_base.m" \
  || { echo "ERROR: wt rename failed" >&2; exit 1; }
grep -q "function \[WFT,freq,varargout\] = wft_base(" "$OUT/ref/wft_base.m" \
  || { echo "ERROR: wft rename failed" >&2; exit 1; }

echo "baseline ($BASELINE_COMMIT) extracted to $OUT"
