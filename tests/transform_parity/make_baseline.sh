#!/usr/bin/env bash
# Extract the pre-vectorisation wt.m / wft.m that this suite compares against.
#
# BASELINE_COMMIT is the commit immediately BEFORE the vectorised per-scale
# transform landed. Both forms are produced:
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
    # grep -c, not -q: under pipefail an early-exiting grep -q gets git show
    # killed by SIGPIPE, the pipeline "fails", and the wrong commit is taken
    if ! git -C "$REPO" show "$c:$WT" 2>/dev/null | grep -c useVectorized >/dev/null; then
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

# The other rewritten routines, renamed <name>_base so each can be swapped in
# on its own (swap_run.m). Only the primary function is renamed; whatever it
# calls by name resolves to the current code, so a difference is that file's.
mkdir -p "$OUT/swap"
for p in allguis/guis/coherence/Functions/wphcoh.m \
         allguis/guis/filtering/Functions/ecurve.m \
         allguis/guis/bispectrum/Functions/bispecWavNew.m \
         allguis/guis/bayesian/Functions/CFprint.m \
         allguis/guis/bayesian/Functions/dirc.m \
         allguis/guis/bayesian/Functions/bayes_main.m \
         allguis/guis/bayesian/Functions/full_bayesian.m; do
  n=$(basename "$p" .m)
  git -C "$REPO" show "$BASELINE_COMMIT:$p" \
    | sed -E "0,/(^[[:space:]]*function[^=]*=[[:space:]]*)$n([[:space:]]*\()/s//\1${n}_base\2/" \
    > "$OUT/swap/${n}_base.m"
  grep -Eq "^[[:space:]]*function[^=]*=[[:space:]]*${n}_base[[:space:]]*\(" "$OUT/swap/${n}_base.m" \
    || { echo "ERROR: $n rename failed" >&2; exit 1; }
done

echo "baseline ($BASELINE_COMMIT) extracted to $OUT"
