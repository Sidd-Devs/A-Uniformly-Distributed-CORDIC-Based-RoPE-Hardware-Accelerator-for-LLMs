#!/usr/bin/env bash
# Run every ISCAS-revision synthesis job. One directory per run so OUTPUT/
# trees never collide. Safe to re-run: completed runs are skipped.
#
#   bash run_all_genus.sh            # all 17
#   bash run_all_genus.sh 01 02 05 06 07 08 15   # the short list
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GEN="$HERE/gen"
RUNROOT="$HERE/../RUN_ISCAS"
mkdir -p "$RUNROOT"

jobs=()
for f in "$GEN"/run_*.tcl; do jobs+=("$(basename "$f" .tcl | sed 's/^run_//')"); done

want=("$@")
for tag in "${jobs[@]}"; do
  if [ ${#want[@]} -gt 0 ]; then
    hit=0; for w in "${want[@]}"; do [[ "$tag" == "$w"* ]] && hit=1; done
    [ $hit -eq 0 ] && continue
  fi
  d="$RUNROOT/$tag"
  if [ -f "$d/.done" ]; then echo "skip $tag (already done)"; continue; fi
  echo "=================================================================="
  echo "  $tag"
  echo "=================================================================="
  mkdir -p "$d"; cd "$d"
  genus -batch -f "$GEN/run_$tag.tcl" 2>&1 | tee genus.log
  if grep -q "=== DONE" genus.log; then touch .done; else echo "  !! $tag did not complete"; fi
  cd "$HERE"
done

echo; echo "=== summary ==="
for tag in "${jobs[@]}"; do
  d="$RUNROOT/$tag"
  if [ -f "$d/.done" ]; then
    a=$(grep -h "^Total Cell Area:" "$d"/OUTPUT/reports_*/*/opt/final.rpt 2>/dev/null | tr -d ',' | awk '{print $NF}' | head -1)
    printf "  %-24s area=%s\n" "$tag" "${a:-?}"
  else
    printf "  %-24s NOT RUN\n" "$tag"
  fi
done
