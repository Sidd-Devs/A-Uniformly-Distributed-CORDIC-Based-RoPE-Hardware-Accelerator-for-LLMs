#!/usr/bin/env bash
# Pull area / slack / power out of every completed run into one CSV.
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "run,area_um2,slack_ps,power_mW"
for d in "$HERE"/../RUN_ISCAS/*/; do
  tag=$(basename "$d")
  f=$(ls "$d"/OUTPUT/reports_*/*/opt/final.rpt 2>/dev/null | head -1)
  p=$(ls "$d"/OUTPUT/reports_*/*/opt/power.rpt 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  a=$(grep "^Total Cell Area:" "$f" | tr -d ',' | awk '{print $NF}')
  s=$(grep "^Slack (ps):"      "$f" | tr -d ',' | awk '{print $NF}')
  w=$(grep -i "Subtotal" "$p" 2>/dev/null | tail -1 | awk '{printf "%.4f",$5*1000}')
  echo "$tag,${a:-},${s:-},${w:-}"
done
