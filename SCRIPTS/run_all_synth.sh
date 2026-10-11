#!/usr/bin/env bash
# ============================================================================
# run_all_synth.sh — all 17 Genus runs for the ISCAS 2027 revision.
#
# Usage:
#   export SYN_LIBBASE=/path/to/gsclib045_all_v4.8      # required
#   ./run_all_synth.sh              # everything
#   ./run_all_synth.sh 1 2 10 11        # CRITICAL PATH -- run these first
#   ./run_all_synth.sh 1 2 10 11 5 6 7 8 15   # only these run numbers
#
# Each run gets its own build/NN_name/ directory holding a parameterised copy
# of the RTL plus its reports, so nothing clobbers anything and any single run
# can be repeated without redoing the rest.
#
# On completion:  python3 SCRIPTS/collect_results.py build > results.csv
# ============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RTL="$ROOT/RTL"
CON="$ROOT/CONSTRAINTS"
BUILD="$ROOT/build"
TCL="$HERE/synth_generic.tcl"

: "${SYN_LIBBASE:?set SYN_LIBBASE to the gsclib045_all_v4.8 directory}"
GENUS="${GENUS:-genus}"

SEQ="$CON/rope.sdc"              # has create_clock — sequential designs
COMB="$CON/original_rope.sdc"    # set_max_delay only — combinational designs

mkdir -p "$BUILD"

# run() id name design "rtl files" sdc "PARAM=VAL ..."
run () {
  local id="$1" name="$2" design="$3" files="$4" sdc="$5" params="$6"
  if [ "$#" -gt 6 ]; then shift 6; fi
  if [ -n "${ONLY:-}" ] && ! grep -qw "$id" <<< "$ONLY"; then return 0; fi

  local dir="$BUILD/$(printf '%02d' "$id")_$name"
  echo
  echo "############ run $id : $name ($design)"
  rm -rf "$dir"; mkdir -p "$dir/rtl"

  local srcs=""
  for f in $files; do srcs="$srcs $RTL/$f"; done
  # shellcheck disable=SC2086
  python3 "$HERE/set_params.py" "$dir/rtl" $srcs -- $params || return 1

  local names=""
  for f in $files; do names="$names $(basename "$f")"; done

  ( cd "$dir" && \
    SYN_DESIGN="$design" SYN_RTL="$names" SYN_RTLDIR="$dir/rtl" \
    SYN_SDC="$sdc" SYN_LIBBASE="$SYN_LIBBASE" SYN_OUT="$dir" \
    "$GENUS" -no_gui -f "$TCL" > "$dir/genus.log" 2>&1 )

  if grep -q "^DONE $design" "$dir/genus.log"; then
    echo "  OK  -> $dir/reports/$design/opt/"
  else
    echo "  FAILED — see $dir/genus.log"
    tail -20 "$dir/genus.log" | sed 's/^/    | /'
  fi
}

ONLY="$*"

# ---- CRITICAL PATH: runs 1, 2, 10, 11 --------------------------------------
# 1-2  decide merged-UD vs Binary-UD (gain compensation cost).
# 10-11 put multiplier and LUT at W=12, directly on the fitted area crossover
#       (W~11.5). The paper's central claim is that the measured precision
#       requirement W=13 sits above it; right now that is interpolated from
#       two points. Test before writing, not after. See plan 2.2.
# ----------------------------------------------------------------------------

# ---- P1: gain compensation — decides the headline architecture -------------
run  1 gaincomp_H1_W12 rope_gain_comp "rope_gain_comp.v" "$COMB" "W=12 MODE=1"
run  2 gaincomp_H1_W14 rope_gain_comp "rope_gain_comp.v" "$COMB" "W=14 MODE=1"
run  3 gaincomp_H2_W12 rope_gain_comp "rope_gain_comp.v" "$COMB" "W=12 MODE=2"
run  4 gaincomp_H2_W14 rope_gain_comp "rope_gain_comp.v" "$COMB" "W=14 MODE=2"

# ---- P2: the angle front-end and the BFP normaliser ------------------------
run  5 angle_encoder   rope_angle_encoder "rope_angle_encoder.v" "$COMB" "N=10 FB=12"
run  6 range_reduce    rope_range_reduce  "rope_angle_encoder.v" "$COMB" "PW=16 FB=12"
run  7 bfp_norm_W12    rope_bfp_norm      "rope_bfp_norm.v"      "$COMB" "W=12 D=64 EW=6"
run  8 bfp_norm_W14    rope_bfp_norm      "rope_bfp_norm.v"      "$COMB" "W=14 D=64 EW=6"
run  9 bfp_denorm_W14  rope_bfp_denorm    "rope_bfp_norm.v"      "$COMB" "W=14 D=64 EW=6"

# ---- P3: comparison table at the corrected operating point -----------------
#      NOTE: runs 10 and 11 are CRITICAL PATH -- see the header above.
run 10 mult_W12    rope_mult_pair      "rope_mult_pair.v" "$SEQ" "W=12"
run 11 lut_W12     rope_lut_pair       "rope_lut_pair.v"  "$SEQ" "W=12 ADDR_W=7"
run 12 std_W12_N10 rope_cordic_pair \
       "rope_cordic_pair.v cordic_standard_core.v cordic_standard_stage.v" \
       "$SEQ" "W=12 STAGES=10"
run 13 std_W14_N10 rope_cordic_pair \
       "rope_cordic_pair.v cordic_standard_core.v cordic_standard_stage.v" \
       "$SEQ" "W=14 STAGES=10"
run 14 binud_W12_N10 rope_binary_ud_pair \
       "rope_binary_ud_pair.v binary_ud_cordic_core.v" "$SEQ" "W=12 STAGES=10"
run 15 binud_W14_N10 rope_binary_ud_pair \
       "rope_binary_ud_pair.v binary_ud_cordic_core.v" "$SEQ" "W=14 STAGES=10"

# ---- P4: block-granularity cost (pairs with VERIFY/block_granularity.py) ---
run 16 bfp_norm_W14_D8  rope_bfp_norm "rope_bfp_norm.v" "$COMB" "W=14 D=8  EW=6"
run 17 bfp_norm_W14_D16 rope_bfp_norm "rope_bfp_norm.v" "$COMB" "W=14 D=16 EW=6"

echo
echo "=================================================================="
echo " all requested runs attempted"
echo " collect with:  python3 $HERE/collect_results.py $BUILD"
echo "=================================================================="
