#!/usr/bin/env bash
# T1.6 — regenerate golden vectors and run all five pair testbenches.
set -e
cd "$(dirname "$0")"
R=../RTL
python3 gen_pair_vectors.py
echo
iverilog -g2005 -s tb_mult   -o /tmp/tbm.vvp tb_pairs.v $R/rope_mult_pair.v
vvp /tmp/tbm.vvp | grep -E "MISMATCH|tb_"
iverilog -g2005 -s tb_lut    -o /tmp/tbl.vvp tb_pairs.v $R/rope_lut_pair.v
vvp /tmp/tbl.vvp | grep -E "MISMATCH|tb_"
iverilog -g2005 -s tb_binary -o /tmp/tbb.vvp tb_pairs.v \
    $R/rope_binary_ud_pair.v $R/binary_ud_cordic_core.v
vvp /tmp/tbb.vvp | grep -E "MISMATCH|tb_"
iverilog -g2005 -s tb_csd    -o /tmp/tbc.vvp tb_pairs.v \
    $R/rope_csd_pair.v $R/csd_ud_cordic_core.v $R/csd_ud_stage.v $R/csd_control.v
vvp /tmp/tbc.vvp | grep -E "MISMATCH|tb_"
iverilog -g2005 -s tb_cordic -o /tmp/tbs.vvp tb_pairs.v \
    $R/rope_cordic_pair.v $R/cordic_standard_core.v $R/cordic_standard_stage.v
vvp /tmp/tbs.vvp | grep -E "MISMATCH|tb_"
rm -f /tmp/tb[mlbcs].vvp
