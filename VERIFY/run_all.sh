#!/usr/bin/env bash
# Reproduce the T1.1 / T1.2 / T1.7 verification results.
# Requires: python3, iverilog
set -e
cd "$(dirname "$0")"

echo "### T1.1  CSD datapath diagnosis"
python3 csd_verify.py
python3 csd_search.py
python3 csd_control_test.py
python3 range_check.py

echo; echo "### T1.7  encoder design-space exploration"
python3 encoder_design.py
python3 encoder_hw.py

echo; echo "### T1.7  RTL equivalence (encoder vs bit-exact model)"
python3 gen_golden.py
iverilog -g2005 -o tb_enc.vvp tb_angle_encoder.v ../RTL/rope_angle_encoder.v
vvp tb_enc.vvp

echo; echo "### T1.5  end-to-end encoder + binary_ud_cordic_core"
for W in 16 12 9; do
  echo "=== W=$W ==="
  sed "s/parameter W      = 16;/parameter W      = $W;/" tb_e2e.v > .tb_e2e_w.v
  iverilog -g2005 -o .e2e.vvp .tb_e2e_w.v ../RTL/rope_angle_encoder.v \
      ../RTL/binary_ud_cordic_core.v
  vvp .e2e.vvp | head -1
  python3 analyze_e2e.py $W
done
rm -f .tb_e2e_w.v .e2e.vvp tb_enc.vvp e2e_out.txt
echo; echo "done."

echo; echo "### T1.9  CSD encoder investigation"
python3 csd_encoder.py
python3 csd_verdict.py
python3 csd_rescue.py

echo; echo "### T1.9 correction: matched-precision comparison"
python3 csd_fair.py

echo; echo "### T1.8  Table I regeneration"
python3 table1_regen.py

echo; echo "### T3.4 mechanism: angle loss vs RoPE base"
python3 angle_loss.py

echo; echo "### T2.2  latency / throughput / energy"
python3 perf_metrics.py

echo; echo "### T2.10  BFP normaliser"
iverilog -g2005 -o /tmp/bfp.vvp tb_bfp.v ../RTL/rope_bfp_norm.v && vvp /tmp/bfp.vvp | tail -4 && rm -f /tmp/bfp.vvp

echo; echo "### T1.6  pair-module testbenches"
bash run_pair_tb.sh

echo; echo "### T2.11  block granularity"
python3 block_granularity.py 2>/dev/null | tail -12

echo; echo "### plan consistency"
python3 check_plan.py
