# A Uniformly Distributed CORDIC RoPE Accelerator for Large Language Models

Artifact for the ISCAS 2027 submission of the same title. This repository holds the
RTL, the synthesis scripts, the numerical verification harness, the perplexity
evaluation, and the raw synthesis reports that every number in the paper is read from.

## What the paper claims, and where to check it

| Claim | Table / section | Files |
|---|---|---|
| Binary UD CORDIC's constant gain `K` folds into attention's `1/sqrt(d_h)` scale for free | §III-C, `tab:gaincomp` | `RTL/binary_ud_cordic_core.v`, `SCRIPTS/rope_gain_comp.tcl`, `RUN/binary_ud/` |
| Constant gain and radix-4 stage merging are mutually exclusive | §II-B | `VERIFY/csd_verdict.py`, `VERIFY/table1_regen.py` |
| Fixed-point RoPE saturates on real activations; block floating point is a functional requirement | §III-B, §IV-A | `KAGGLE/rope_hw.py`, `RTL/rope_bfp_norm.v`, `VERIFY/block_granularity.py` |
| The precision requirement is set by the RoPE base, not model size (`F=11` for Qwen2, `F=8` suffices for LLaMA-2) | §IV-A, `tab:fsweep` | `KAGGLE/eval_perplexity.py`, `KAGGLE/validate_hw.py` |
| Area, power and energy against standard CORDIC, a sin/cos LUT and a multiplier at `W=12` and `W=14` | §IV-B, `tab:perf` | `SCRIPTS/collect_results.py`; reports pending, see below |

## Layout

```
RTL/          synthesisable Verilog: binary UD core, merged radix-4 UD core,
              standard CORDIC, sin/cos LUT and multiplier baselines,
              angle encoder, range reduction, BFP normalisation, gain compensation
CONSTRAINTS/  SDC: 2.0 ns (500 MHz), input/output delays, load
SCRIPTS/      Genus TCL, one per design, plus run_all_synth.sh and the
              report collector that produces the CSV behind tab:perf
RUN/          synthesis reports as produced by Genus, per design and per width
              (see "Coverage of RUN/" below -- this is the earlier campaign)
VERIFY/       Python models and Verilog testbenches; golden vectors;
              table1_regen.py regenerates the angle-error table
KAGGLE/       perplexity evaluation on WikiText-2 against a bit-accurate
              fixed-point RoPE model
PAPER/        LaTeX source (iscas2027_main.tex is the submission)
```

## Reproducing

**Synthesis** (needs Cadence Genus and a `gsclib045` installation):

```bash
./SCRIPTS/run_all_synth.sh          # all 17 runs
./SCRIPTS/run_all_synth.sh 1 2      # the two critical-path runs only
python3 SCRIPTS/collect_results.py  # areas, slack and power into one CSV
```

Runs are ordered in `SYNTHESIS_RUN_LIST.md`; runs 1, 2, 10 and 11 are the ones the
headline table depends on.

**Numerical verification** (Python 3.11+, numpy; Icarus or Xcelium for the testbenches):

```bash
./VERIFY/run_all.sh
python3 VERIFY/table1_regen.py
```

**Perplexity** — see `KAGGLE/KAGGLE_INSTRUCTIONS.md`. Needs `transformers` 5.17.0 and
`torch` 2.10.0, a GPU, and a Hugging Face token supplied through the environment
(never committed; on Kaggle it comes from *Add-ons → Secrets* as `HF_TOKEN`).

## Tool and library versions

- Cadence Genus 21.10-p002
- Cadence GPDK045, `gsclib045` with the hvt/lvt kits
- Corner `fast_vdd1v2` at PVT_1P32V_0C (1.32 V, 0 °C), which is optimistic relative
  to nominal; every design in the comparison is synthesised at the same corner
- 2.0 ns clock period (500 MHz)

## Coverage of `RUN/`

`RUN/` holds the **earlier synthesis campaign**: the merged UD datapath swept over
`W=6..14`, and the multiplier, sin/cos LUT and standard CORDIC baselines at a
subset of widths. Where it overlaps the paper it agrees exactly, for instance
`mult_rope/W=14` at 4184.712 um^2 against the 4185 reported, `sin_cos_rope/W=14` at
5375.214 against 5375, and `csd_ud/14` at 3466.512 against the 3467 uncompensated
row of `tab:gaincomp`.

It does **not** yet contain the later campaign of 17 runs: the Binary UD rotator at
`W=12` and `W=14`, the H1/H2 gain-compensation runs, and the BFP block-granularity
sweep. Those reports are being added; until then the parts of `tab:perf` that rest
on them are not reproducible from this repository alone.

## A note on what is not here

Genus design databases (`*.db`), formal-verification maps and mapped netlists are
excluded. They are proprietary-format binaries that embed characterised standard-cell
library data from GPDK045, so they are not ours to redistribute, and they are
unreadable without a Genus licence. The human-readable reports under `RUN/` carry
every quantity the paper reports: cell area, slack, power and the gate census.

Report headers contain the absolute working directory of the lab machine the runs
were performed on. They are left intact so the reports stand as unedited tool output.

## Licence

MIT, see `LICENSE`.
