# A Uniformly Distributed CORDIC RoPE Accelerator for Large Language Models

RTL, synthesis scripts, numerical verification and perplexity evaluation for a
multiplier-free rotary position embedding (RoPE) unit built on uniformly
distributed (UD) CORDIC.

S. Deore, P. Shetty and M. Rao, IIIT Bangalore.

## Results and where to find them

| Result | Paper | Files |
|---|---|---|
| Binary UD CORDIC's constant gain `K` folds into attention's `1/sqrt(d_h)` scale at zero cost | Sec. III-C, Table I | `RTL/binary_ud_cordic_core.v`, `RTL/rope_gain_comp.v` |
| Constant gain and radix-4 stage merging cannot be obtained together | Sec. III-C, Table I | `VERIFY/csd_verdict.py`, `SCRIPTS/gen/run_01..04_*` |
| Reading micro-rotation directions straight from the binary angle mis-encodes the leading stages (2.20° RMS); the angle encoder corrects this to 0.094° for 136 µm² | Sec. III-A | `RTL/rope_angle_encoder.v`, `VERIFY/encoder_design.py`, `VERIFY/table1_regen.py`, `SCRIPTS/gen/run_05_*`, `run_06_*` |
| Fixed-point RoPE saturates on real activations, so block floating point (BFP) is required; blocks of 8 pairs beat blocks of 64 | Sec. III-B | `RTL/rope_bfp_norm.v`, `VERIFY/block_granularity.py`, `KAGGLE/rope_hw.py`, `SCRIPTS/gen/run_07..09_*`, `run_16_*`, `run_17_*` |
| Required precision follows the rotary base, not model size (`F=11` recommended; LLaMA-2-7B stays within 0.32% at `F=8`) | Sec. IV-A, Table II | `KAGGLE/eval_perplexity.py`, `KAGGLE/validate_hw.py` |
| Area, power and energy against standard CORDIC, a sin/cos LUT and a multiplier at `W=12` and `W=14` | Sec. IV-B, Table III | `SCRIPTS/gen/run_10..15_*`, `SCRIPTS/collect_results.py`, `RUN/` |

## Layout

```
RTL/          synthesisable Verilog: binary UD core, merged radix-4 UD core,
              standard CORDIC, sin/cos LUT and multiplier baselines,
              angle encoder with range reduction, BFP normalisation,
              gain compensation
CONSTRAINTS/  SDC: 2.0 ns (500 MHz), input/output delays, load
SCRIPTS/      Genus TCL, run_all_synth.sh, and collect_results.py, which
              gathers area, slack and power into one CSV
SCRIPTS/gen/  the 17 generated synthesis runs (top-level wrapper + TCL each)
RUN/          Genus reports per design and word length
VERIFY/       Python models, Verilog testbenches and golden vectors
KAGGLE/       perplexity evaluation on WikiText-2 with a bit-accurate model
              of the synthesised datapath
PAPER/        LaTeX source
```

## Synthesis runs

| Runs | Design |
|---|---|
| 1-4 | Gain compensation H1 / H2 for the merged UD datapath, `W=12` and `W=14` |
| 5-6 | Angle encoder and range reduction |
| 7-9 | BFP normaliser (`D=64`, `W=12` and `W=14`) and denormaliser |
| 10-11 | Multiplier and sin/cos LUT baselines, `W=12` |
| 12-13 | Standard CORDIC, `W=12` and `W=14` |
| 14-15 | Binary UD rotator, `W=12` and `W=14` |
| 16-17 | BFP normaliser, finer block sizes, `W=14` |

`RUN/` holds Genus reports from an earlier sweep: the merged UD datapath at
`W=6..14`, and the multiplier, sin/cos LUT, standard CORDIC and Binary UD
designs at several word lengths. Where they overlap the paper they agree
with it: for example `mult_rope/W=14` gives 4184.712 µm² (paper: 4185),
`sin_cos_rope/W=14` gives 5375.214 µm² (5375) and `csd_ud/14` gives
3466.512 µm² (3467, uncompensated row of Table I). The remaining Table III
rows are produced by runs 10-15 above.

## Reproducing

**Synthesis** (Cadence Genus and a GPDK045 `gsclib045` installation):

```bash
./SCRIPTS/run_all_synth.sh          # all 17 runs
./SCRIPTS/run_all_synth.sh 14 15    # selected runs only, e.g. the Binary UD rotator
python3 SCRIPTS/collect_results.py  # area, slack and power into one CSV
```

**Numerical verification** (Python 3.11+, numpy; Icarus or Xcelium for the
testbenches):

```bash
./VERIFY/run_all.sh
python3 VERIFY/table1_regen.py
```

**Perplexity**: see `KAGGLE/KAGGLE_INSTRUCTIONS.md`. Needs `transformers`
5.17.0 and `torch` 2.10.0, a GPU, and a Hugging Face token supplied through
the environment (on Kaggle: *Add-ons → Secrets* as `HF_TOKEN`).

## Tools and library

- Cadence Genus 21.10-p002
- Cadence GPDK045, `gsclib045` with the hvt/lvt cells
- Corner `fast_vdd1v2` (1.32 V, 0 °C); every design and baseline uses the
  same corner
- 2.0 ns clock period (500 MHz)

## Not included

Genus design databases (`*.db`), formal-verification maps and mapped
netlists are not included. They embed characterised GPDK045 library data,
which is licensed and cannot be redistributed. The text reports carry every
quantity used in the paper: cell area, slack, power and gate counts. Report
headers keep the original working directory of the synthesis machine,
unedited.

## Citation

```bibtex
@inproceedings{deore2027udrope,
  author    = {Deore, Siddhant and Shetty, Pratham and Rao, Madhav},
  title     = {A Uniformly Distributed {CORDIC} {RoPE} Accelerator for Large Language Models},
  booktitle = {Proc. IEEE International Symposium on Circuits and Systems (ISCAS)},
  year      = {2027},
  note      = {Under review}
}
```

## Licence

MIT, see `LICENSE`.
