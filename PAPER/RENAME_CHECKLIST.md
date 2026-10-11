# T1.3 Option A — rename checklist

Decision (2026-09-14): the design is **not** CSD-based, so rename it to match
the hardware rather than change the hardware to match the name.

**Why:** `csd_control.v` implements radix-4 modified Booth recoding, and the
digit set `csd_ud_stage.v` realises is {0, ±1, ±3/2} — neither canonical signed
digit (which needs digits in {0,±1} with no two non-zero adjacent) nor Booth
(which yields {0,±1,±2}). Section II-D's merging argument currently rests on
CSD's non-adjacency property, which does not describe this datapath.

Option B (implement true CSD) was rejected: it invalidates the whole synthesis
campaign three weeks from deadline, and T1.2 measured ±3/2 as *more* accurate
than ±2 (0.0595° vs 0.0648° RMS at N=5) with identical reachable granularity.

---

## Naming

| old | new |
|---|---|
| CSD UD-CORDIC | merged UD-CORDIC (M-UD) |
| CSD UD | M-UD |
| CSD UD (Ext.) | M-UD, $N$ stages |
| "CSD encoding is applied to θ" | "adjacent radix-2 positions are merged into one radix-4 stage" |

Keep the term **CSD** only where it genuinely means canonical signed digit —
i.e. in the one sentence of §II-D that explains what the digit set is *not*,
and in any citation to prior work that does use CSD.

## Manuscript edits

- [ ] **Title** — "A Uniformly Distributed CORDIC-Based RoPE Hardware Accelerator for LLMs". Drop "CSD-Based". Also fixes the "Rope" → "RoPE" capitalisation (R1-1).
- [ ] **Abstract** — remove "Canonical Signed Digit (CSD)" and "CSD UD-CORDIC"; re-describe as the merged radix-4 variant. The abstract needs rewriting anyway (T4.8) once the operating point and architecture decision land.
- [ ] **Index terms** — drop CSD if present.
- [ ] **§I** — the sentence introducing CSD encoding.
- [ ] **§II-D** — replaced wholesale by `background_ud.tex`.
- [ ] **Eq. 8** — replaced by eq:merged: θ ≈ Σ arctan(vᵢ2^−2i) ≈ Σ vᵢ2^−2i, vᵢ ∈ {0,±1,±3/2}. The submitted form had no weight term at all (R4-6).
- [ ] **§III-A** — "leverages the non-adjacent property of Canonical Signed Digits" and "similar to radix-4 Booth encoding" are both wrong; describe the actual digit set.
- [ ] **§IV, Table II / Table III** — row and column labels.
- [ ] **§V conclusion** — "by leveraging Canonical Signed Digit (CSD) encoding, the CSD UD-CORDIC merges consecutive binary stages".
- [ ] **Fig. 1 legend** — "Area (CSD UD)", "Power (CSD UD)", "Perplexity Deg. (CSD)".

## Artifact edits (T5)

Rename for consistency with the paper, or leave the RTL filenames and note the
mapping in the README — either is defensible, but do not leave the paper and
the repo using different names for the same block.

- [ ] `RTL/csd_ud_stage.v` → `merged_ud_stage.v`
- [ ] `RTL/csd_ud_cordic_core.v` → `merged_ud_core.v`
- [ ] `RTL/csd_control.v` → `merged_digit_sel.v` (it is a digit selector, not a CSD recoder)
- [ ] `RTL/rope_csd_pair.v` → `rope_merged_pair.v`
- [ ] `SCRIPTS/csd_ud.tcl`, `RUN/csd_ud/` paths
- [ ] `KAGGLE/rope_hw.py` — `CSDTable`, `CSD_DIGITS`, `_csd_*`, and the `"csd"` variant string
- [ ] `KAGGLE/eval_perplexity.py` — `csd_gain` config-set name, `csd_N5_*` config names
- [ ] `VERIFY/csd_*.py` filenames and the strings they print

**Caution:** renaming the Python identifiers changes the `config` names written
into the results JSON. The Kaggle runs already completed use `csd_N5_nocomp`,
`csd_N5_H1`, `csd_N5_H2`. Either keep those strings stable, or re-run — do not
silently mix naming across result files.

## What does not change

The RTL behaviour, every synthesis number, and every perplexity number. This is
a naming and justification correction, not a design change.
