# Running the perplexity sweeps on Kaggle

Everything here has been smoke-tested locally against a real (tiny) LLaMA, and
the hardware model is verified bit-exact against the iverilog RTL. What Kaggle
adds is a GPU big enough for 7B models.

**Send back:** the `results_*.json` files, plus the self-test block from stdout.
That is all I need.

---

## 0. Before you start — the one thing that matters

Every run prints a **self-test** first:

```
self-test — relative vector error of the rotation
  config                     rel err   (0 means the patch is NOT live)
  fp16_baseline              0.00000   (baseline, unpatched)
  binaryUD_N10_F14           0.00165
  binaryUD_N10_F7            0.03778
```

If a non-baseline config shows `0.00000`, the RoPE patch did not attach and the
perplexity numbers are meaningless — stop and tell me the model name. This
happens when a model family defines `apply_rotary_pos_emb` somewhere unusual
(the script raises a clear error in the obvious cases, but `trust_remote_code`
models can hide it).

Expected self-test values, so you can eyeball them:

| config | rel err |
|---|---|
| binaryUD_N10_F14 | 0.0017 |
| binaryUD_N10_F9 | 0.0092 |
| binaryUD_N10_F7 | 0.0378 |
| binaryUD_N10_F4 | 0.3157 |
| csd_N5_nocomp | 0.2490 |
| csd_N5_H1 | 0.0253 |
| csd_N5_H2 | 0.0019 |

---

## 1. Notebook setup

- **Accelerator:** `GPU T4 x2` (preferred — 2×16 GB, a 7B model in fp16 needs
  ~14 GB and `device_map="auto"` will split it). `GPU P100` also works for 7B
  but leaves little headroom at 4k+ context.
- **Internet:** ON (needed for the model and dataset downloads).
- **Persistence:** files only.

Cell 1 — dependencies. Kaggle usually has recent `transformers`, but pin so the
run is reproducible and record the version (Reviewer 4 asked for it):

```python
!pip -q install -U transformers datasets accelerate
import transformers, torch
print(transformers.__version__, torch.__version__, torch.cuda.device_count())
```

Cell 2 — get the two scripts in. Easiest is to upload `rope_hw.py` and
`eval_perplexity.py` as a Kaggle Dataset, then:

```python
!cp /kaggle/input/<your-dataset-name>/rope_hw.py .
!cp /kaggle/input/<your-dataset-name>/eval_perplexity.py .
```

(Or paste each file into its own cell with `%%writefile rope_hw.py`.)

Cell 3 — confirm the model math before spending GPU time:

```python
!python eval_perplexity.py --model dummy --configs sweep_F --selftest_only
!python eval_perplexity.py --model dummy --configs csd_gain --selftest_only
```

This needs no GPU and no model download. Check it against the table above.

---

## 2. Which models

Open, no token needed — **start with these**:

| model | id | notes |
|---|---|---|
| Qwen2-7B | `Qwen/Qwen2-7B` | the Table IV outlier; rope base 1e6, most interesting |
| Phi-3-mini | `microsoft/Phi-3-mini-4k-instruct` | small and fast, good first run |
| Falcon-7B | `tiiuae/falcon-7b` | may need `trust_remote_code` |

Gated — need a fresh HF token with the licence accepted on the model page
(your old tokens are revoked, so make a new read-only one and put it in Kaggle
*Add-ons → Secrets* as `HF_TOKEN`):

```python
from kaggle_secrets import UserSecretsClient
import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

| model | id |
|---|---|
| LLaMA-2 7B | `meta-llama/Llama-2-7b-hf` |
| LLaMA-2 13B | `meta-llama/Llama-2-13b-hf` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` |
| Gemma-2 9B | `google/gemma-2-9b` |

**Priority if time is short:** Qwen2-7B and LLaMA-2-7B cover the two extremes
(rope base 1e6 vs 1e4) and that contrast is what explains the Table IV outlier.

---

## 3. The runs

Do a short trial first — 50k tokens, a few minutes, confirms the plumbing:

```python
!python eval_perplexity.py --model Qwen/Qwen2-7B --configs sweep_F \
    --max_tokens 50000 --out trial.json
```

Then the three real sweeps. Each writes JSON incrementally, so a timeout still
leaves usable partial results.

**A — precision sweep** (replaces the paper's Table IV; ~10 configs):

```python
!python eval_perplexity.py --model Qwen/Qwen2-7B --configs sweep_F \
    --seq_len 2048 --out results_sweepF_qwen2.json
```

**B — CSD gain compensation** (settles T3.5: does the LLM tolerate H=1's 7.3%
residual gain spread? This decides whether CSD keeps its 24% area advantage):

```python
!python eval_perplexity.py --model Qwen/Qwen2-7B --configs csd_gain \
    --seq_len 2048 --out results_csdgain_qwen2.json
```

**C — long context** (the gap Reviewer 4 called out; RoPE angle grows with
position so error is position-dependent and 2k perplexity cannot show it):

```python
!python eval_perplexity.py --model Qwen/Qwen2-7B --configs longctx \
    --long_ctx 1024,2048,4096,8192 --out results_longctx_qwen2.json
```

Drop 8192 if you hit OOM — 4096 is enough to show a trend.

Repeat A–C per model, changing `--out`.

---

## 4. Runtime and memory

- WikiText-2 test is ~280–340k tokens depending on tokenizer. At ctx 2048
  non-overlapping that is ~150 windows per config.
- Sweep A on a 7B model, T4 x2: roughly **30–60 min** for all 10 configs. The
  hardware rotation adds maybe 2–3× over the fp16 baseline, but the digit
  tables are computed once per context length, not per layer.
- Long-context runs cost more per window; 8192 on a 7B may OOM on a single T4.
- If a run dies, lower `--max_tokens` (e.g. 150000) and note it — the script
  records `eval_tokens` so partial runs are still comparable **to each other at
  the same token count**, just not to full-split numbers.

---

## 5. What comes back, and why each field is there

`results_*.json` has a `meta` block that answers Reviewer 4's list item by
item — FP baseline, token count, sequence length, stride, batch size, KV-cache,
what was quantized, and the harness:

```json
"meta": {
  "model": "...", "dtype": "float16",
  "transformers_version": "...", "torch_version": "...",
  "dataset": "wikitext-2-raw-v1 / test split, whole split joined by \\n\\n",
  "total_tokens": 336000, "batch_size": 1, "kv_cache": false,
  "quantized_scope": "RoPE rotation only; weights/activations/Q/K remain float16",
  "loss_protocol": "sliding window, loss on the last `stride` tokens only",
  "harness": "custom (this script); NOT lm-eval-harness"
}
```

and one row per config with `perplexity`, `eval_tokens`, `seq_len`, `stride`.

Two honesty notes I will carry into the paper: this is a **custom harness, not
lm-eval-harness**, and `use_cache=False` throughout. Both are stated rather
than glossed.

---

## 6. If something breaks

| symptom | cause | fix |
|---|---|---|
| self-test all zeros | patch not attached | send me the model id |
| `no apply_rotary_pos_emb` | unusual model family | send me the id; needs a bespoke patch |
| OOM at load | 7B on one 16 GB GPU | use T4 x2, confirm `device_map="auto"` |
| OOM mid-run | context too long | lower `--long_ctx` |
| very slow | fp64 fallback on GPU | expected on T4; tell me and I will trim the angle path |
| gated repo 401 | token missing or licence not accepted | accept on the model page, re-create the token |
