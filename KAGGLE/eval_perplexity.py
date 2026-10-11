"""T1.8 part B — WikiText-2 perplexity with the hardware RoPE model.

Answers Reviewer 4's methodology list explicitly: every quantity they asked
for is recorded in the JSON output and echoed to stdout.

Design notes
------------
* Only the ROTATION is approximated. Weights, activations, Q/K, and the whole
  rest of the model stay in the loaded dtype (fp16). This is stated in the
  output as `quantized_scope`.
* The digit string depends only on (position, dim), never on the data, so it
  is computed once per (seq_len, head_dim, config) and reused across every
  layer, head, and batch element. Without this the sweep is impractically slow.
* Angles are recovered as atan2(sin, cos) from the cos/sin the model itself
  builds, so this works across model families without assuming a rope layout.
* Loss is computed EXPLICITLY here rather than via `labels=`. HF shifts labels
  internally, and the exact semantics move between versions; doing the shift
  ourselves is unambiguous and version-proof.

Usage
    python eval_perplexity.py --model Qwen/Qwen2-7B --configs sweep_F
    python eval_perplexity.py --model Qwen/Qwen2-7B --configs longctx
    python eval_perplexity.py --model Qwen/Qwen2-7B --configs csd_gain
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F
from rope_hw import (encode_binary, rotate_binary, binary_gain,
                     range_reduce, CSDTable)


class Cfg:
    def __init__(self, name, variant, W=9, N=10, FB=12, comp="none",
                 fold_gain=True):
        self.name, self.variant, self.W, self.N, self.FB = name, variant, W, N, FB
        self.comp, self.fold_gain = comp, fold_gain
    def __repr__(self):
        return (f"{self.name}(variant={self.variant},W={self.W},N={self.N},"
                f"comp={self.comp},fold_gain={self.fold_gain})")


_CACHE = {}

def rope_tables(theta, cfg, device):
    """Precompute per-(position,dim) rotation data. theta: [S, H] float64."""
    key = (cfg.name, theta.shape, float(theta[-1, -1]), str(device))
    if key in _CACHE:
        return _CACHE[key]
    quad, res = range_reduce(theta.to(torch.float64))
    if cfg.variant == "binary":
        bits = encode_binary(res, cfg.N, cfg.FB).to(device)
        out = ("binary", quad.to(device), bits, None, None)
    else:
        tab = CSDTable(N=cfg.N, comp=cfg.comp, device="cpu")
        ang, g = tab.apply(res)
        out = ("csd", quad.to(device),
               None, ang.to(device).float(), g.to(device).float())
    _CACHE[key] = out
    return out


def block_exponent(x1, x2):
    """Shared exponent per (batch, head, token) — block floating point.

    Q1.F represents [-2, 2), but real LLM query/key activations run to |q|~130
    and |k|~180 (measured on Qwen2), so a raw fixed-point datapath saturates
    everything. Any real RoPE accelerator therefore normalises before the
    rotator. A power-of-two scale is free in hardware: it is a barrel shift on
    the way in and out, with the exponent carried alongside the mantissa.

    Scaled so |x| < 1, leaving headroom for the CORDIC gain (1.164) so no
    intermediate stage overflows Q1.F.
    """
    amax = torch.maximum(x1.abs(), x2.abs()).amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=1e-12)
    e = torch.floor(torch.log2(amax))
    return torch.exp2(-(e + 1.0))


def hw_rotate_pairs(x1, x2, cfg, tables):
    """x1, x2: [B, Hh, S, H]. tables broadcast over [S, H]."""
    kind, quad, bits, ang, g = tables
    x1 = x1.float(); x2 = x2.float()
    bs = block_exponent(x1, x2)
    x1 = x1 * bs; x2 = x2 * bs

    if kind == "binary":
        sc = 1 << (cfg.W - 2)
        lim = 1 << (cfg.W - 1)
        xq = torch.clamp(torch.round(x1 * sc), -lim, lim - 1).to(torch.int32)
        yq = torch.clamp(torch.round(x2 * sc), -lim, lim - 1).to(torch.int32)
        b = bits.view(1, 1, *bits.shape)
        xo, yo = rotate_binary(xq, yq, b, cfg.N, cfg.W)
        xr = xo.float() / sc
        yr = yo.float() / sc
        if cfg.fold_gain:
            K = binary_gain(cfg.N)
            xr, yr = xr / K, yr / K
    else:
        c, s = torch.cos(ang), torch.sin(ang)
        xr = (x1 * c - x2 * s) * g
        yr = (x1 * s + x2 * c) * g
        sc = 1 << (cfg.W - 2)
        lim = 1 << (cfg.W - 1)
        xr = torch.clamp(torch.round(xr * sc), -lim, lim - 1) / sc
        yr = torch.clamp(torch.round(yr * sc), -lim, lim - 1) / sc

    xr = xr / bs; yr = yr / bs

    q = quad.view(1, 1, *quad.shape)
    xf = torch.where(q == 0, xr, torch.where(q == 1, -yr,
         torch.where(q == 2, -xr, yr)))
    yf = torch.where(q == 0, yr, torch.where(q == 1, xr,
         torch.where(q == 2, -yr, -xr)))
    return xf, yf


def make_patch(cfg):
    def patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, **kw):
        if cos.dim() == 3:
            c = cos.unsqueeze(unsqueeze_dim); s = sin.unsqueeze(unsqueeze_dim)
        else:
            c, s = cos, sin
        H = q.shape[-1] // 2
        c1 = c[..., :H].to(torch.float64)
        s1 = s[..., :H].to(torch.float64)
        theta = torch.atan2(s1, c1)
        while theta.dim() > 2:
            theta = theta[0]
        tabs = rope_tables(theta.cpu(), cfg, q.device)
        q1, q2 = q[..., :H], q[..., H:]
        k1, k2 = k[..., :H], k[..., H:]
        qo1, qo2 = hw_rotate_pairs(q1, q2, cfg, tabs)
        ko1, ko2 = hw_rotate_pairs(k1, k2, cfg, tabs)
        return (torch.cat([qo1, qo2], -1).to(q.dtype),
                torch.cat([ko1, ko2], -1).to(k.dtype))
    return patched


def patch_model(model, cfg):
    """Replace apply_rotary_pos_emb in whatever module defines this model."""
    import sys
    mod = sys.modules[type(model).__module__]
    if not hasattr(mod, "apply_rotary_pos_emb"):
        raise RuntimeError(f"{mod.__name__} has no apply_rotary_pos_emb; "
                           "this model family needs a bespoke patch")
    orig = mod.apply_rotary_pos_emb
    mod.apply_rotary_pos_emb = make_patch(cfg) if cfg else orig
    return mod, orig


# -------------------------------------------------------------- perplexity
@torch.no_grad()
def perplexity(model, ids, seq_len, stride, device, chunk=256):
    """Sliding-window perplexity, loss computed explicitly.

    For each window the model sees tokens [i, end). Predictions come from
    logits[:, :-1] against targets inp[:, 1:] -- the shift is done HERE, not
    by passing `labels=`, because HF also shifts internally and doing both
    silently evaluates P(t+2 | t).

    When stride < seq_len the leading context is scored with -100 so each
    token is counted exactly once.
    """
    nll, ntok = 0.0, 0
    prev_end = 0
    for i in range(0, ids.size(1) - 1, stride):
        end = min(i + seq_len, ids.size(1))
        if end - i < 2:
            break
        inp = ids[:, i:end].to(device)
        out = model(inp, use_cache=False)
        logits = out.logits[:, :-1, :]           # predict position t+1 from t
        targets = inp[:, 1:].clone()

        trg_len = end - max(prev_end, i + 1)     # newly scored tokens
        n_ctx = targets.size(1) - trg_len
        if n_ctx > 0:
            targets[:, :n_ctx] = -100

        # chunked cross-entropy: the vocab is large and the logits tensor is
        # the memory peak at long context
        tot = 0.0
        for a in range(0, logits.size(1), chunk):
            lg = logits[:, a:a + chunk, :].reshape(-1, logits.size(-1)).float()
            tg = targets[:, a:a + chunk].reshape(-1)
            tot += float(F.cross_entropy(lg, tg, reduction="sum",
                                         ignore_index=-100))
        n = int((targets != -100).sum())
        nll += tot
        ntok += n
        prev_end = end
        if end >= ids.size(1):
            break
    return math.exp(nll / ntok), ntok


def build_configs(which):
    if which == "sweep_F":
        return [Cfg("fp16_baseline", "exact")] + [
            Cfg(f"binaryUD_N10_F{W-2}", "binary", W=W, N=10) for W in
            (16, 14, 12, 11, 10, 9, 8, 7, 6)]
    if which == "csd_gain":
        return [Cfg("fp16_baseline", "exact"),
                Cfg("binaryUD_N10_F14", "binary", W=16, N=10),
                Cfg("csd_N5_nocomp", "csd", W=16, N=5, comp="none"),
                Cfg("csd_N5_H1", "csd", W=16, N=5, comp="H1"),
                Cfg("csd_N5_H2", "csd", W=16, N=5, comp="H2")]
    if which == "stages":
        return [Cfg("fp16_baseline", "exact")] + [
            Cfg(f"binaryUD_N{N}_F14", "binary", W=16, N=N)
            for N in (6, 8, 10, 12)]
    if which == "longctx":
        return [Cfg("fp16_baseline", "exact"),
                Cfg("binaryUD_N10_F7", "binary", W=9, N=10),
                Cfg("binaryUD_N10_F14", "binary", W=16, N=10)]
    raise SystemExit(f"unknown config set {which}")


def _find_rope_base(cfg):
    """rope_theta moves around between transformers versions and model families."""
    for attr in ("rope_theta", "rotary_emb_base", "rope_base"):
        v = getattr(cfg, attr, None)
        if v:
            return float(v)
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict):
        for k in ("rope_theta", "theta", "base"):
            if rp.get(k):
                return float(rp[k])
    for sub in ("text_config", "llm_config"):
        c = getattr(cfg, sub, None)
        if c is not None:
            v = _find_rope_base(c)
            if v:
                return v
    return 0.0


def _set_rope_base(cfg, val):
    """Write the RoPE base wherever this transformers version keeps it.

    Setting cfg.rope_theta alone is NOT enough: in transformers 5.x the value
    lives in cfg.rope_parameters["rope_theta"], and assigning a bare attribute
    silently creates one the model never reads. Returns the list of locations
    actually written, so the caller can fail loudly when that list is empty.
    """
    hit = []
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict):
        for k in ("rope_theta", "theta", "base"):
            if k in rp:
                rp[k] = val
                hit.append(f"rope_parameters[{k!r}]")
    for attr in ("rope_theta", "rotary_emb_base", "rope_base"):
        if getattr(cfg, attr, None) is not None:
            setattr(cfg, attr, val)
            hit.append(attr)
    for sub in ("text_config", "llm_config"):
        c = getattr(cfg, sub, None)
        if c is not None:
            hit += _set_rope_base(c, val)
    return hit


def selftest(configs, device="cpu"):
    """Confirm each config actually perturbs the rotation, and by how much."""
    torch.manual_seed(0)
    S, H = 2048, 64
    pos = torch.arange(S, dtype=torch.float64)
    inv = 10000.0 ** (-torch.arange(0, H, dtype=torch.float64) * 2 / (2 * H))
    theta = pos[:, None] * inv[None, :]
    x1 = torch.randn(1, 1, S, H) * 0.3
    x2 = torch.randn(1, 1, S, H) * 0.3
    c, s = torch.cos(theta).float(), torch.sin(theta).float()
    r1, r2 = x1 * c - x2 * s, x1 * s + x2 * c
    denom = torch.sqrt((r1 ** 2 + r2 ** 2).mean())
    print("self-test — relative vector error of the rotation")
    print(f"  {'config':<24}{'rel err':>10}   (0 means the patch is NOT live)")
    for cfg in configs:
        if cfg.variant == "exact":
            print(f"  {cfg.name:<24}{0.0:>10.5f}   (baseline, unpatched)")
            continue
        _CACHE.clear()
        t = rope_tables(theta, cfg, torch.device(device))
        y1, y2 = hw_rotate_pairs(x1, x2, cfg, t)
        rel = float(torch.sqrt(((y1 - r1) ** 2 + (y2 - r2) ** 2).mean()) / denom)
        print(f"  {cfg.name:<24}{rel:>10.5f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="sweep_F",
                    choices=["sweep_F", "csd_gain", "stages", "longctx"])
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=0, help="0 => seq_len")
    ap.add_argument("--max_tokens", type=int, default=0, help="0 => all")
    ap.add_argument("--long_ctx", default="1024,2048,4096,8192")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--fix_rope_scaling", action="store_true",
                    help="clear/translate a 'default' rope_scaling entry; "
                         "needed by Phi-3's repo code, breaks native models")
    ap.add_argument("--no_remote_code", action="store_true",
                    help="load with the native transformers implementation "
                         "instead of the repo's code (needed for Falcon)")
    ap.add_argument("--rope_base", type=float, default=0.0,
                    help="override config.rope_theta (0 = leave as shipped)")
    ap.add_argument("--selftest_only", action="store_true")
    args = ap.parse_args()

    if args.selftest_only:
        selftest(build_configs(args.configs))
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import transformers

    from transformers import AutoConfig
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=not args.no_remote_code)

    trc = not args.no_remote_code
    cfg_obj = AutoConfig.from_pretrained(args.model, trust_remote_code=trc)

    # Some repo-hosted modeling code reads rope_scaling["type"] and accepts only
    # the scaling variants it knows, so a modern {"rope_type": "default"} config
    # makes it raise. Clearing the field fixes those models -- but in
    # transformers 5.x `rope_scaling` and `rope_parameters` are aliases, so
    # clearing it also breaks every NATIVE model, whose code subscripts
    # rope_parameters["rope_type"] unconditionally. The shim is therefore
    # opt-in: pass --fix_rope_scaling only for models that need it (Phi-3).
    if args.fix_rope_scaling:
        rs = getattr(cfg_obj, "rope_scaling", None)
        if isinstance(rs, dict):
            kind = rs.get("type") or rs.get("rope_type")
            if kind in (None, "default"):
                cfg_obj.rope_scaling = None
                print(f"compat: cleared unscaled rope_scaling (was {kind!r})")
            elif "type" not in rs:
                rs["type"] = kind
                print(f"compat: mirrored rope_type={kind!r} -> type")

    # Controlled test of the frequency-schedule hypothesis: override the RoPE
    # base on a fixed model, so the precision requirement can be attributed to
    # the schedule rather than to differences between model families.
    base_orig = _find_rope_base(cfg_obj)
    if args.rope_base:
        hit = _set_rope_base(cfg_obj, float(args.rope_base))
        if not hit:
            raise SystemExit("--rope_base: could not find where this config "
                             "stores the RoPE base; override would be silent")
        now = _find_rope_base(cfg_obj)
        if abs(now - args.rope_base) > 1e-6:
            raise SystemExit(f"--rope_base: wrote {hit} but readback gives "
                             f"{now}, not {args.rope_base}")
        print(f"rope_theta override: {base_orig} -> {now}  via {', '.join(hit)}")

    print(f"loading {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg_obj, dtype=getattr(torch, args.dtype),
        device_map="auto", trust_remote_code=trc, low_cpu_mem_usage=True)
    model.eval()

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids
    if args.max_tokens:
        ids = ids[:, :args.max_tokens]
    print(f"WikiText-2 test: {ids.size(1)} tokens after tokenisation")

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dataset": "wikitext-2-raw-v1 / test split, whole split joined by \\n\\n",
        "total_tokens": int(ids.size(1)),
        "batch_size": 1,
        "kv_cache": False,
        "quantized_scope": "RoPE rotation only; weights/activations/Q/K remain "
                           + args.dtype,
        "loss_protocol": "sliding window; logits[:, :-1] vs input[:, 1:], "
                         "shift done explicitly (not via labels=)",
        "harness": "custom (this script); NOT lm-eval-harness",
        "encoder": "rope_angle_encoder.v equivalent, verified bit-exact",
        "rope_theta": _find_rope_base(cfg_obj),
        "rope_theta_overridden": bool(args.rope_base),
    }

    selftest(build_configs(args.configs))
    results, mod, orig = [], None, None

    ctxs = ([int(x) for x in args.long_ctx.split(",")]
            if args.configs == "longctx" else [args.seq_len])

    baseline = {}
    for cfg in build_configs(args.configs):
        for ctx in ctxs:
            stride = args.stride or ctx
            _CACHE.clear()
            mod, orig = patch_model(model, None if cfg.variant == "exact" else cfg)
            t0 = time.time()
            try:
                ppl, ntok = perplexity(model, ids, ctx, stride, dev)
                err = None
            except Exception as e:
                ppl, ntok, err = float("nan"), 0, repr(e)
            mod.apply_rotary_pos_emb = orig
            if cfg.variant == "exact":
                baseline[ctx] = ppl
            delta = (100 * (ppl / baseline[ctx] - 1)) if ctx in baseline else float("nan")
            row = {"config": cfg.name, "variant": cfg.variant, "W": cfg.W,
                   "F": cfg.W - 2, "N": cfg.N, "comp": cfg.comp,
                   "fold_gain": cfg.fold_gain, "seq_len": ctx,
                   "stride": stride, "eval_tokens": ntok,
                   "perplexity": ppl, "delta_pct_vs_fp16": delta,
                   "seconds": round(time.time() - t0, 1), "error": err}
            results.append(row)
            print(f"  {cfg.name:<22} ctx={ctx:<6} ppl={ppl:10.4f} "
                  f"delta={delta:+7.3f}%  tok={ntok:<8} {row['seconds']}s"
                  + (f"  ERROR {err}" if err else ""), flush=True)
            json.dump({"meta": meta, "results": results},
                      open(args.out, "w"), indent=2)

    for ctx, b in baseline.items():
        if not (3.0 < b < 100.0):
            print(f"\n!! WARNING: fp16 baseline at ctx={ctx} is {b:.2f}, which is "
                  f"outside the plausible 3-100 range for WikiText-2.\n"
                  f"   Something is wrong with the evaluation, not the hardware "
                  f"model. Do not use these numbers.")

    print(f"\nwrote {args.out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
