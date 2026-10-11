"""T2.11 — how coarse can the block floating-point exponent be?

One shared exponent per (head, token) covers d_h/2 = 64 pairs. A finer block
tracks the local magnitude better and loses less mantissa, but needs a
max-reduction tree and leading-zero count per block instead of per head.
This measures the accuracy half; synthesis runs 7/16/17 give the cost half.

Uses REAL query/key activations captured from Qwen2-0.5B, because the entire
question is how much dynamic range sits inside one block -- synthetic data
with a chosen distribution would assume the answer.
"""
import math, sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "KAGGLE"))
from rope_hw import encode_binary, rotate_binary, binary_gain, range_reduce

CAP = {}

def capture(model, tok, text, layers=(0, 6, 11)):
    import transformers.models.qwen2.modeling_qwen2 as M
    orig = M.apply_rotary_pos_emb
    n = {"i": 0}
    def spy(q, k, cos, sin, *a, **kw):
        if n["i"] in layers:
            CAP[n["i"]] = (q.detach().float().cpu(), k.detach().float().cpu())
        n["i"] += 1
        return orig(q, k, cos, sin, *a, **kw)
    M.apply_rotary_pos_emb = spy
    ids = tok(text, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        model(ids)
    M.apply_rotary_pos_emb = orig

def block_scale(x1, x2, D):
    """Shared exponent over blocks of D pairs along the last axis."""
    *lead, H = x1.shape
    if D >= H:
        amax = torch.maximum(x1.abs(), x2.abs()).amax(-1, keepdim=True)
        return torch.exp2(-(torch.floor(torch.log2(amax.clamp(min=1e-12))) + 1))
    nb = H // D
    a = torch.maximum(x1.abs(), x2.abs())[..., :nb * D].reshape(*lead, nb, D)
    amax = a.amax(-1, keepdim=True).clamp(min=1e-12)
    s = torch.exp2(-(torch.floor(torch.log2(amax)) + 1))
    return s.expand(*lead, nb, D).reshape(*lead, nb * D)

def err_for(x1, x2, theta, D, W, N=10, FB=12):
    quad, res = range_reduce(theta.double())
    bits = encode_binary(res, N, FB)
    H = min(x1.shape[-1], (x1.shape[-1] // D) * D if D < x1.shape[-1] else x1.shape[-1])
    x1, x2 = x1[..., :H], x2[..., :H]
    bits = bits[:H] if bits.shape[0] >= H else bits
    bs = block_scale(x1, x2, D)
    sc, lim = 1 << (W - 2), 1 << (W - 1)
    xq = torch.clamp(torch.round(x1 * bs * sc), -lim, lim - 1).to(torch.int64)
    yq = torch.clamp(torch.round(x2 * bs * sc), -lim, lim - 1).to(torch.int64)
    b = bits.view(*([1] * (x1.dim() - 1)), H, N)
    xo, yo = rotate_binary(xq, yq, b, N, W)
    K = binary_gain(N)
    xr = xo.double() / sc / K / bs
    yr = yo.double() / sc / K / bs
    c, s = torch.cos(theta[:H].double()), torch.sin(theta[:H].double())
    r1 = x1.double() * c - x2.double() * s
    r2 = x1.double() * s + x2.double() * c
    num = ((xr - r1) ** 2 + (yr - r2) ** 2).mean()
    den = (r1 ** 2 + r2 ** 2).mean()
    return float(torch.sqrt(num / den))

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B",
                                             dtype=torch.float32).eval()
    capture(m, tok, "The quick brown fox jumps over the lazy dog. " * 60)
    q, k = CAP[6]
    H = q.shape[-1] // 2
    x1, x2 = q[0, :, :, :H], q[0, :, :, H:]
    print(f"captured q from layer 6: {tuple(q.shape)}  "
          f"|q|max={float(q.abs().max()):.1f}  median={float(q.abs().median()):.2f}")
    S = x1.shape[1]
    pos = torch.arange(S, dtype=torch.float64)
    inv = 1e6 ** (-torch.arange(0, H, dtype=torch.float64) * 2 / (2 * H))
    theta = (pos[:, None] * inv[None, :])[0]     # per-dimension angles

    print("\nRelative vector error vs block size (Binary UD N=10, real Qwen2 q)")
    print(f"{'W':>3} {'F':>3} | " + " ".join(f"{'D='+str(d):>9}" for d in (1,2,4,8,16,32,64)))
    print("-" * 78)
    for W in (16, 14, 13, 12):
        row = [err_for(x1, x2, theta, D, W) for D in (1,2,4,8,16,32,64)]
        print(f"{W:>3} {W-2:>3} | " + " ".join(f"{v:>9.5f}" for v in row))
    print("""
D = 64 is one exponent per (head, token) -- what the evaluation used.
D = 1 is a per-pair exponent, i.e. a true per-element float.""")
