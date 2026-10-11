"""Prove the torch model == the verified RTL, before anything runs on Kaggle."""
import math, torch, sys
sys.path.insert(0, ".")
from rope_hw import encode_binary, rotate_binary, binary_gain, HWRoPE, range_reduce

N, FB, W = 10, 12, 9

# ---- 1. encoder vs iverilog golden vectors -------------------------------
gv = "../VERIFY/golden_vectors.txt"
tqs, exps = [], []
for line in open(gv):
    a, b = line.split()
    v = int(a, 16); v = v - 16384 if v >= 8192 else v
    tqs.append(v); exps.append(int(b, 16))
theta = torch.tensor(tqs, dtype=torch.float64) / (1 << FB)
bits = encode_binary(theta, N, FB)
got = sum(bits[:, i].to(torch.int64) << i for i in range(N))
exp = torch.tensor(exps, dtype=torch.int64)
n_bad = int((got != exp).sum())
print(f"[1] encoder vs RTL golden vectors : {len(tqs)-n_bad}/{len(tqs)} match"
      f"  -> {'PASS' if n_bad==0 else 'FAIL'}")

# ---- 2. end-to-end rotation vs the iverilog e2e figures ------------------
print("\n[2] end-to-end rotation error (compare to iverilog run_all.sh):")
span = math.pi/4
th = torch.linspace(-span, span, 401, dtype=torch.float64)
for Wt, ref in ((16, 0.0899), (12, 0.1071), (9, 0.4536)):
    q, res = range_reduce(th)
    b = encode_binary(res, N, FB)
    sc = 1 << (Wt-2)
    xq = torch.full_like(th, sc).to(torch.int64)
    yq = torch.zeros_like(th).to(torch.int64)
    xo, yo = rotate_binary(xq, yq, b, N, Wt)
    got_ang = torch.atan2(yo.double(), xo.double()) + q.double()*(math.pi/2)
    err = torch.remainder(got_ang - th + math.pi, 2*math.pi) - math.pi
    rms = math.degrees(float(torch.sqrt((err**2).mean())))
    K = torch.sqrt(xo.double()**2 + yo.double()**2)/sc
    print(f"    W={Wt:>2}  torch RMS {rms:.4f} deg   iverilog {ref:.4f} deg"
          f"   {'MATCH' if abs(rms-ref)<0.01 else 'DIFFER'}"
          f"   | gain {float(K.mean()):.6f} spread {100*float(K.max()/K.min()-1):.3f}%")

# ---- 3. full RoPE path over unbounded angles -----------------------------
print("\n[3] HWRoPE over unbounded RoPE angles (m up to 4096):")
torch.manual_seed(0)
m = torch.arange(0, 4096, 7, dtype=torch.float64)
inv = 10000.0 ** (-torch.arange(0, 64, dtype=torch.float64)*2/128)
ang = (m[:, None] * inv[None, :]).reshape(-1)
xe = torch.randn(ang.shape, dtype=torch.float64)*0.3
xo_ = torch.randn(ang.shape, dtype=torch.float64)*0.3
ce, co = torch.cos(ang), torch.sin(ang)
ref_e = xe*ce - xo_*co; ref_o = xe*co + xo_*ce
for Wt in (16, 12, 9):
    hw = HWRoPE(variant="binary", W=Wt, N=N, FB=FB, fold_gain=True)
    ye, yo2 = hw(xe, xo_, ang)
    rel = torch.sqrt(((ye-ref_e)**2 + (yo2-ref_o)**2).mean()
                     / ((ref_e**2+ref_o**2).mean()))
    print(f"    W={Wt:>2}  relative vector error {float(rel):.5f}")

print("\n[4] CSD gain compensation sweep (relative vector error):")
for comp in ("none", "H1", "H2"):
    hw = HWRoPE(variant="csd", W=16, N=5, comp=comp)
    ye, yo2 = hw(xe, xo_, ang)
    rel = torch.sqrt(((ye-ref_e)**2 + (yo2-ref_o)**2).mean()
                     / ((ref_e**2+ref_o**2).mean()))
    print(f"    comp={comp:<5} relative vector error {float(rel):.5f}")
