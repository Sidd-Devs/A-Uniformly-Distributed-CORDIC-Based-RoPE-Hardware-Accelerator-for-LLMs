"""Bit-exact replica of rope_angle_encoder.v + golden vector generation."""
import math, random

N, FB = 10, 12
A_CONST, HS_01, HS_10, HS_11, TMAX = 3920, 1003, 1899, 2903, 1017

def weights(n):
    return [math.atan(2.0 ** -(i + 1)) for i in range(n)]
W = weights(N)

def rtl_encode(theta_q):
    """Mirrors the Verilog exactly: integer ops, arithmetic shift = floor."""
    s = (theta_q + A_CONST) >> 1
    b0 = 1 if s >= HS_10 else 0
    if s >= HS_11:
        b1 = 1
    elif s >= HS_10:
        b1 = 0
    else:
        b1 = 1 if s >= HS_01 else 0
    headsum = (0, HS_01, HS_10, HS_11)[(b0 << 1) | b1]
    r = s - headsum
    r = 0 if r < 0 else (TMAX if r > TMAX else r)
    bits = [b0, b1] + [(r >> (FB - j - 1)) & 1 for j in range(2, N)]
    return bits

def achieved(bits):
    return sum((1 if b else -1) * w for b, w in zip(bits, W))

# ---------------------------------------------------------------- accuracy
SPAN = math.pi / 4
errs = []
for k in range(4000):
    th = -SPAN + 2 * SPAN * k / 3999
    tq = int(round(th * (1 << FB)))
    errs.append(abs(achieved(rtl_encode(tq)) - th))
rms = math.degrees(math.sqrt(sum(e * e for e in errs) / len(errs)))
print(f"Bit-exact RTL model, N={N} FB={FB}:")
print(f"  RMS {rms:.4f} deg   max {math.degrees(max(errs)):.4f} deg")
print(f"  paper's reported Binary UD N=10 : 2.2000 deg")
print(f"  improvement                     : {2.20/rms:.1f}x")

# ---------------------------------------------------------------- vectors
random.seed(7)
lim = int(SPAN * (1 << FB))
vecs = [(-lim), (-lim + 1), -1, 0, 1, (lim - 1), lim]
vecs += [random.randint(-lim, lim) for _ in range(500)]
vecs = sorted(set(vecs))

with open("golden_vectors.txt", "w") as f:
    for tq in vecs:
        bits = rtl_encode(tq)
        word = sum(b << i for i, b in enumerate(bits))   # bit i -> stage i
        f.write(f"{tq & 0x3FFF:04x} {word:04x}\n")
print(f"\nWrote {len(vecs)} golden vectors to golden_vectors.txt")
print("  format: <theta_res as 14-bit hex> <ud_bits as hex>")
