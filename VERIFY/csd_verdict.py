"""T1.9 decision experiment: what RoPE actually cares about.

RoPE needs the rotated vector, not just the angle. Error therefore has two
parts: angle error, and gain error. A CONSTANT gain is free to remove (fold
K^2 into the 1/sqrt(d) attention scale -- contribution C1). A gain that varies
with the angle is NOT, and shows up as a per-position, per-dimension magnitude
distortion of q and k.

Metric: || v_out / K_nom  -  R(theta) v_in || / || v_in ||
        with K_nom the single best constant scale for that architecture.
"""
import math
from csd_encoder import enc_ff, achieved, gain_of, DIGIT_SETS

SPAN = math.pi / 4

def binw(N):
    return [math.atan(2.0 ** -(i + 1)) for i in range(N)]

def bin_enc(theta, N):
    """The T1.7 encoder (prefix P=2 + geometric tail)."""
    a = binw(N); A = sum(a); S = (theta + A) / 2.0
    tailw = a[2:]; Tmax = sum(tailw); best = None
    for c in range(4):
        head = [(c >> 1) & 1, c & 1]
        R = min(max(S - sum(h * w for h, w in zip(head, a[:2])), 0.0), Tmax)
        tb, acc = [], R
        for w in tailw:
            if acc >= w: tb.append(1); acc -= w
            else:        tb.append(0)
        bits = head + tb
        got = sum((1 if b else -1) * w for b, w in zip(bits, a))
        e = abs(got - theta)
        if best is None or e < best[0]: best = (e, bits, got)
    return best[1], best[2]

def bin_gain(N):
    return math.prod(math.hypot(1, 2.0 ** -(i + 1)) for i in range(N))

def vec_err(angles_gains, Knom):
    """RMS relative vector error after dividing out the constant Knom."""
    tot = 0.0
    for th, got, K in angles_gains:
        s = K / Knom
        dx = s * math.cos(got) - math.cos(th)
        dy = s * math.sin(got) - math.sin(th)
        tot += dx * dx + dy * dy
    return math.sqrt(tot / len(angles_gains))

n = 1500
print("=" * 78)
print("T1.9 verdict — RMS relative vector error after best constant-gain")
print("            normalisation (this is what RoPE actually sees)")
print("=" * 78)

# Binary UD
print(f"\n{'Binary UD (T1.7 encoder)':<32} {'ang RMS':>9} {'gain spread':>13} {'vec err':>10}")
for N in (8, 10, 12):
    ag = []
    for j in range(n):
        th = -SPAN + 2 * SPAN * j / (n - 1)
        _, got = bin_enc(th, N)
        ag.append((th, got, bin_gain(N)))
    ang = math.degrees(math.sqrt(sum((g - t) ** 2 for t, g, _ in ag) / len(ag)))
    Ks = [K for _, _, K in ag]
    print(f"{'  N=' + str(N):<32} {ang:>9.4f} "
          f"{100*(max(Ks)/min(Ks)-1):>12.3f}% {vec_err(ag, bin_gain(N)):>10.5f}")

# CSD
for name, dset in list(DIGIT_SETS.items())[:2]:
    print(f"\n{'CSD ' + name:<32} {'ang RMS':>9} {'gain spread':>13} {'vec err':>10}")
    for N in (4, 5, 6):
        ag = []
        for j in range(n):
            th = -SPAN + 2 * SPAN * j / (n - 1)
            d = enc_ff(th, N, dset, P=2)
            ag.append((th, achieved(d), gain_of(d)))
        ang = math.degrees(math.sqrt(sum((g - t) ** 2 for t, g, _ in ag) / len(ag)))
        Ks = [K for _, _, K in ag]
        Knom = math.sqrt(min(Ks) * max(Ks))
        print(f"{'  N=' + str(N):<32} {ang:>9.4f} "
              f"{100*(max(Ks)/min(Ks)-1):>12.1f}% {vec_err(ag, Knom):>10.5f}")

print("\n" + "=" * 78)
print("Interpretation")
print("=" * 78)
print("""  Binary UD  : gain spread ~0%, so the constant K folds into the attention
               scale for free (C1) and the residual vector error IS the
               angle error.
  CSD        : angle error is fine, but gain varies by 50-60% with the angle.
               No constant can remove it, so the vector error is ~100x worse
               than Binary UD's and appears as position-dependent magnitude
               distortion on q and k.

  Root cause: stage gain is sqrt(1 + v^2*4^-i), so it depends on the DIGIT
  MAGNITUDE. Any digit set with more than one magnitude has angle-dependent
  gain. A fixed-magnitude set means one bit per stage -- which is exactly
  Binary UD.

  => Constant gain and radix-4 stage merging are mutually exclusive.
     Removing the CSD gain needs a real multiplier, which costs more than
     the adders the merging saves.""")
