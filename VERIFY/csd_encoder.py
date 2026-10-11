"""T1.9 — CSD UD-CORDIC angle encoder.

Stage i applies x -= t*y ; y += t*x with t = v * 2^-2i, v drawn from the
stage's digit set. Rotation angle = atan(v * 2^-2i), gain = sqrt(1 + v^2*4^-2i).

Two questions:
  Q1  Is a feed-forward encoder possible for the AS-BUILT digit set
      {0, +-1, +-1.5}?
  Q2  How does the gain behave? (S-11 says it is angle-dependent, which
      would rule out contribution C1 for CSD.)

Convergence condition for a redundant CORDIC: after picking the best digit at
stage i, the worst-case leftover is half the largest gap in the digit set, and
the remaining stages must be able to cover it.

    digit set          max gap   worst residual   remaining capacity
    {0,+-1,+-1.5}        1.0         0.50               0.500     MARGINAL
    {0,+-1,+-2}          1.0         0.50               0.667     OK
    {+-0.5,+-1.5}        1.0         0.50               0.500     MARGINAL
    {0,+-1,+-2,+-3}      1.0         0.50               1.000     OK

(all in units of 2^-2i; remaining capacity = sum_{j>i} max_v * 4^-(j-i))
"""
import math

DIGIT_SETS = {
    "as-built {0,+-1,+-1.5}": [0, 1, -1, 1.5, -1.5],
    "booth   {0,+-1,+-2}":    [0, 1, -1, 2, -2],
    "merge   {+-0.5,+-1.5}":  [0.5, -0.5, 1.5, -1.5],
    "rich    {0,+-1,+-2,+-3}": [0, 1, -1, 2, -2, 3, -3],
}

def stage_angle(v, i):
    return math.atan(v * 2.0 ** (-2 * i))

def stage_gain(v, i):
    return math.hypot(1.0, v * 2.0 ** (-2 * i))

def capacity(dset, i, N):
    return sum(max(dset) * 4.0 ** -(j - i) for j in range(i + 1, N))

# ------------------------------------------------------------------ encoders
def enc_greedy(theta, N, dset):
    """Sequential oracle (a Z-path). Upper bound on what any encoder can do."""
    cur, digs = 0.0, []
    for i in range(N):
        best = None
        for v in dset:
            e = abs(cur + stage_angle(v, i) - theta)
            if best is None or e < best[0]:
                best = (e, v)
        digs.append(best[1])
        cur += stage_angle(best[1], i)
    return digs

def enc_ff(theta, N, dset, P=2):
    """Feed-forward: brute-force the P non-geometric head stages in parallel,
    then quantise the geometric tail digit-by-digit with NO feedback -- each
    tail digit is read straight off the residual's base-4 expansion."""
    best = None
    heads = [[]]
    for i in range(min(P, N)):
        heads = [h + [v] for h in heads for v in dset]
    for head in heads:
        cur = sum(stage_angle(v, i) for i, v in enumerate(head))
        R = theta - cur
        digs = list(head)
        # tail: angle ~= v*4^-i, so digits come from a fixed-point read of R
        for i in range(len(head), N):
            w = 4.0 ** -i
            q = R / w
            v = min(dset, key=lambda d: abs(d - q))
            digs.append(v)
            R -= v * w                      # constant-weight, not a rotation
        got = sum(stage_angle(v, i) for i, v in enumerate(digs))
        e = abs(got - theta)
        if best is None or e < best[0]:
            best = (e, digs)
    return best[1]

def achieved(digs):
    return sum(stage_angle(v, i) for i, v in enumerate(digs))

def gain_of(digs):
    return math.prod(stage_gain(v, i) for i, v in enumerate(digs))

# ---------------------------------------------------------------- evaluation
SPAN = math.pi / 4

def ev(enc, N, dset, n=1500, **kw):
    errs, gains = [], []
    for j in range(n):
        th = -SPAN + 2 * SPAN * j / (n - 1)
        d = enc(th, N, dset, **kw)
        errs.append(abs(achieved(d) - th))
        gains.append(gain_of(d))
    return (math.degrees(math.sqrt(sum(e * e for e in errs) / len(errs))),
            min(gains), max(gains))

print("=" * 86)
print("T1.9  CSD encoder — angular accuracy (RMS deg) and gain spread")
print("=" * 86)
print(f"{'digit set':>24} {'N':>3} | {'greedy*':>9} {'feed-fwd P=1':>13} "
      f"{'feed-fwd P=2':>13} | {'gain min':>9} {'gain max':>9}")
print("-" * 86)
for name, dset in DIGIT_SETS.items():
    for N in (5, 6, 8):
        g = ev(enc_greedy, N, dset)[0]
        f1, gmn, gmx = ev(enc_ff, N, dset, P=1)
        f2 = ev(enc_ff, N, dset, P=2)[0]
        print(f"{name:>24} {N:>3} | {g:>9.4f} {f1:>13.4f} {f2:>13.4f} "
              f"| {gmn:>9.4f} {gmx:>9.4f}")
    print()
print("* greedy is sequential (a Z-path) -- shown as the achievable bound only.")

print("\n" + "=" * 86)
print("Convergence check: can the remaining stages absorb the worst residual?")
print("=" * 86)
print(f"{'digit set':>24} {'max digit':>10} {'worst resid':>12} "
      f"{'capacity':>10} {'verdict':>12}")
for name, dset in DIGIT_SETS.items():
    s = sorted(set(abs(d) for d in dset))
    gaps = [s[k + 1] - s[k] for k in range(len(s) - 1)] or [max(s)]
    worst = max(gaps) / 2
    cap = capacity(dset, 0, 12)
    print(f"{name:>24} {max(dset):>10.1f} {worst:>12.3f} {cap:>10.3f} "
          f"{'OK' if cap >= worst else 'MARGINAL':>12}")

print("\n" + "=" * 86)
print("Gain decomposition — which stage causes the spread?")
print("=" * 86)
for name, dset in list(DIGIT_SETS.items())[:2]:
    print(f"\n  {name}")
    for i in range(4):
        gs = [stage_gain(v, i) for v in dset]
        print(f"    stage {i}: gain in [{min(gs):.6f}, {max(gs):.6f}]  "
              f"spread {100*(max(gs)/min(gs)-1):6.2f}%")

print("\n" + "=" * 86)
print("Hardware cost of the digit-2 term (per stage)")
print("=" * 86)
print("  v = +-1.5 : (x >> 2i) + (x >> 2i+1)   -> needs a REAL ADDER")
print("  v = +-2   : (x >> 2i-1)               -> WIRE ONLY, no adder")
print("  => the textbook +-2 digit is both more accurate AND cheaper")
