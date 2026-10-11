"""T1.7 — synthesizable form of the Binary UD angle encoder.

Reduces the prefix-P2 encoder to logic that maps to hardware directly:

    S    = (theta + A) >> 1                  1 constant adder + wire shift
    head = threshold_decode(S)               3 constant comparators
    R    = S - HEADSUM[head]                 4:1 constant mux + 1 subtractor
    tail = fixed-point bits of R             WIRES ONLY

The tail is free because a_j = atan(2^-(j+1)) is within 0.13% of 2^-(j+1)
for j >= 3, so the geometric bit-expansion of R already selects the right
digits. Only the head stages are non-geometric and need explicit decode.

Everything here is combinational: no Z-path, which is the point of UD-CORDIC.
"""
import math

def weights(N):
    return [math.atan(2.0 ** -(i + 1)) for i in range(N)]

def achieved(bits, N):
    return sum((1 if b else -1) * w for b, w in zip(bits, weights(N)))

# --------------------------------------------------- synthesizable encoder
def enc_hw(theta, N, P=2, FB=16):
    """FB = fractional bits of the internal angle datapath."""
    a = weights(N)
    A = sum(a)
    q = 1 << FB
    Sq = int(round((theta + A) / 2.0 * q))          # fixed-point S

    headsum = []
    for c in range(1 << P):
        hb = [(c >> (P - 1 - k)) & 1 for k in range(P)]
        headsum.append((hb, int(round(sum(h * w for h, w in zip(hb, a[:P])) * q))))
    headsum.sort(key=lambda t: t[1])

    Tmax = int(round(sum(a[P:]) * q))

    # threshold decode: pick the largest head whose residual is still >= 0,
    # falling back so the residual stays within [0, Tmax]
    chosen = headsum[0]
    for hb, hs in headsum:
        if Sq >= hs:
            chosen = (hb, hs)
    hb, hs = chosen
    R = Sq - hs
    if R > Tmax:
        R = Tmax
    if R < 0:
        R = 0

    # tail: pure bit extraction, weight 2^-(j+1) -> bit (FB-j-1) of R
    tail = []
    for j in range(P, N):
        sh = FB - (j + 1)
        tail.append((R >> sh) & 1 if sh >= 0 else 0)
    return hb + tail

def enc_greedy(theta, N):
    a = weights(N); bits, cur = [], 0.0
    for i in range(N):
        if abs(cur + a[i] - theta) <= abs(cur - a[i] - theta):
            bits.append(1); cur += a[i]
        else:
            bits.append(0); cur -= a[i]
    return bits

# ------------------------------------------------------------- range reduce
def range_reduce(theta):
    """Fold any angle into +/-45 deg. Quadrant rotation is free in hardware
    (swap x/y and negate); returns (quadrant, residual)."""
    t = theta % (2 * math.pi)
    q = int(round(t / (math.pi / 2))) % 4
    r = t - q * (math.pi / 2)
    if r > math.pi:
        r -= 2 * math.pi
    return q, r

def rotate_full(theta, N, FB=16):
    """Full path: range reduce -> encode -> rotate -> undo quadrant."""
    q, r = range_reduce(theta)
    bits = enc_hw(r, N, FB=FB)
    return achieved(bits, N) + q * (math.pi / 2)

def angdiff(a, b):
    d = (a - b) % (2 * math.pi)
    return d - 2 * math.pi if d > math.pi else d

SPAN = math.pi / 4
def ev(fn, N, n=4000, **kw):
    e = [abs(fn(-SPAN + 2 * SPAN * j / (n - 1), N, **kw) -
             (-SPAN + 2 * SPAN * j / (n - 1))) for j in range(n)]
    return (math.degrees(math.sqrt(sum(x * x for x in e) / len(e))),
            math.degrees(max(e)))

print("=" * 78)
print("Synthesizable encoder vs. sequential oracle  (theta in +/-45 deg)")
print("=" * 78)
print(f"{'N':>3} | {'enc_hw FB=12':>13} {'enc_hw FB=16':>13} "
      f"{'enc_hw FB=20':>13} | {'greedy oracle':>14}")
print("-" * 78)
for N in (6, 8, 10, 12, 14):
    r12 = ev(lambda t, n: achieved(enc_hw(t, n, FB=12), n), N)[0]
    r16 = ev(lambda t, n: achieved(enc_hw(t, n, FB=16), n), N)[0]
    r20 = ev(lambda t, n: achieved(enc_hw(t, n, FB=20), n), N)[0]
    rg = ev(lambda t, n: achieved(enc_greedy(t, n), n), N)[0]
    print(f"{N:>3} | {r12:>13.4f} {r16:>13.4f} {r20:>13.4f} | {rg:>14.4f}")

print("\nFull path incl. range reduction, theta over [0, 2*pi):")
print(f"{'N':>3} {'RMS deg':>10} {'max deg':>10}")
for N in (8, 10, 12):
    errs = []
    for j in range(4000):
        th = 2 * math.pi * j / 3999
        errs.append(abs(angdiff(rotate_full(th, N), th)))
    print(f"{N:>3} {math.degrees(math.sqrt(sum(e*e for e in errs)/len(errs))):>10.4f} "
          f"{math.degrees(max(errs)):>10.4f}")

print("\nConstants for the RTL (N=10, FB=16, Q0.16 angle datapath):")
N, FB = 10, 16
a = weights(N); A = sum(a); q = 1 << FB
print(f"  A          = {A:.10f}  -> {int(round(A*q))}  (0x{int(round(A*q)):04X})")
for c in range(4):
    hb = [(c >> 1) & 1, c & 1]
    hs = sum(h * w for h, w in zip(hb, a[:2]))
    print(f"  HEADSUM[{c}] = {hs:.10f}  -> {int(round(hs*q)):5d}  "
          f"(0x{int(round(hs*q)):04X})   b0={hb[0]} b1={hb[1]}")
print(f"  TMAX       = {sum(a[2:]):.10f}  -> {int(round(sum(a[2:])*q))}")
print(f"  K          = {math.prod(math.hypot(1,2.0**-(i+1)) for i in range(N)):.10f}")
