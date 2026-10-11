"""T1.1 controls.

CONTROL 1: does the Binary UD core converge in this same harness?
           (paper claims 3.88 deg @ N=5, 2.20 deg @ N=10)
           If yes -> harness is sound, the CSD failure is real.

CONTROL 2: is the CSD DATAPATH capable of fine rotation if the digits were
           chosen optimally (greedy search instead of csd_control.v)?
           Separates "encoder is wrong" from "architecture is deficient".
"""
import math

# ------------------------------------------------------- Binary UD (control 1)
def binary_ud(bits, N):
    """RTL/binary_ud_cordic_core.v: bit 1 -> +rot, bit 0 -> -rot, shift i+1."""
    x, y = 1.0, 0.0
    for i in range(N):
        s = 2.0 ** (-(i + 1))
        d = 1.0 if bits[i] else -1.0
        x, y = x - d * s * y, y + d * s * x
    return math.atan2(y, x), math.hypot(x, y)


def binary_greedy(theta, N):
    """Best achievable: pick each bit to minimise residual (this is what a
    correct UD angle encoder does)."""
    x, y = 1.0, 0.0
    bits = []
    for i in range(N):
        s = 2.0 ** (-(i + 1))
        best = None
        for d in (1.0, -1.0):
            nx, ny = x - d * s * y, y + d * s * x
            e = abs(math.atan2(ny, nx) - theta)
            if best is None or e < best[0]:
                best = (e, d, nx, ny)
        _, d, x, y = best
        bits.append(1 if d > 0 else 0)
    return math.atan2(y, x), bits


# --------------------------------------------------------- CSD UD (control 2)
def csd_apply(x, y, digits, N, d2mode):
    for i in range(N):
        s1 = 2.0 ** (-2 * i)
        s2 = 2.0 ** (-(2 * i + 1))
        big = (s1 + s2) if d2mode == "1.5" else 2 * s1
        t = {2: big, 1: s1, 0: 0.0, -1: -s1, -2: -big}[digits[i]]
        x, y = x - t * y, y + t * x
    return x, y


def csd_greedy(theta, N, d2mode):
    """Oracle encoder: choose each radix-4 digit to minimise residual angle.
    If this cannot converge, the DATAPATH is the problem, not csd_control.v."""
    x, y = 1.0, 0.0
    digs = []
    for i in range(N):
        s1 = 2.0 ** (-2 * i)
        s2 = 2.0 ** (-(2 * i + 1))
        big = (s1 + s2) if d2mode == "1.5" else 2 * s1
        best = None
        for d in (2, 1, 0, -1, -2):
            t = {2: big, 1: s1, 0: 0.0, -1: -s1, -2: -big}[d]
            nx, ny = x - t * y, y + t * x
            e = abs(math.atan2(ny, nx) - theta)
            if best is None or e < best[0]:
                best = (e, d, nx, ny)
        _, d, x, y = best
        digs.append(d)
    return math.atan2(y, x), digs


def rms_deg(fn, N, span=1.0, n=1200):
    errs = []
    for j in range(n):
        th = span * j / (n - 1)
        got = fn(th, N)
        errs.append(abs(got - th))
    return math.degrees(math.sqrt(sum(e * e for e in errs) / len(errs)))


print("=" * 76)
print("CONTROL 1 — Binary UD core, greedy (correct) angle encoding")
print("  Paper Table I claims: N=5 -> 3.88 deg, N=10 -> 2.20 deg")
print("=" * 76)
print(f"{'N':>4} {'RMS err (deg)':>15}")
for N in (4, 5, 6, 8, 10, 12, 14):
    print(f"{N:>4} {rms_deg(lambda t, n: binary_greedy(t, n)[0], N):>15.4f}")

print("\n  Binary UD with NAIVE binary-expansion bits (no greedy search):")
def bin_naive(theta, N):
    I = round(theta * (1 << N))
    I = max(0, min(I, (1 << N) - 1))
    bits = [(I >> (N - 1 - k)) & 1 for k in range(N)]
    return binary_ud(bits, N)[0]
print(f"{'N':>4} {'RMS err (deg)':>15}")
for N in (4, 5, 6, 8, 10, 12):
    print(f"{N:>4} {rms_deg(bin_naive, N):>15.4f}")

print("\n" + "=" * 76)
print("CONTROL 2 — CSD datapath with ORACLE (greedy) digit selection")
print("  This is the best the as-built datapath can possibly do.")
print("=" * 76)
print(f"{'N':>4} {'digit2=1.5 (RTL)':>20} {'digit2=2.0 (fixed)':>20}")
for N in (4, 5, 6, 8, 10, 12):
    a = rms_deg(lambda t, n: csd_greedy(t, n, "1.5")[0], N)
    b = rms_deg(lambda t, n: csd_greedy(t, n, "2.0")[0], N)
    print(f"{N:>4} {a:>20.4f} {b:>20.4f}")

print("\n" + "=" * 76)
print("CONTROL 3 — what the CSD stage-0 rotation actually is")
print("=" * 76)
for i in range(4):
    s1 = 2.0 ** (-2 * i)
    print(f"  stage {i}: shift {2*i:>2} -> digit1 rotates atan({s1:.5f}) = "
          f"{math.degrees(math.atan(s1)):7.3f} deg   "
          f"gain {math.hypot(1, s1):.4f}")
print("\n  Binary UD for comparison:")
for i in range(4):
    s = 2.0 ** (-(i + 1))
    print(f"  stage {i}: shift {i+1:>2} -> rotates atan({s:.5f}) = "
          f"{math.degrees(math.atan(s)):7.3f} deg   gain {math.hypot(1,s):.4f}")

print("\n" + "=" * 76)
print("CONTROL 4 — reachable angle set size (N=5), oracle digits")
print("=" * 76)
for mode in ("1.5", "2.0"):
    reach = set()
    def rec(i, x, y, N=5):
        if i == N:
            reach.add(round(math.atan2(y, x), 6)); return
        s1 = 2.0 ** (-2 * i); s2 = 2.0 ** (-(2 * i + 1))
        big = (s1 + s2) if mode == "1.5" else 2 * s1
        for d in (2, 1, 0, -1, -2):
            t = {2: big, 1: s1, 0: 0.0, -1: -s1, -2: -big}[d]
            rec(i + 1, x - t * y, y + t * x)
    rec(0, 1.0, 0.0)
    pts = sorted(p for p in reach if 0 <= p <= 1.0)
    gaps = [pts[k + 1] - pts[k] for k in range(len(pts) - 1)]
    print(f"  digit2={mode}: {len(reach)} distinct angles, "
          f"{len(pts)} in [0,1] rad, max gap "
          f"{math.degrees(max(gaps)) if gaps else float('nan'):.4f} deg")
