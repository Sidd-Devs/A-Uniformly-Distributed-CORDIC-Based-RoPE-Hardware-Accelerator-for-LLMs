"""Bit-accurate model of the CSD UD-CORDIC datapath (T1.1 / T1.2).

Mirrors RTL/csd_ud_stage.v, csd_ud_cordic_core.v, csd_control.v exactly,
then compares against candidate fixes.
"""
import math

# ---------------------------------------------------------------- Booth decode
def booth_digits(bits, STAGES):
    """csd_control.v: delta_i = -2*b[2i+1] + b[2i] + b[2i-1]."""
    d = []
    for i in range(STAGES):
        bm1 = 0 if i == 0 else bits[2 * i - 1]
        b0 = bits[2 * i]
        b1 = bits[2 * i + 1]
        d.append(-2 * b1 + b0 + bm1)
    return d


# ---------------------------------------------------------- effective rotation
def stage_terms(delta, i, mode):
    """Return the multiplier applied to the cross term at stage i.

    RTL: x_out = x - y_comb ; y_out = y + x_comb
    with y_comb/x_comb built from shifts SHIFT1=2i, SHIFT2=2i+1.
    """
    s1 = 2.0 ** (-2 * i)
    s2 = 2.0 ** (-(2 * i + 1))
    if mode == "rtl":                       # as synthesized
        table = {2: s1 + s2, 1: s1, 0: 0.0, -1: -s1, -2: -(s1 + s2)}
    elif mode == "booth":                   # digit 2 -> true 2*2^-2i
        table = {2: 2 * s1, 1: s1, 0: 0.0, -1: -s1, -2: -2 * s1}
    else:
        raise ValueError(mode)
    return table[delta]


def rotate_float(x, y, bits, STAGES, mode):
    """Infinite-precision datapath: isolates algorithmic error."""
    for i in range(STAGES):
        d = booth_digits(bits, STAGES)[i]
        t = stage_terms(d, i, mode)
        x, y = x - t * y, y + t * x
    return x, y


def sar(v, s):
    """Arithmetic right shift with floor semantics, matching Verilog >>>."""
    return v >> s if s >= 0 else v << (-s)


def wrap(v, W):
    m = 1 << W
    v &= m - 1
    return v - m if v >= (m >> 1) else v


def rotate_fixed(x, y, bits, STAGES, W, mode):
    """Bit-accurate integer datapath in Q1.(W-2)."""
    digits = booth_digits(bits, STAGES)
    for i in range(STAGES):
        d = digits[i]
        s1, s2 = 2 * i, 2 * i + 1
        if mode == "rtl":
            if d == 2:
                xc, yc = sar(x, s1) + sar(x, s2), sar(y, s1) + sar(y, s2)
            elif d == 1:
                xc, yc = sar(x, s1), sar(y, s1)
            elif d == 0:
                xc, yc = 0, 0
            elif d == -1:
                xc, yc = -sar(x, s1), -sar(y, s1)
            else:
                xc, yc = -(sar(x, s1) + sar(x, s2)), -(sar(y, s1) + sar(y, s2))
        else:  # booth: digit 2 == shift by (2i-1)
            if d == 2:
                xc, yc = sar(x, s1 - 1), sar(y, s1 - 1)
            elif d == 1:
                xc, yc = sar(x, s1), sar(y, s1)
            elif d == 0:
                xc, yc = 0, 0
            elif d == -1:
                xc, yc = -sar(x, s1), -sar(y, s1)
            else:
                xc, yc = -sar(x, s1 - 1), -sar(y, s1 - 1)
        x, y = wrap(x - yc, W), wrap(y + xc, W)
    return x, y


# ------------------------------------------------------------------- encoding
def encode(theta, STAGES):
    """angle_bits[k] has datapath weight 2^-k, k = 0 .. 2*STAGES-1."""
    nb = 2 * STAGES
    I = round(theta * (1 << (nb - 1)))
    I = max(0, min(I, (1 << nb) - 1))
    return [(I >> (nb - 1 - k)) & 1 for k in range(nb)]


def decoded_angle(bits, STAGES):
    """What the Booth digits claim the angle is, at datapath weights."""
    return sum(d * 2.0 ** (-2 * i) for i, d in enumerate(booth_digits(bits, STAGES)))


# ------------------------------------------------------------------ experiment
def sweep(STAGES, mode, theta_max, n=2000, fixed=False, W=9):
    errs, gains = [], []
    for j in range(n):
        th = theta_max * j / (n - 1)
        bits = encode(th, STAGES)
        if fixed:
            scale = 1 << (W - 2)
            xi, yi = scale, 0
            xo, yo = rotate_fixed(xi, yi, bits, STAGES, W, mode)
            if xo == 0 and yo == 0:
                continue
            got = math.atan2(yo, xo)
            gains.append(math.hypot(xo, yo) / scale)
        else:
            xo, yo = rotate_float(1.0, 0.0, bits, STAGES, mode)
            got = math.atan2(yo, xo)
            gains.append(math.hypot(xo, yo))
        errs.append(abs(got - th))
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    return math.degrees(rms), math.degrees(max(errs)), sum(gains) / len(gains)


print("=" * 78)
print("T1.1/T1.2  CSD UD-CORDIC datapath verification")
print("=" * 78)

print("\n[A] Per-stage effective rotation term (units of 2^-2i)")
print(f"{'digit':>6} {'RTL as-built':>16} {'true Booth':>14} {'error':>10}")
for d in (2, 1, 0, -1, -2):
    r = stage_terms(d, 0, "rtl")
    b = stage_terms(d, 0, "booth")
    tag = "" if abs(r - b) < 1e-12 else f"{(r-b):+.3f}  <-- BUG"
    print(f"{d:>6} {r:>16.4f} {b:>14.4f} {tag:>10}")

print("\n[B] Algorithmic angular error, infinite precision (theta in [0,1] rad)")
print(f"{'N':>3} {'mode':>7} {'RMS err (deg)':>15} {'max err (deg)':>15} {'mean gain':>11}")
for N in (4, 5, 6, 8, 10, 12):
    for mode in ("rtl", "booth"):
        rms, mx, g = sweep(N, mode, 1.0)
        print(f"{N:>3} {mode:>7} {rms:>15.4f} {mx:>15.4f} {g:>11.4f}")

print("\n[C] Bit-accurate fixed point, W=9 (Q1.7), theta in [0,1] rad")
print(f"{'N':>3} {'mode':>7} {'RMS err (deg)':>15} {'max err (deg)':>15} {'mean gain':>11}")
for N in (4, 5, 6, 8, 10, 12):
    for mode in ("rtl", "booth"):
        rms, mx, g = sweep(N, mode, 1.0, fixed=True, W=9)
        print(f"{N:>3} {mode:>7} {rms:>15.4f} {mx:>15.4f} {g:>11.4f}")

print("\n[D] Does the encoded angle even match the requested angle?")
print("    (Booth-decoded value at datapath weights vs. requested theta)")
print(f"{'theta':>8} {'N=5 decoded':>14} {'err':>10} {'N=10 decoded':>14} {'err':>10}")
for th in (0.1, 0.25, 0.5, 0.75, 1.0):
    d5 = decoded_angle(encode(th, 5), 5)
    d10 = decoded_angle(encode(th, 10), 10)
    print(f"{th:>8.3f} {d5:>14.5f} {d5-th:>10.5f} {d10:>14.5f} {d10-th:>10.5f}")

print("\n[E] Where does the error come from? Achieved vs. commanded, N=5, RTL")
print(f"{'theta':>8} {'decoded':>10} {'achieved':>10} {'gain':>8}")
for th in (0.1, 0.25, 0.5, 0.75, 1.0):
    bits = encode(th, 5)
    x, y = rotate_float(1.0, 0.0, bits, 5, "rtl")
    print(f"{th:>8.3f} {decoded_angle(bits,5):>10.5f} "
          f"{math.atan2(y,x):>10.5f} {math.hypot(x,y):>8.4f}")
