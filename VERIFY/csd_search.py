"""T1.1: exhaustive search over encoding conventions.

Question: is ANY consistent interpretation of angle_bits under which the
as-built CSD datapath converges? If yes, the bug is a wiring/encoding issue.
If no, the architecture itself is broken.
"""
import math
import itertools

def booth(bits, STAGES, carry):
    """carry='prev' = RTL (b[2i-1]); 'next' = descending-weight-correct (b[2i+2])."""
    n = len(bits)
    d = []
    for i in range(STAGES):
        b0, b1 = bits[2 * i], bits[2 * i + 1]
        if carry == "prev":
            c = 0 if i == 0 else bits[2 * i - 1]
        else:
            c = bits[2 * i + 2] if 2 * i + 2 < n else 0
        d.append(-2 * b1 + b0 + c)
    return d


def run(bits, STAGES, carry, d2, order):
    """order='asc'  -> stage i shift 2i   (RTL as built)
       order='desc' -> stage i shift 2(N-1-i)"""
    digits = booth(bits, STAGES, carry)
    x, y = 1.0, 0.0
    for i in range(STAGES):
        sh = 2 * i if order == "asc" else 2 * (STAGES - 1 - i)
        s1 = 2.0 ** (-sh)
        s2 = 2.0 ** (-(sh + 1))
        big = (s1 + s2) if d2 == "1.5" else (2 * s1)
        t = {2: big, 1: s1, 0: 0.0, -1: -s1, -2: -big}[digits[i]]
        x, y = x - t * y, y + t * x
    return math.atan2(y, x), math.hypot(x, y)


def encode(theta, STAGES, bitorder, span):
    nb = 2 * STAGES
    I = round(theta / span * ((1 << nb) - 1))
    I = max(0, min(I, (1 << nb) - 1))
    if bitorder == "msb":            # bits[k] weight 2^-k
        return [(I >> (nb - 1 - k)) & 1 for k in range(nb)]
    return [(I >> k) & 1 for k in range(nb)]   # bits[k] weight 2^k


def evaluate(STAGES, carry, d2, order, bitorder, span, n=1500):
    errs = []
    for j in range(n):
        th = span * j / (n - 1)
        got, _ = run(encode(th, STAGES, bitorder, span), STAGES, carry, d2, order)
        errs.append(abs(got - th))
    return math.degrees(math.sqrt(sum(e * e for e in errs) / len(errs)))


SPAN = 1.0
print("Search over conventions. RMS angular error (deg), theta in [0, 1] rad.")
print("Converging = error must DROP as N grows.\n")
hdr = f"{'carry':>6} {'dig2':>5} {'order':>5} {'bits':>5} | " + \
      " ".join(f"{'N='+str(n):>8}" for n in (4, 5, 6, 8, 10, 12))
print(hdr)
print("-" * len(hdr))

results = []
for carry, d2, order, bitorder in itertools.product(
        ("prev", "next"), ("1.5", "2.0"), ("asc", "desc"), ("msb", "lsb")):
    row = [evaluate(N, carry, d2, order, bitorder, SPAN) for N in (4, 5, 6, 8, 10, 12)]
    conv = row[-1] < row[0] * 0.5          # meaningfully improving with N
    tag = "  <== CONVERGES" if conv and row[-1] < 5 else ""
    flag = " *RTL*" if (carry, d2, order) == ("prev", "1.5", "asc") else ""
    print(f"{carry:>6} {d2:>5} {order:>5} {bitorder:>5} | " +
          " ".join(f"{v:>8.3f}" for v in row) + tag + flag)
    results.append(((carry, d2, order, bitorder), row, conv))

print("\n" + "=" * 78)
best = min(results, key=lambda r: r[1][-1])
print(f"Best convention: carry={best[0][0]} digit2={best[0][1]} "
      f"shift-order={best[0][2]} bitorder={best[0][3]}")
print(f"  errors by N: " + " ".join(f"{v:.3f}" for v in best[1]))

rtl = [r for r in results if r[0][:3] == ("prev", "1.5", "asc")]
print("\nAs-built RTL rows (both bit orders):")
for k, row, _ in rtl:
    print(f"  bitorder={k[3]}: " + " ".join(f"{v:8.3f}" for v in row))
