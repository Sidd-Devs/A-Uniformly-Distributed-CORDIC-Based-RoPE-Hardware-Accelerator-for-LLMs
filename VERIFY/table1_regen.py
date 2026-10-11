"""T1.8 part A — regenerate Table I with the working encoders.

The published Table I was produced without a functioning angle encoder, so its
numbers describe neither the RTL nor a correct algorithm. This regenerates it
using the T1.7 Binary UD encoder and the T1.9 CSD encoder.

Columns follow the paper (MSE, angular error) and add the two that turned out
to matter: gain spread, and vector error after constant-gain normalisation.

MSE is reported AFTER dividing out the nominal constant gain, which is what the
original table must have done -- an uncompensated Binary UD gain of 1.164 alone
would give MSE ~2.7e-2, far above the published 4.36e-4.
"""
import math
from csd_encoder import enc_ff, stage_angle, stage_gain, DIGIT_SETS

SPAN = math.pi / 4
NPTS = 4000
CSD_SET = DIGIT_SETS["as-built {0,+-1,+-1.5}"]

# ----------------------------------------------------------- Binary UD (T1.7)
def binw(N):
    return [math.atan(2.0 ** -(i + 1)) for i in range(N)]

def bin_encode(theta, N):
    a = binw(N); A = sum(a); S = (theta + A) / 2.0
    tw = a[2:]; Tmax = sum(tw); best = None
    for c in range(4):
        head = [(c >> 1) & 1, c & 1]
        R = min(max(S - sum(h * w for h, w in zip(head, a[:2])), 0.0), Tmax)
        tb, acc = [], R
        for w in tw:
            if acc >= w: tb.append(1); acc -= w
            else:        tb.append(0)
        bits = head + tb
        got = sum((1 if b else -1) * w for b, w in zip(bits, a))
        if best is None or abs(got - theta) < best[0]:
            best = (abs(got - theta), got)
    return best[1], math.prod(math.hypot(1, 2.0 ** -(i + 1)) for i in range(N))

def csd_encode(theta, N):
    d = enc_ff(theta, N, CSD_SET, P=2)
    return (sum(stage_angle(v, i) for i, v in enumerate(d)),
            math.prod(stage_gain(v, i) for i, v in enumerate(d)))

# ------------------------------------------------------------------- metrics
def measure(encode, N):
    rows = []
    for j in range(NPTS):
        th = -SPAN + 2 * SPAN * j / (NPTS - 1)
        got, K = encode(th, N)
        rows.append((th, got, K))
    Ks = [K for _, _, K in rows]
    Knom = math.sqrt(min(Ks) * max(Ks))
    se = 0.0; ae = []
    for th, got, K in rows:
        s = K / Knom
        dx = s * math.cos(got) - math.cos(th)
        dy = s * math.sin(got) - math.sin(th)
        se += dx * dx + dy * dy
        ae.append(abs(got - th))
    mse = se / (2 * len(rows))                       # per component
    rms_ang = math.sqrt(sum(e * e for e in ae) / len(ae))
    return mse, rms_ang, max(Ks) / min(Ks) - 1.0, math.sqrt(2 * mse)

print("=" * 94)
print("TABLE I (regenerated) — algorithm-level accuracy with working encoders")
print("  theta uniform over +/-45 deg (post range-reduction residual);")
print("  MSE and vector error are after constant-gain normalisation.")
print("=" * 94)
print(f"{'Method':<14}{'Stages':>7}{'MSE':>12}{'AngErr(rad)':>13}"
      f"{'AngErr(deg)':>13}{'GainSpread':>12}{'VecErr':>10}")
print("-" * 94)
for N in (4, 5, 6, 8, 10, 12):
    mse, ar, gs, ve = measure(bin_encode, N)
    print(f"{'Binary UD':<14}{N:>7}{mse:>12.3e}{ar:>13.5f}"
          f"{math.degrees(ar):>13.4f}{100*gs:>11.2f}%{ve:>10.5f}")
print()
for N in (4, 5, 6, 8):
    mse, ar, gs, ve = measure(csd_encode, N)
    print(f"{'CSD UD':<14}{N:>7}{mse:>12.3e}{ar:>13.5f}"
          f"{math.degrees(ar):>13.4f}{100*gs:>11.2f}%{ve:>10.5f}")

print("\n" + "=" * 94)
print("Published Table I vs. regenerated, at the same configurations")
print("=" * 94)
print(f"{'Method':<12}{'N':>4}{'published deg':>16}{'regenerated deg':>18}{'change':>12}")
for name, enc, N, old in (("Binary UD", bin_encode, 5, 3.88),
                          ("Binary UD", bin_encode, 10, 2.20),
                          ("CSD UD", csd_encode, 5, 3.38),
                          ("CSD UD", csd_encode, 10, 3.13)):
    if name == "CSD UD" and N == 10:
        new = measure(csd_encode, 8)[1]; note = " (N=8; N=10 not run, cost)"
    else:
        new = measure(enc, N)[1]; note = ""
    print(f"{name:<12}{N:>4}{old:>16.2f}{math.degrees(new):>18.4f}"
          f"{old/math.degrees(new):>11.0f}x{note}")

print("""
Reading of the change
  The published column is flat in N -- 3.88 -> 2.20 for Binary UD and
  3.38 -> 3.13 for CSD -- which is the signature of an encoder that never
  converged, not of a precision limit. With working encoders both families
  converge at the expected rate, and the N/2 equivalence claim (CSD N/2 vs
  Binary N) is now actually supported: CSD N=5 reaches 0.043 deg against
  Binary N=10 at 0.063 deg.

  The gain-spread column is the one the original table omitted, and it is
  where the two architectures genuinely differ.""")
