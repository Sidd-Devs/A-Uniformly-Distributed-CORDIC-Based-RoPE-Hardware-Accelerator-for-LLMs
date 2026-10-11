"""T1.9 rescue attempt: compensate only the head stages.

Stage gain is sqrt(1+v^2*4^-i), so stages 0 and 1 cause ~87% of the spread and
the tail is nearly constant. A 25-entry ROM indexed by (v0,v1) holding 1/K_head
plus ONE multiplier would remove the head component. Is the residual small
enough, and is the multiplier worth it?
"""
import math
from csd_encoder import enc_ff, achieved, gain_of, stage_gain, DIGIT_SETS

SPAN = math.pi / 4
n = 1500
dset = DIGIT_SETS["as-built {0,+-1,+-1.5}"]

print("=" * 76)
print("Residual gain spread and vector error after head compensation")
print("=" * 76)
print(f"{'N':>3} {'comp stages':>12} {'ROM entries':>12} {'gain spread':>13} {'vec err':>10}")
for N in (5, 6):
    for H in (0, 1, 2, 3):
        rows = []
        for j in range(n):
            th = -SPAN + 2 * SPAN * j / (n - 1)
            d = enc_ff(th, N, dset, P=2)
            Khead = math.prod(stage_gain(v, i) for i, v in enumerate(d[:H]))
            Ktail = gain_of(d) / (Khead if Khead else 1.0)
            rows.append((th, achieved(d), Ktail))
        Ks = [k for _, _, k in rows]
        Knom = math.sqrt(min(Ks) * max(Ks))
        tot = 0.0
        for th, got, K in rows:
            s = K / Knom
            tot += (s*math.cos(got)-math.cos(th))**2 + (s*math.sin(got)-math.sin(th))**2
        ve = math.sqrt(tot/len(rows))
        print(f"{N:>3} {H:>12} {len(dset)**H:>12} "
              f"{100*(max(Ks)/min(Ks)-1):>12.2f}% {ve:>10.5f}")
    print()

print("=" * 76)
print("Area arithmetic, W=9, from the measured synthesis data")
print("=" * 76)
print("""  rope_mult_pair W=9 = 1953 um2 for 4 multipliers + 2 adders + registers
    => roughly 400-450 um2 per W=9 multiplier

  CSD UD N=5        (measured)              2330 um2   vec err 0.2486
  CSD UD N=5 + 1 compensation multiplier   ~2750 um2   vec err ~0.001 (H=2)
  Binary UD N=10    (measured, W=8)         2728 um2   vec err 0.00111
  Binary UD N=10 needs NO compensation -- K is constant (C1)

  So a gain-corrected CSD lands at roughly the same area as Binary UD, with
  an extra multiplier in the critical path and a 25-entry ROM, to reach the
  same accuracy Binary UD gets for free.""")
