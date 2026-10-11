"""T1.9 — practical CSD encoder: ROM-decoded head + geometric tail.

Replaces the 25-way brute force with a small ROM indexed by the top bits of
theta. Head digits are a monotone function of theta, so a coarse lookup picks
them correctly; the tail is geometric and reads straight off the residual.

  ROM_HEAD : 2^HB entries -> (v0, v1) packed, 6 bits each
  ROM_ANG  : 25 entries   -> head angle constant, FB bits
  then 1 subtractor + wired tail extraction
"""
import math
from csd_encoder import enc_ff, stage_angle, stage_gain, DIGIT_SETS

SPAN = math.pi / 4
dset = DIGIT_SETS["as-built {0,+-1,+-1.5}"]
DIDX = {v: i for i, v in enumerate(dset)}

def build_head_rom(N, HB, P=2):
    rom = []
    for b in range(1 << HB):
        th = -SPAN + 2 * SPAN * (b + 0.5) / (1 << HB)
        d = enc_ff(th, N, dset, P=P)
        rom.append(tuple(d[:P]))
    return rom

def enc_lut(theta, N, rom, HB, P=2, FB=14):
    b = min(max(int((theta + SPAN) / (2 * SPAN) * (1 << HB)), 0), (1 << HB) - 1)
    head = list(rom[b])
    R = theta - sum(stage_angle(v, i) for i, v in enumerate(head))
    digs = list(head)
    q = 1 << FB
    Rq = int(round(R * q))                     # fixed point residual
    for i in range(P, N):
        wq = q >> (2 * i)                      # weight 4^-i in Q.FB
        if wq == 0:
            digs.append(0); continue
        ratio = Rq / wq
        v = min(dset, key=lambda d: abs(d - ratio))
        digs.append(v)
        Rq -= int(round(v * wq))
    return digs

def stats(N, HB, FB=14, n=1500):
    rom = build_head_rom(N, HB)
    errs, gains = [], []
    for j in range(n):
        th = -SPAN + 2 * SPAN * j / (n - 1)
        d = enc_lut(th, N, rom, HB, FB=FB)
        errs.append(abs(sum(stage_angle(v, i) for i, v in enumerate(d)) - th))
        gains.append(math.prod(stage_gain(v, i) for i, v in enumerate(d)))
    return (math.degrees(math.sqrt(sum(e*e for e in errs)/len(errs))),
            math.degrees(max(errs)), min(gains), max(gains))

print("=" * 74)
print("CSD encoder: ROM head + geometric tail  (angular error, deg)")
print("=" * 74)
print(f"{'N':>3} {'HB':>4} {'ROM entries':>12} {'RMS':>9} {'max':>9} "
      f"{'25-way brute':>13}")
for N in (5, 6):
    ref = None
    for HB in (4, 5, 6, 7):
        rms, mx, _, _ = stats(N, HB)
        if ref is None:
            import statistics
            e = []
            for j in range(1500):
                th = -SPAN + 2*SPAN*j/1499
                d = enc_ff(th, N, dset, P=2)
                e.append(abs(sum(stage_angle(v,i) for i,v in enumerate(d))-th))
            ref = math.degrees(math.sqrt(sum(x*x for x in e)/len(e)))
        print(f"{N:>3} {HB:>4} {1<<HB:>12} {rms:>9.4f} {mx:>9.4f} {ref:>13.4f}")
    print()
