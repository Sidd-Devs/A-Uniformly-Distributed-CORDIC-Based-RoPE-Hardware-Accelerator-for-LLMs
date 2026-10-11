"""Why the RoPE base sets the precision requirement — derived, not measured.

For a small residual angle the rotation is x' ~ x - theta*y. With the input
block-normalised so |y| ~ 1, that perturbation is representable only when
theta > 2^-F. Below the LSB the pair does not rotate at all: its positional
signal is discarded rather than degraded, which is why the end-to-end failure
is a cliff rather than a slope.

This needs no model and no GPU -- it is a property of the frequency schedule
Theta_i = b^(-2i/d) alone, which is what makes it a mechanism rather than a
correlation across model families.
"""
import math

D, MAXPOS, STEP = 128, 2048, 16

def lost_fraction(base, F, d=D, maxpos=MAXPOS, step=STEP):
    lsb = 2.0 ** -F
    lost = tot = 0
    for i in range(d // 2):
        th_i = base ** (-2 * i / d)
        for m in range(1, maxpos, step):
            a = abs(((m * th_i + math.pi) % (2 * math.pi)) - math.pi)
            tot += 1
            lost += (a < lsb)
    return 100.0 * lost / tot

if __name__ == "__main__":
    bases = [10.0 ** e for e in (4, 5, 6)]
    print("Share of (position, dimension) pairs whose rotation quantises to zero")
    print(f"  d = {D}, positions 1..{MAXPOS} step {STEP}\n")
    print(f"{'F':>3} {'LSB':>10} | " + " ".join(f"{'b=1e'+str(e):>9}" for e in (4,5,6)))
    print("-" * 52)
    for F in (14, 12, 10, 9, 8, 7, 6):
        vals = [lost_fraction(b, F) for b in bases]
        print(f"{F:>3} {2.0**-F:>10.5f} | " + " ".join(f"{v:>8.2f}%" for v in vals))
    print("""
Measured cliffs for comparison:
  base 1e4 (LLaMA-2, Mistral, Phi-3): graceful to F=7, knee at F=6
  base 1e6 (Qwen2):                   +0.10% at F=10, +6.36% at F=9, +131% at F=8

The b=1e6 column crosses ~6% loss between F=10 and F=9, and ~11% at F=8 --
tracking the measured collapse. The b=1e4 column stays under 1% throughout.""")
