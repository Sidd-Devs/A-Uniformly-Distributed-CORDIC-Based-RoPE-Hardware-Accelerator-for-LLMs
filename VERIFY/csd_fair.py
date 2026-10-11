"""Correcting the T1.9 area verdict: matched PRECISION, not matched stages."""
import math
from csd_encoder import stage_gain, DIGIT_SETS
dset = DIGIT_SETS["as-built {0,+-1,+-1.5}"]

print("="*72); print("1. Is the N/2 equivalence real? (measured, T1.9 feed-forward encoder)")
print("="*72)
print("   Binary UD N=10 : 0.063 deg angle RMS")
print("   CSD UD    N=5  : 0.043 deg angle RMS   -> YES, N/2 matches on ANGLE")

print("\n"+"="*72); print("2. Area at matched precision, W=9 (measured synthesis)")
print("="*72)
BIN10, CSD5, STD10 = 3068, 2330, 3292     # um2, W=9
print(f"   Std CORDIC N=10          {STD10:>6} um2")
print(f"   Binary UD  N=10          {BIN10:>6} um2   ({100*(1-BIN10/STD10):.1f}% vs std)")
print(f"   CSD UD     N=5           {CSD5:>6} um2   ({100*(1-CSD5/STD10):.1f}% vs std, "
      f"{100*(1-CSD5/BIN10):.1f}% vs Binary)")
print("   -> at matched precision CSD IS smaller. My earlier claim compared")
print("      CSD(Ext.) N=8/10/12 against std N=8/10/12, which is 2x the")
print("      precision on the CSD side. That comparison was not meaningful.")

print("\n"+"="*72); print("3. Cost of gain compensation — the real question")
print("="*72)
m9, m8 = 1953, 1471          # rope_mult_pair W=9, W=8 (measured)
k = (m9-m8)/(4*(81-64))
print(f"   per-multiplier area ~ k*W^2, k = {k:.2f} um2/bit^2 from mult_rope W=8 vs W=9")
print(f"   one W=9 multiplier   ~ {k*81:>6.0f} um2      TWO (x and y) ~ {2*k*81:.0f} um2")
print(f"   CSD5 + 2 multipliers ~ {CSD5+2*k*81:>6.0f} um2  vs Binary UD {BIN10} um2"
      f"  -> {100*((CSD5+2*k*81)/BIN10-1):+.1f}%")
print("   => with a FULL multiplier the advantage is gone. But:")

print("\n"+"="*72); print("4. Compensation does not need a multiplier")
print("="*72)
for H in (1, 2):
    vals = set()
    def rec(i, K):
        if i == H: vals.add(round(K, 6)); return
        for v in dset: rec(i+1, K*stage_gain(v, i))
    rec(0, 1.0)
    print(f"   H={H}: {len(vals)} distinct head gains -> 1/K in "
          f"{sorted(round(1/v,4) for v in vals)}")
print("""
   These are CONSTANTS. A constant multiply is a shift-add network (2-3 adders
   each), not a multiplier. H=1 needs a 3-way mux over 3 constant scalers on
   two datapaths -- order 100-150 um2, not 1000.
   H=1 leaves 7.3% gain spread (vector err 0.025); H=2 leaves 0.47% (0.0019).
   Whether 7.3% matters is an LLM-perplexity question, not a hardware one.""")

print("\n"+"="*72); print("REVISED VERDICT"); print("="*72)
print("""   CSD is NOT dead. Corrected position:
     - N/2 equivalence holds on angle; CSD N=5 is 24% smaller / 44% lower
       power than Binary UD N=10 at W=9.
     - It still cannot carry C1 (gain is angle-dependent) -- that stands.
     - But compensation via muxed constant shift-add scalers is far cheaper
       than the multiplier I assumed, and may preserve most of the win.
   ACTION: synthesise the compensation block and run the H=1 vs H=2 accuracy
   question through the LLM eval before choosing the headline architecture.""")
