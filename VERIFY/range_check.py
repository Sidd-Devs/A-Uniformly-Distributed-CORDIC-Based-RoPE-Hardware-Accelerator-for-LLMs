import math
print("Reachable rotation range (all digits at max, infinite precision)")
for N in (4,5,6,8,10,12):
    b = sum(math.atan(2.0**-(i+1)) for i in range(N))
    c15 = sum(math.atan(2.0**(-2*i)+2.0**(-(2*i+1))) for i in range(N))
    c20 = sum(math.atan(2*2.0**(-2*i)) for i in range(N))
    print(f"  N={N:>2}  BinaryUD +/-{math.degrees(b):6.2f} deg   "
          f"CSD(1.5) +/-{math.degrees(c15):6.2f} deg   CSD(2.0) +/-{math.degrees(c20):6.2f} deg")
print("\n  RoPE needs [0, 2pi) = 360 deg  ->  range reduction is MANDATORY for every variant")

print("\nCORDIC gain K (uncompensated), Binary UD:")
for N in (4,5,6,8,10,12):
    K = math.prod(math.hypot(1, 2.0**-(i+1)) for i in range(N))
    print(f"  N={N:>2}  K={K:.6f}   K^2={K*K:.6f}  (folds into 1/sqrt(d) attention scale)")
print("\nCSD gain K (digit-dependent -> NOT constant):")
for N in (5,10):
    lo = math.prod(1.0 for i in range(N))
    hi = math.prod(math.hypot(1, 2.0**(-2*i)+2.0**(-(2*i+1))) for i in range(N))
    print(f"  N={N:>2}  K ranges {lo:.6f} .. {hi:.6f}   <-- varies with angle, cannot fold")
