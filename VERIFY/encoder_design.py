"""T1.7 — feed-forward angle encoder for UD-CORDIC RoPE.

Constraint: NO Z-path. The digit string must be derivable from theta by
combinational logic (compare / subtract / bit-select), never by a sequential
residual loop -- otherwise we have reinvented standard CORDIC.

Why the encoder in the paper fails
----------------------------------
Stage i rotates by a_i = atan(2^-(i+1)), NOT by 2^-(i+1). Rotations compose
additively, so the achieved angle is exactly sum_i d_i a_i with d_i in {-1,+1}.
Substituting d_i = 2 b_i - 1:

    S := (theta + A) / 2 = sum_i b_i a_i ,      A = sum_i a_i

Encoding is therefore a subset-sum in the weights {a_i}, which are only
APPROXIMATELY geometric:

    a_i / 2^-(i+1) = 0.9273, 0.9799, 0.9948, 0.9987, 0.9997, ...

The tail is geometric to <0.1%, but a_0 and a_1 are off by 7.3% and 2.0%.
Taking the raw bits of theta (paper Eq. 5) ignores this entirely.

Encoders compared
-----------------
  affine  : one scale+offset, then bit extraction.  1 multiplier-free map.
  prefix-P: brute-force the P non-geometric head stages (2^P candidates in
            parallel), clamp the residual, bit-extract the geometric tail.
  lutcorr : affine, plus a 2^L-entry ROM that cancels the systematic residual
            of the affine map.  1 small ROM + 1 adder.
  greedy  : sequential oracle. NOT synthesizable feed-forward; upper bound only.
"""
import math

def binary_weights(N):
    return [math.atan(2.0 ** -(i + 1)) for i in range(N)]

def achieved(bits, N, w=None):
    """Rotations compose additively, so this is exact."""
    w = w or binary_weights(N)
    return sum((1 if b else -1) * wi for b, wi in zip(bits, w))

def gain(N):
    return math.prod(math.hypot(1, 2.0 ** -(i + 1)) for i in range(N))

# ------------------------------------------------------------------- encoders
def enc_affine(theta, N):
    a = binary_weights(N); A = sum(a)
    I = round((theta + A) / (2 * A) * ((1 << N) - 1))
    I = max(0, min(I, (1 << N) - 1))
    return [(I >> (N - 1 - k)) & 1 for k in range(N)]

def enc_prefix(theta, N, P=2):
    """Brute-force head, clamped residual, bit-extract tail. Combinational."""
    a = binary_weights(N); A = sum(a)
    P = min(P, N)
    S = (theta + A) / 2.0
    tail = a[P:]; Tmax = sum(tail)
    best = None
    for cand in range(1 << P):
        head = [(cand >> (P - 1 - k)) & 1 for k in range(P)]
        R = S - sum(h * w for h, w in zip(head, a[:P]))
        R = min(max(R, 0.0), Tmax)          # clamp instead of discard
        tb, acc = [], R
        for w in tail:                      # geometric -> plain bit extraction
            if acc >= w:
                tb.append(1); acc -= w
            else:
                tb.append(0)
        bits = head + tb
        e = abs(achieved(bits, N, a) - theta)
        if best is None or e < best[0]:
            best = (e, bits)
    return best[1]

_LUT_CACHE = {}
def _build_lut(N, L, span):
    """Systematic error of the affine map, averaged per bin. Precomputed
    offline, stored as a 2^L-entry ROM."""
    key = (N, L, round(span, 9))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    nbin = 1 << L
    acc = [0.0] * nbin; cnt = [0] * nbin
    for j in range(4000):
        th = -span + 2 * span * j / 3999
        b = min(int((th + span) / (2 * span) * nbin), nbin - 1)
        acc[b] += achieved(enc_affine(th, N), N) - th
        cnt[b] += 1
    lut = [(acc[i] / cnt[i]) if cnt[i] else 0.0 for i in range(nbin)]
    _LUT_CACHE[key] = lut
    return lut

def enc_lutcorr(theta, N, L=4, span=math.pi / 4):
    lut = _build_lut(N, L, span)
    nbin = 1 << L
    b = min(max(int((theta + span) / (2 * span) * nbin), 0), nbin - 1)
    return enc_affine(theta - lut[b], N)

def enc_greedy(theta, N):
    a = binary_weights(N)
    bits, cur = [], 0.0
    for i in range(N):
        if abs(cur + a[i] - theta) <= abs(cur - a[i] - theta):
            bits.append(1); cur += a[i]
        else:
            bits.append(0); cur -= a[i]
    return bits

# ------------------------------------------------------------------ evaluation
SPAN = math.pi / 4          # residual after free quadrant folding

def ev(enc, N, n=4000, **kw):
    errs = []
    a = binary_weights(N)
    for j in range(n):
        th = -SPAN + 2 * SPAN * j / (n - 1)
        errs.append(abs(achieved(enc(th, N, **kw), N, a) - th))
    return (math.degrees(math.sqrt(sum(e * e for e in errs) / len(errs))),
            math.degrees(max(errs)))

print("=" * 80)
print("T1.7  Binary UD angle encoder — RMS angular error (deg)")
print("      theta swept over +/-45 deg (residual after free quadrant fold)")
print("=" * 80)
print(f"{'N':>3} | {'affine':>9} {'prefix P=2':>11} {'prefix P=3':>11} "
      f"{'lutcorr L=4':>12} {'lutcorr L=6':>12} | {'greedy*':>9}")
print("-" * 80)
for N in (4, 5, 6, 8, 10, 12, 14):
    row = [ev(enc_affine, N)[0], ev(enc_prefix, N, P=2)[0],
           ev(enc_prefix, N, P=3)[0], ev(enc_lutcorr, N, L=4)[0],
           ev(enc_lutcorr, N, L=6)[0], ev(enc_greedy, N)[0]]
    print(f"{N:>3} | {row[0]:>9.4f} {row[1]:>11.4f} {row[2]:>11.4f} "
          f"{row[3]:>12.4f} {row[4]:>12.4f} | {row[5]:>9.4f}")
print("\n* greedy is sequential (a Z-path). Shown as the achievable bound only.")

print("\nMax error, proposed encoders:")
print(f"{'N':>3} {'prefix P=2 RMS':>15} {'max':>9} {'lutcorr L=6 RMS':>17} {'max':>9}")
for N in (5, 8, 10, 12):
    r1, m1 = ev(enc_prefix, N, P=2)
    r2, m2 = ev(enc_lutcorr, N, L=6)
    print(f"{N:>3} {r1:>15.4f} {m1:>9.4f} {r2:>17.4f} {m2:>9.4f}")

print(f"\nConstant CORDIC gain (contribution C1), Binary UD:")
for N in (5, 8, 10, 12):
    K = gain(N)
    print(f"  N={N:>2}  K={K:.8f}  K^2={K*K:.8f}   <- angle-independent, "
          f"folds into 1/sqrt(d)")

print("\nWhy the head needs brute force but the tail does not:")
print(f"{'i':>3} {'a_i':>14} {'2^-(i+1)':>12} {'ratio':>10} {'note':>22}")
for i in range(7):
    a = math.atan(2.0 ** -(i + 1)); g = 2.0 ** -(i + 1)
    note = "non-geometric" if a / g < 0.995 else "geometric to <0.5%"
    print(f"{i:>3} {a:>14.8f} {g:>12.8f} {a/g:>10.6f} {note:>22}")
