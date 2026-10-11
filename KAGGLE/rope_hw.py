"""Bit-accurate, vectorised hardware RoPE model (torch).

Mirrors the synthesized RTL exactly:
    RTL/rope_range_reduce.v   -> phase fold to +/- pi/4
    RTL/rope_angle_encoder.v  -> feed-forward digit encoder (no Z-path)
    RTL/binary_ud_cordic_core.v -> pipelined shift-add rotator, Q1.(W-2)

Verified against the iverilog golden vectors by validate_hw.py -- do not edit
one without re-running the other.

Variants
    binary : Binary UD-CORDIC. Gain K is a constant, so it folds into the
             1/sqrt(d_head) attention scale for free (contribution C1).
    csd    : CSD UD-CORDIC. Gain depends on the digit magnitudes, hence on the
             angle; `comp` selects how many head stages are compensated.
"""
import math
import torch

# ----------------------------------------------------------------- constants
def binary_weights(N):
    return [math.atan(2.0 ** -(i + 1)) for i in range(N)]

def binary_gain(N):
    return math.prod(math.hypot(1.0, 2.0 ** -(i + 1)) for i in range(N))

def encoder_constants(N, FB):
    """A, HEADSUM[0..3], TMAX — identical to the localparams in the RTL."""
    a = binary_weights(N)
    q = 1 << FB
    A = int(round(sum(a) * q))
    hs = [0,
          int(round(a[1] * q)),
          int(round(a[0] * q)),
          int(round((a[0] + a[1]) * q))]
    tmax = int(round(sum(a[2:]) * q))
    return A, hs, tmax


# ------------------------------------------------------------ range reduction
def range_reduce(theta):
    """theta (any magnitude, float64) -> (quadrant 0..3, residual in +/-pi/4)."""
    two_pi = 2.0 * math.pi
    frac = torch.remainder(theta / two_pi, 1.0)          # mod 2*pi, free in HW
    scaled = frac * 4.0
    quad = torch.floor(scaled)
    rem = scaled - quad                                   # [0,1)
    quad = torch.where(rem >= 0.5, quad + 1.0, quad)      # recentre
    rem = torch.where(rem >= 0.5, rem - 1.0, rem)         # [-0.5, 0.5)
    return (quad.to(torch.int64) % 4), rem * (math.pi / 2.0)


# ------------------------------------------------------------------- encoder
def encode_binary(theta_res, N, FB):
    """Vectorised twin of rope_angle_encoder.v. Returns bool[..., N]."""
    A, hs, tmax = encoder_constants(N, FB)
    tq = torch.round(theta_res * (1 << FB)).to(torch.int64)
    s = (tq + A) >> 1

    b0 = s >= hs[2]
    b1 = torch.where(s >= hs[3], torch.ones_like(s, dtype=torch.bool),
         torch.where(s >= hs[2], torch.zeros_like(s, dtype=torch.bool),
                     s >= hs[1]))
    idx = (b0.to(torch.int64) << 1) | b1.to(torch.int64)
    headsum = torch.zeros_like(s)
    for k in range(4):
        headsum = torch.where(idx == k, torch.full_like(s, hs[k]), headsum)

    r = torch.clamp(s - headsum, min=0, max=tmax)
    bits = [b0, b1]
    for j in range(2, N):
        sh = FB - j - 1
        bits.append(((r >> sh) & 1).to(torch.bool) if sh >= 0
                    else torch.zeros_like(s, dtype=torch.bool))
    return torch.stack(bits, dim=-1)


# ------------------------------------------------------------------- rotator
def _wrap(v, W):
    m = 1 << W
    return ((v + (m >> 1)) % m) - (m >> 1)

def rotate_binary(xq, yq, bits, N, W):
    """Twin of binary_ud_cordic_core.v, integer Q1.(W-2)."""
    x, y = xq, yq
    for i in range(N):
        d = bits[..., i]
        xs = torch.floor_divide(x, 1 << (i + 1))     # arithmetic >>> (floor)
        ys = torch.floor_divide(y, 1 << (i + 1))
        xn = torch.where(d, x - ys, x + ys)
        yn = torch.where(d, y + xs, y - xs)
        x, y = _wrap(xn, W), _wrap(yn, W)
    return x, y


# ----------------------------------------------------------------- CSD model
CSD_DIGITS = [0.0, 1.0, -1.0, 1.5, -1.5]

def _csd_stage_angle(v, i):
    return math.atan(v * 2.0 ** (-2 * i))

def _csd_stage_gain(v, i):
    return math.hypot(1.0, v * 2.0 ** (-2 * i))

def _csd_encode_np(theta, N, P=2):
    """Reference CSD encoder (T1.9 feed-forward P=2). Scalar; used to build the
    lookup that the vectorised path consumes."""
    best = None
    heads = [[]]
    for i in range(min(P, N)):
        heads = [h + [v] for h in heads for v in CSD_DIGITS]
    for head in heads:
        R = theta - sum(_csd_stage_angle(v, i) for i, v in enumerate(head))
        digs = list(head)
        for i in range(len(head), N):
            w = 4.0 ** -i
            v = min(CSD_DIGITS, key=lambda d: abs(d - R / w))
            digs.append(v); R -= v * w
        got = sum(_csd_stage_angle(v, i) for i, v in enumerate(digs))
        if best is None or abs(got - theta) < best[0]:
            best = (abs(got - theta), digs)
    return best[1]

class CSDTable:
    """Precomputed angle->(achieved angle, gain) map on a fine grid.

    The CSD rotator is modelled at the angle/gain level rather than bit level:
    what the LLM sees is the rotated vector, and this captures both the angular
    error and the angle-dependent gain (S-13), which is the point of the
    experiment. `comp` selects head-stage gain compensation.
    """
    def __init__(self, N=5, comp="none", grid=4096, device="cpu"):
        H = {"none": 0, "H1": 1, "H2": 2}[comp]
        span = math.pi / 4
        ang, gain = [], []
        for k in range(grid):
            th = -span + 2 * span * (k + 0.5) / grid
            d = _csd_encode_np(th, N)
            ang.append(sum(_csd_stage_angle(v, i) for i, v in enumerate(d)))
            Kfull = math.prod(_csd_stage_gain(v, i) for i, v in enumerate(d))
            Khead = math.prod(_csd_stage_gain(v, i) for i, v in enumerate(d[:H]))
            gain.append(Kfull / (Khead if Khead else 1.0))
        self.grid = grid
        self.span = span
        self.ang = torch.tensor(ang, dtype=torch.float64, device=device)
        self.gain = torch.tensor(gain, dtype=torch.float64, device=device)
        self.Knom = math.sqrt(float(self.gain.min()) * float(self.gain.max()))

    def apply(self, theta_res):
        idx = torch.clamp(((theta_res + self.span) / (2 * self.span)
                           * self.grid).to(torch.int64), 0, self.grid - 1)
        return self.ang[idx], self.gain[idx] / self.Knom


# ------------------------------------------------------------------ top level
class HWRoPE:
    """Drop-in hardware RoPE.

    variant   'binary' | 'csd' | 'exact'
    W         datapath word length, Q1.(W-2)   (F = W-2 fractional bits)
    N         rotator stages
    FB        angle datapath fractional bits
    comp      CSD gain compensation: 'none' | 'H1' | 'H2'
    fold_gain divide out the constant gain. For Binary UD this is free in
              hardware (C1): K is angle-independent so it folds into the
              1/sqrt(d_head) attention scale.
    """
    def __init__(self, variant="binary", W=9, N=10, FB=12,
                 comp="none", fold_gain=True, device="cpu"):
        self.variant, self.W, self.N, self.FB = variant, W, N, FB
        self.fold_gain = fold_gain
        self.scale = 1 << (W - 2)
        self.K = binary_gain(N)
        self.csd = CSDTable(N=N, comp=comp, device=device) if variant == "csd" else None

    def __call__(self, x_even, x_odd, theta):
        """x_* : float tensors. theta : float64 tensor of rotation angles."""
        if self.variant == "exact":
            c, s = torch.cos(theta), torch.sin(theta)
            return x_even * c - x_odd * s, x_even * s + x_odd * c

        quad, res = range_reduce(theta.to(torch.float64))

        if self.variant == "binary":
            bits = encode_binary(res, self.N, self.FB)
            xq = torch.clamp(torch.round(x_even.to(torch.float64) * self.scale),
                             -(1 << (self.W - 1)), (1 << (self.W - 1)) - 1).to(torch.int64)
            yq = torch.clamp(torch.round(x_odd.to(torch.float64) * self.scale),
                             -(1 << (self.W - 1)), (1 << (self.W - 1)) - 1).to(torch.int64)
            xo, yo = rotate_binary(xq, yq, bits, self.N, self.W)
            xr = xo.to(torch.float64) / self.scale
            yr = yo.to(torch.float64) / self.scale
            if self.fold_gain:
                xr, yr = xr / self.K, yr / self.K
        else:  # csd
            ang, g = self.csd.apply(res)
            c, s = torch.cos(ang), torch.sin(ang)
            xe = x_even.to(torch.float64); xd = x_odd.to(torch.float64)
            xr = (xe * c - xd * s) * g
            yr = (xe * s + xd * c) * g
            # datapath quantisation
            xr = torch.round(xr * self.scale) / self.scale
            yr = torch.round(yr * self.scale) / self.scale

        # undo the quadrant fold (free in hardware: swap and negate)
        q = quad
        xf = torch.where(q == 0, xr, torch.where(q == 1, -yr,
             torch.where(q == 2, -xr, yr)))
        yf = torch.where(q == 0, yr, torch.where(q == 1, xr,
             torch.where(q == 2, -yr, -xr)))
        return xf.to(x_even.dtype), yf.to(x_odd.dtype)
