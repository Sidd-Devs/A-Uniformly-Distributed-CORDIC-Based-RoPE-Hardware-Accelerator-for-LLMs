"""T1.6 — bit-accurate golden models for the five rope_*_pair modules.

Each model mirrors its RTL exactly: same truncation points, same wrap-around,
same arithmetic-shift (floor) semantics. The LUT and ATAN tables are parsed
out of the RTL rather than retyped, so the two cannot drift apart.

Emits one vector file per module for VERIFY/tb_pairs.v.
"""
import random, re, os

RTL = os.path.join(os.path.dirname(__file__), "..", "RTL")

def wrap(v, W):
    m = 1 << W
    v &= m - 1
    return v - m if v >= (m >> 1) else v

def parse_table(fname, func_name):
    """Pull the `idx: base = N;` case table out of a Verilog function."""
    src = open(os.path.join(RTL, fname)).read()
    body = src[src.index(func_name):]
    body = body[:body.index("endfunction")]
    pairs = re.findall(r"(\d+)\s*:\s*\w+\s*=\s*(\d+)\s*;", body)
    t = {}
    for k, v in pairs:
        t[int(k)] = int(v)
    return [t[i] for i in range(len(t))]

SIN_BASE  = parse_table("rope_lut_pair.v",       "get_sin")
ATAN_BASE = parse_table("cordic_standard_core.v", "get_atan")

def scale16(base, W):
    return (base >> (16 - W)) if 16 > W else (base << (W - 16))

# --------------------------------------------------------------- the models
def rot_mul(xe, xo, sv, cv, W):
    """Shared by mult and LUT: 4 products, truncate Q2.2F -> Q1.F, 2 adds."""
    F = W - 2
    tr = lambda p: wrap(p >> F, W)
    return (wrap(tr(cv * xe) - tr(sv * xo), W),
            wrap(tr(sv * xe) + tr(cv * xo), W))

def mult_pair(xe, xo, sv, cv, W):
    return rot_mul(xe, xo, sv, cv, W)

def lut_pair(xe, xo, angle_index, W, ADDR_W=7):
    quad = (angle_index >> ADDR_W) & 3
    addr = angle_index & ((1 << ADDR_W) - 1)
    comp = ((1 << ADDR_W) - 1) - addr
    g = lambda i: wrap(scale16(SIN_BASE[i], W), W)
    if   quad == 0: sv, cv =  g(addr),  g(comp)
    elif quad == 1: sv, cv =  g(comp), -g(addr)
    elif quad == 2: sv, cv = -g(addr), -g(comp)
    else:           sv, cv = -g(comp),  g(addr)
    return rot_mul(xe, xo, wrap(sv, W), wrap(cv, W), W)

def binary_pair(xe, xo, bits, W, N):
    x, y = xe, xo
    for i in range(N):
        xs, ys = x >> (i + 1), y >> (i + 1)
        if bits[i]: xn, yn = x - ys, y + xs
        else:       xn, yn = x + ys, y - xs
        x, y = wrap(xn, W), wrap(yn, W)
    return x, y

def booth(b2, N):
    d = []
    for i in range(N):
        bm = 0 if i == 0 else b2[2 * i - 1]
        d.append(-2 * b2[2 * i + 1] + b2[2 * i] + bm)
    return d

def csd_pair(xe, xo, b2, W, N):
    x, y = xe, xo
    for i, v in enumerate(booth(b2, N)):
        s1, s2 = 2 * i, 2 * i + 1
        xs1, xs2 = wrap(x >> s1, W), wrap(x >> s2, W)
        ys1, ys2 = wrap(y >> s1, W), wrap(y >> s2, W)
        if   v ==  2: xc, yc =  (xs1 + xs2),  (ys1 + ys2)
        elif v ==  1: xc, yc =   xs1,          ys1
        elif v ==  0: xc, yc =   0,            0
        elif v == -1: xc, yc =  -xs1,         -ys1
        else:         xc, yc = -(xs1 + xs2), -(ys1 + ys2)
        xc, yc = wrap(xc, W), wrap(yc, W)
        x, y = wrap(x - yc, W), wrap(y + xc, W)
    return x, y

def cordic_pair(xe, xo, theta, W, N):
    x, y, z = xe, xo, theta
    for i in range(N):
        neg = z < 0                       # z_in[W-1]
        xs, ys = x >> i, y >> i
        a = wrap(scale16(ATAN_BASE[i] if i < len(ATAN_BASE) else 0, W), W)
        if neg: xn, yn, zn = x + ys, y - xs, z + a
        else:   xn, yn, zn = x - ys, y + xs, z - a
        x, y, z = wrap(xn, W), wrap(yn, W), wrap(zn, W)
    return x, y

# ------------------------------------------------------------------ emitters
W, N_BIN, N_CSD, N_STD, AW = 12, 10, 5, 10, 7
LIM = 1 << (W - 3)
hx  = lambda v, w=W: f"{v & ((1 << w) - 1):0{(w + 3) // 4}x}"

def emit(name, rows):
    with open(os.path.join(os.path.dirname(__file__), name), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  {name:<26} {len(rows)} vectors")

random.seed(11)
print(f"golden vectors (W={W}, sin table {len(SIN_BASE)}, atan table {len(ATAN_BASE)})")

rows = []
for _ in range(400):
    xe, xo = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    sv, cv = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    ye, yo = mult_pair(xe, xo, sv, cv, W)
    rows.append(f"{hx(xe)} {hx(xo)} {hx(sv)} {hx(cv)} {hx(ye)} {hx(yo)}")
emit("gv_mult.txt", rows)

rows = []
for _ in range(400):
    xe, xo = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    ai = random.randint(0, (1 << (AW + 2)) - 1)
    ye, yo = lut_pair(xe, xo, ai, W, AW)
    rows.append(f"{hx(xe)} {hx(xo)} {hx(ai, AW + 2)} {hx(ye)} {hx(yo)}")
emit("gv_lut.txt", rows)

rows = []
for _ in range(400):
    xe, xo = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    bits = [random.randint(0, 1) for _ in range(N_BIN)]
    word = sum(b << i for i, b in enumerate(bits))
    ye, yo = binary_pair(xe, xo, bits, W, N_BIN)
    rows.append(f"{hx(xe)} {hx(xo)} {hx(word, N_BIN)} {hx(ye)} {hx(yo)}")
emit("gv_binary.txt", rows)

rows = []
for _ in range(400):
    xe, xo = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    b2 = [random.randint(0, 1) for _ in range(2 * N_CSD)]
    word = sum(b << i for i, b in enumerate(b2))
    ye, yo = csd_pair(xe, xo, b2, W, N_CSD)
    rows.append(f"{hx(xe)} {hx(xo)} {hx(word, 2 * N_CSD)} {hx(ye)} {hx(yo)}")
emit("gv_csd.txt", rows)

rows = []
for _ in range(400):
    xe, xo = random.randint(-LIM, LIM), random.randint(-LIM, LIM)
    th = random.randint(-(1 << (W - 2)), (1 << (W - 2)))
    ye, yo = cordic_pair(xe, xo, th, W, N_STD)
    rows.append(f"{hx(xe)} {hx(xo)} {hx(th)} {hx(ye)} {hx(yo)}")
emit("gv_cordic.txt", rows)
