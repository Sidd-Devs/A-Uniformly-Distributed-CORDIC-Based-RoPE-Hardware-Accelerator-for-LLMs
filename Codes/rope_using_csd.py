import numpy as np

# ----------------------------
# Float RoPE
# ----------------------------
def rope_float(x, y, theta):
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
    )

# ----------------------------
# Angle → CSD digits
# ----------------------------
def angle_to_csd(theta, stages):
    digits = []
    rem = theta
    for i in range(stages):
        w = 2 ** (-(i + 1))
        if rem > w / 2:
            digits.append(1)
            rem -= w
        elif rem < -w / 2:
            digits.append(-1)
            rem += w
        else:
            digits.append(0)
    return digits

# ----------------------------
# CSD-UD-CORDIC
# ----------------------------
def csd_ud_cordic(x, y, digits):
    for i, d in enumerate(digits):
        shift = 2 ** (-(i + 1))
        if d == 1:
            x, y = x - y * shift, y + x * shift
        elif d == -1:
            x, y = x + y * shift, y - x * shift
    return x, y

# ----------------------------
# RoPE-CSD
# ----------------------------
if __name__ == "__main__":
    x, y = 0.8, 0.2
    theta = np.deg2rad(30)
    STAGES = 10

    digits = angle_to_csd(theta, STAGES)

    xf, yf = rope_float(x, y, theta)
    xc, yc = csd_ud_cordic(x, y, digits)

    print("CSD digits:", digits)
    print("Float RoPE:", xf, yf)
    print("CSD-RoPE  :", xc, yc)
