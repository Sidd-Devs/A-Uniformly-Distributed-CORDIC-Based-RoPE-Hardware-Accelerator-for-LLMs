import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS (LOCK THESE)
# -----------------------------
N_STAGES = 12
FRAC_BITS = 14  # fixed-point fractional bits

# -----------------------------
# FLOAT (GROUND TRUTH)
# -----------------------------
def rope_float(x, y, theta):
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    )

# -----------------------------
# Binary → CSD conversion
# Produces digits in {-1, 0, +1}
# -----------------------------
def binary_to_csd(x, n_bits):
    """
    Convert signed fixed-point integer x
    into CSD digits (LSB first)
    """
    csd = []
    carry = 0

    for i in range(n_bits):
        bit = ((x >> i) & 1) + carry

        if bit == 0:
            csd.append(0)
            carry = 0
        elif bit == 1:
            csd.append(+1)
            carry = 0
        elif bit == 2:
            csd.append(0)
            carry = 1
        elif bit == 3:
            csd.append(-1)
            carry = 1

    return csd[:n_bits]

# -----------------------------
# UD-CORDIC with CSD control
# -----------------------------
def ud_cordic_csd(x, y, csd):
    xi, yi = x, y
    x_hist = [xi]
    y_hist = [yi]

    for i, d in enumerate(csd):
        if d == +1:
            xi, yi = xi - yi * (2**-i), yi + xi * (2**-i)
        elif d == -1:
            xi, yi = xi + yi * (2**-i), yi - xi * (2**-i)
        else:
            pass  # SKIP (no operation)

        x_hist.append(xi)
        y_hist.append(yi)

    # Gain compensation
    K = np.prod([1 / np.sqrt(1 + 2 ** (-2 * i)) for i in range(len(csd))])
    return xi * K, yi * K, x_hist, y_hist

# -----------------------------
# TEST INPUT
# -----------------------------
x_in, y_in = 0.8, 0.2
theta = np.deg2rad(30)

# Fixed-point angle representation (normalized)
theta_norm = theta / (np.pi / 2)
theta_fp = int(theta_norm * (1 << FRAC_BITS))

# -----------------------------
# PIPELINE
# -----------------------------
csd_digits = binary_to_csd(theta_fp, N_STAGES)

x_exp, y_exp = rope_float(x_in, y_in, theta)
x_ud, y_ud, x_path, y_path = ud_cordic_csd(x_in, y_in, csd_digits)

# -----------------------------
# ERROR METRICS
# -----------------------------
angle_exp = np.arctan2(y_exp, x_exp)
angle_ud  = np.arctan2(y_ud, x_ud)

angle_error = np.rad2deg(angle_ud - angle_exp)
mag_error = np.hypot(x_ud, y_ud) - np.hypot(x_exp, y_exp)

print("===== RESULTS =====")
print(f"CSD Digits       : {csd_digits}")
print(f"Expected (Float) : ({x_exp:.6f}, {y_exp:.6f})")
print(f"UD-CORDIC Output : ({x_ud:.6f}, {y_ud:.6f})")
print(f"Angle Error (°)  : {angle_error:.4f}")
print(f"Mag Error        : {mag_error:.6e}")

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(7,7))
plt.arrow(0,0,x_in,y_in,width=0.01,label="Input",color='blue')
plt.arrow(0,0,x_exp,y_exp,width=0.01,label="Expected",color='green')
plt.arrow(0,0,x_ud,y_ud,width=0.01,label="UD-CORDIC (CSD)",color='red')
plt.plot(x_path,y_path,'o--',alpha=0.6,label="UD-CORDIC Path")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title(f"CSD UD-CORDIC RoPE\nAngle Error = {angle_error:.3f}°")
plt.show()
