import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS (LOCK THESE)
# ======================================================
N_STAGES  = 10      # Number of UD-CORDIC stages
FRAC_BITS = 16      # Fixed-point fractional bits for angle

# ======================================================
# GOLDEN REFERENCE (FLOAT ROTATION)
# ======================================================
def float_rotation(x, y, theta):
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    )

# ======================================================
# ANGLE → FIXED-POINT BINARY
# Angle assumed in [0, pi/2]
# ======================================================
def angle_to_fixed(theta, frac_bits=FRAC_BITS):
    """
    Normalize angle to [0,1) w.r.t pi/2
    """
    theta_norm = theta / (np.pi / 2)
    return int(theta_norm * (1 << frac_bits))

# ======================================================
# BINARY → CSD CONVERSION
# Produces digits in {-1, 0, +1}
# LSB first (stage 0)
# ======================================================
def binary_to_csd(x, n_bits):
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

# ======================================================
# CSD-UD-CORDIC CORE
# ======================================================
def csd_ud_cordic(x, y, csd_digits):
    xi, yi = x, y
    gain = 1.0

    x_path = [xi]
    y_path = [yi]

    for i, d in enumerate(csd_digits):
        shift = 2.0 ** (-(i + 1))

        if d == +1:
            gain *= np.sqrt(1 + shift * shift)
            xi, yi = xi - yi * shift, yi + xi * shift

        elif d == -1:
            gain *= np.sqrt(1 + shift * shift)
            xi, yi = xi + yi * shift, yi - xi * shift

        # d == 0 → SKIP (no operation)

        x_path.append(xi)
        y_path.append(yi)

    # Gain compensation
    xi /= gain
    yi /= gain

    return xi, yi, x_path, y_path

# ======================================================
# MAIN TEST (RoPE-LIKE)
# ======================================================
def main():
    np.random.seed(0)

    # Example input vector
    x_in, y_in = 0.8, 0.2

    # Example RoPE angle (range-reduced)
    theta = np.deg2rad(30)
    theta_test = theta % (np.pi / 2)  # Quadrant mapper assumed

    # --------------------------------------------------
    # Golden reference
    # --------------------------------------------------
    x_ref, y_ref = float_rotation(x_in, y_in, theta_test)

    # --------------------------------------------------
    # CSD-UD-CORDIC
    # --------------------------------------------------
    theta_fp = angle_to_fixed(theta_test)
    csd_digits = binary_to_csd(theta_fp, N_STAGES)

    x_ud, y_ud, x_path, y_path = csd_ud_cordic(x_in, y_in, csd_digits)

    # --------------------------------------------------
    # Error metrics
    # --------------------------------------------------
    angle_ref = np.arctan2(y_ref, x_ref)
    angle_ud  = np.arctan2(y_ud,  x_ud)

    angle_error = np.rad2deg(angle_ud - angle_ref)
    mag_error   = np.hypot(x_ud, y_ud) - np.hypot(x_ref, y_ref)

    print("===== CSD-UD-CORDIC RESULTS =====")
    print(f"CSD Digits        : {csd_digits}")
    print(f"Expected (Float)  : ({x_ref:.6f}, {y_ref:.6f})")
    print(f"CSD-UD-CORDIC     : ({x_ud:.6f}, {y_ud:.6f})")
    print(f"Angle Error (°)   : {angle_error:.4f}")
    print(f"Magnitude Error   : {mag_error:.6e}")
    print(f"Active Rotations  : {sum(1 for d in csd_digits if d != 0)} / {N_STAGES}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(7,7))
    plt.arrow(0,0,x_in,y_in,width=0.01,label="Input",color="blue")
    plt.arrow(0,0,x_ref,y_ref,width=0.01,label="Expected",color="green")
    plt.arrow(0,0,x_ud,y_ud,width=0.01,label="CSD-UD-CORDIC",color="red")
    plt.plot(x_path,y_path,'o--',alpha=0.6,label="UD-CORDIC Path")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        f"CSD-UD-CORDIC Rotation\n"
        f"Angle Error = {angle_error:.2f}°, "
        f"Active = {sum(d!=0 for d in csd_digits)}"
    )
    plt.legend()
    plt.show()

# ======================================================
if __name__ == "__main__":
    main()
