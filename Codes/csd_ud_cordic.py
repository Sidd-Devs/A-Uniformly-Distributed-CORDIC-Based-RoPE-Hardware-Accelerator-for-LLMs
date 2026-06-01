import numpy as np
import matplotlib.pyplot as plt

# ======================================
# PARAMETERS
# ======================================
N_STAGES = 10

# ======================================
# FLOAT REFERENCE
# ======================================
def float_rotation(x, y, theta):
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
    )

# ======================================
# ANGLE → CSD DIGITS
# digits ∈ {-1, 0, +1}
# ======================================
def angle_to_csd(theta, n_stages):
    digits = []
    rem = theta

    for i in range(n_stages):
        w = 2.0 ** (-(i + 1))
        if rem > +w / 2:
            digits.append(1)
            rem -= w
        elif rem < -w / 2:
            digits.append(-1)
            rem += w
        else:
            digits.append(0)

    return digits

# ======================================
# CSD-UD-CORDIC (FLOAT, UNCOMPENSATED)
# ======================================
def csd_ud_cordic(x, y, digits):
    xi, yi = x, y
    px, py = [xi], [yi]

    for i, d in enumerate(digits):
        shift = 2.0 ** (-(i + 1))
        if d == 1:
            xi, yi = xi - yi * shift, yi + xi * shift
        elif d == -1:
            xi, yi = xi + yi * shift, yi - xi * shift

        px.append(xi)
        py.append(yi)

    return xi, yi, px, py

# ======================================
# MAIN
# ======================================
if __name__ == "__main__":

    # Input vector
    x_in, y_in = 0.8, 0.2

    # Test angle
    theta = np.deg2rad(30)

    # Float reference
    x_ref, y_ref = float_rotation(x_in, y_in, theta)

    # CSD-UD
    digits = angle_to_csd(theta, N_STAGES)
    x_csd, y_csd, px, py = csd_ud_cordic(x_in, y_in, digits)

    # ----------------------------------
    # PRINT RESULTS
    # ----------------------------------
    print("CSD digits (math) :", digits)
    print("Float ref         :", (x_ref, y_ref))
    print("CSD-UD output     :", (x_csd, y_csd))

    # ----------------------------------
    # VIVADO-READY ENCODING
    # +1 → 01,  -1 → 11,  0 → 00
    # LSB = stage 0
    # ----------------------------------
    enc = {1: "01", -1: "11", 0: "00"}
    csd_bits_verilog = "".join(enc[d] for d in digits[::-1])

    print("\nVivado CSD digits (LSB -> MSB):")
    print(f"{2*N_STAGES}'b{csd_bits_verilog}")

    # ----------------------------------
    # PLOT
    # ----------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot(px, py, "o--", alpha=0.4, label="CSD Path")

    plt.scatter(
        [x_ref], [y_ref],
        color="blue", s=80, label="Float"
    )

    plt.scatter(
        [x_csd], [y_csd],
        color="orange", marker="x",
        s=120, linewidths=3, label="CSD-UD"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("CSD-UD-CORDIC Rotation")
    plt.show()
