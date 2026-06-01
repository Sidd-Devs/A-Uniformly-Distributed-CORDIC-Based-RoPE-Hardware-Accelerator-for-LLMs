import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================

N_STAGES = 10
FRAC_BITS = 16
NUM_TESTS = 10000

# ======================================================
# FLOAT REFERENCE ROTATION
# ======================================================

def float_rotation(x, y, theta):

    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)

    return x_new, y_new


# ======================================================
# ANGLE → FIXED POINT
# ======================================================

def angle_to_fixed(theta, frac_bits=FRAC_BITS):

    theta_norm = theta / (np.pi / 2)

    return int(theta_norm * (1 << frac_bits))


# ======================================================
# BINARY → CSD CONVERSION
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
# CSD-UD-CORDIC ROTATION
# ======================================================

def csd_ud_cordic(x, y, digits):

    xi, yi = x, y

    gain = 1.0

    x_path = [xi]
    y_path = [yi]

    for i, d in enumerate(digits[:N_STAGES]):

        shift = 2.0 ** (-(i + 1))

        if d == +1:

            gain *= np.sqrt(1 + shift * shift)

            x_next = xi - yi * shift
            y_next = yi + xi * shift

            xi, yi = x_next, y_next

        elif d == -1:

            gain *= np.sqrt(1 + shift * shift)

            x_next = xi + yi * shift
            y_next = yi - xi * shift

            xi, yi = x_next, y_next

        x_path.append(xi)
        y_path.append(yi)

    xi /= gain
    yi /= gain

    return xi, yi, x_path, y_path


# ======================================================
# ERROR METRICS
# ======================================================

def compute_metrics(x_ref, y_ref, x_pred, y_pred):

    vec_ref = np.stack([x_ref, y_ref], axis=1)
    vec_pred = np.stack([x_pred, y_pred], axis=1)

    diff = vec_ref - vec_pred

    mse = np.mean(diff ** 2)

    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(diff))

    max_error = np.max(np.abs(diff))

    mag_ref = np.sqrt(x_ref**2 + y_ref**2)
    mag_pred = np.sqrt(x_pred**2 + y_pred**2)

    mag_error = np.mean(np.abs(mag_pred - mag_ref))

    angle_ref = np.arctan2(y_ref, x_ref)
    angle_pred = np.arctan2(y_pred, x_pred)

    angle_error = np.mean(np.abs(angle_pred - angle_ref))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAX_ERROR": max_error,
        "MAG_ERROR": mag_error,
        "ANGLE_ERROR_RAD": angle_error,
        "ANGLE_ERROR_DEG": np.rad2deg(angle_error)
    }


# ======================================================
# MONTE CARLO TEST
# ======================================================

def monte_carlo_test():

    x_ref_list = []
    y_ref_list = []

    x_csd_list = []
    y_csd_list = []

    active_rotations = []

    for _ in range(NUM_TESTS):

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        theta = np.random.uniform(0, np.pi/2)

        x_ref, y_ref = float_rotation(x, y, theta)

        theta_fp = angle_to_fixed(theta)

        csd_digits = binary_to_csd(theta_fp, N_STAGES)

        x_csd, y_csd, _, _ = csd_ud_cordic(x, y, csd_digits)

        x_ref_list.append(x_ref)
        y_ref_list.append(y_ref)

        x_csd_list.append(x_csd)
        y_csd_list.append(y_csd)

        active_rotations.append(sum(1 for d in csd_digits if d != 0))


    x_ref_arr = np.array(x_ref_list)
    y_ref_arr = np.array(y_ref_list)

    x_csd_arr = np.array(x_csd_list)
    y_csd_arr = np.array(y_csd_list)

    metrics = compute_metrics(
        x_ref_arr,
        y_ref_arr,
        x_csd_arr,
        y_csd_arr
    )

    avg_active = np.mean(active_rotations)

    return metrics, avg_active


# ======================================================
# VISUALIZATION
# ======================================================

def plot_example():

    x = 0.8
    y = 0.2

    theta = np.deg2rad(30)

    x_ref, y_ref = float_rotation(x, y, theta)

    theta_fp = angle_to_fixed(theta)

    csd_digits = binary_to_csd(theta_fp, N_STAGES)

    x_csd, y_csd, x_path, y_path = csd_ud_cordic(x, y, csd_digits)

    plt.figure(figsize=(7,7))

    plt.arrow(0,0,x,y,width=0.01,label="Input")
    plt.arrow(0,0,x_ref,y_ref,width=0.01,label="Float Reference")
    plt.arrow(0,0,x_csd,y_csd,width=0.01,label="CSD-UD CORDIC")

    plt.plot(x_path,y_path,"o--",label="CORDIC Path")

    plt.grid(True)
    plt.axis("equal")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("RoPE CSD-UD CORDIC Rotation")

    plt.legend()

    plt.show()


# ======================================================
# MAIN
# ======================================================

def main():

    np.random.seed(0)

    metrics, avg_active = monte_carlo_test()

    print("\n===== CSD-UD-CORDIC FLOATING POINT EVALUATION =====\n")

    for k, v in metrics.items():

        print(f"{k:20s}: {v}")

    print("\nAverage Active Rotations :", avg_active, "/", N_STAGES)

    plot_example()


# ======================================================

if __name__ == "__main__":
    main()