import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================

N_STAGES = 10              # Number of UD-CORDIC stages
ANGLE_BITS = 16            # Angle precision
NUM_TESTS = 10000          # Monte Carlo samples


# ======================================================
# FLOATING POINT ROTATION (GROUND TRUTH)
# ======================================================

def float_rotation(x, y, theta):
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new


# ======================================================
# ANGLE TO BINARY UD REPRESENTATION
# ======================================================

def angle_to_binary_ud(theta, n_bits=ANGLE_BITS):

    bits = []
    remaining = theta

    for i in range(n_bits):

        alpha = 2.0 ** (-(i + 1))

        if remaining >= alpha:
            bits.append(1)
            remaining -= alpha
        else:
            bits.append(0)

    return bits


# ======================================================
# BINARY UD-CORDIC ROTATION
# ======================================================

def binary_ud_cordic(x, y, bits):

    xi, yi = x, y
    gain = 1.0

    x_path = [xi]
    y_path = [yi]

    for i, b in enumerate(bits[:N_STAGES]):

        if b == 1:

            shift = 2.0 ** (-(i + 1))

            gain *= np.sqrt(1 + shift * shift)

            x_next = xi - yi * shift
            y_next = yi + xi * shift

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
# MONTE CARLO TESTING
# ======================================================

def monte_carlo_test():

    x_ref_list = []
    y_ref_list = []

    x_ud_list = []
    y_ud_list = []

    for _ in range(NUM_TESTS):

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        theta = np.random.uniform(0, np.pi/2)

        x_ref, y_ref = float_rotation(x, y, theta)

        bits = angle_to_binary_ud(theta)

        x_ud, y_ud, _, _ = binary_ud_cordic(x, y, bits)

        x_ref_list.append(x_ref)
        y_ref_list.append(y_ref)

        x_ud_list.append(x_ud)
        y_ud_list.append(y_ud)

    x_ref_arr = np.array(x_ref_list)
    y_ref_arr = np.array(y_ref_list)

    x_ud_arr = np.array(x_ud_list)
    y_ud_arr = np.array(y_ud_list)

    metrics = compute_metrics(
        x_ref_arr, y_ref_arr,
        x_ud_arr, y_ud_arr
    )

    return metrics


# ======================================================
# VISUALIZATION FOR ONE SAMPLE
# ======================================================

def plot_example():

    x_in = 0.8
    y_in = 0.2

    theta = np.deg2rad(30)

    x_ref, y_ref = float_rotation(x_in, y_in, theta)

    bits = angle_to_binary_ud(theta)

    x_ud, y_ud, x_path, y_path = binary_ud_cordic(x_in, y_in, bits)

    plt.figure(figsize=(7,7))

    plt.arrow(0,0,x_in,y_in,width=0.01,color="blue",label="Input")

    plt.arrow(0,0,x_ref,y_ref,width=0.01,color="green",label="Float Reference")

    plt.arrow(0,0,x_ud,y_ud,width=0.01,color="red",label="Binary UD-CORDIC")

    plt.plot(x_path,y_path,"o--",alpha=0.7,label="CORDIC Path")

    plt.grid(True)
    plt.axis("equal")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("Binary UD-CORDIC Rotation")

    plt.legend()

    plt.show()


# ======================================================
# MAIN
# ======================================================

def main():

    np.random.seed(0)

    metrics = monte_carlo_test()

    print("\n===== BINARY UD-CORDIC FLOATING POINT EVALUATION =====\n")

    for k, v in metrics.items():
        print(f"{k:20s}: {v}")

    print("\nRunning visualization example...\n")

    plot_example()


# ======================================================

if __name__ == "__main__":
    main()