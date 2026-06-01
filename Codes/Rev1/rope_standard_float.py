import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================

N_STAGES = 10
NUM_TESTS = 10000

# ======================================================
# FLOAT REFERENCE ROTATION
# ======================================================

def float_rotation(x, y, theta):

    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)

    return x_new, y_new


# ======================================================
# STANDARD CORDIC ANGLE TABLE
# ======================================================

def cordic_angle_table(n):

    return [np.arctan(2**(-i)) for i in range(n)]


# ======================================================
# STANDARD CORDIC ROTATION MODE
# ======================================================

def standard_cordic(x, y, theta):

    xi, yi = x, y
    zi = theta

    x_path = [xi]
    y_path = [yi]

    angles = cordic_angle_table(N_STAGES)

    gain = 1.0

    for i in range(N_STAGES):

        di = 1 if zi >= 0 else -1

        shift = 2**(-i)

        x_next = xi - di * yi * shift
        y_next = yi + di * xi * shift
        z_next = zi - di * angles[i]

        xi, yi, zi = x_next, y_next, z_next

        gain *= np.sqrt(1 + shift*shift)

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

    x_cordic_list = []
    y_cordic_list = []

    for _ in range(NUM_TESTS):

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)

        theta = np.random.uniform(0, np.pi/2)

        x_ref, y_ref = float_rotation(x, y, theta)

        x_cordic, y_cordic, _, _ = standard_cordic(x, y, theta)

        x_ref_list.append(x_ref)
        y_ref_list.append(y_ref)

        x_cordic_list.append(x_cordic)
        y_cordic_list.append(y_cordic)

    x_ref_arr = np.array(x_ref_list)
    y_ref_arr = np.array(y_ref_list)

    x_cordic_arr = np.array(x_cordic_list)
    y_cordic_arr = np.array(y_cordic_list)

    metrics = compute_metrics(
        x_ref_arr,
        y_ref_arr,
        x_cordic_arr,
        y_cordic_arr
    )

    return metrics


# ======================================================
# VISUALIZATION
# ======================================================

def plot_example():

    x = 0.8
    y = 0.2

    theta = np.deg2rad(30)

    x_ref, y_ref = float_rotation(x, y, theta)

    x_cordic, y_cordic, x_path, y_path = standard_cordic(x, y, theta)

    plt.figure(figsize=(7,7))

    plt.arrow(0,0,x,y,width=0.01,label="Input")
    plt.arrow(0,0,x_ref,y_ref,width=0.01,label="Float Reference")
    plt.arrow(0,0,x_cordic,y_cordic,width=0.01,label="Standard CORDIC")

    plt.plot(x_path,y_path,"o--",label="CORDIC Path")

    plt.grid(True)
    plt.axis("equal")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.title("RoPE Standard CORDIC Rotation")

    plt.legend()

    plt.show()


# ======================================================
# MAIN
# ======================================================

def main():

    np.random.seed(0)

    metrics = monte_carlo_test()

    print("\n===== STANDARD CORDIC FLOATING POINT EVALUATION =====\n")

    for k, v in metrics.items():

        print(f"{k:20s}: {v}")

    plot_example()


# ======================================================

if __name__ == "__main__":
    main()