import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================

FRAC_BITS   = 7
SCALE       = 1 << FRAC_BITS          # Q1.F data scale

ANGLE_FRAC_BITS = 5
ANGLE_SCALE     = 1 << ANGLE_FRAC_BITS  # fixed-point angle scale

N_STAGES  = 5
NUM_TESTS = 10000


# ======================================================
# FLOAT REFERENCE ROTATION
# ======================================================

def float_rotation(x, y, theta):
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new


# ======================================================
# FIXED-POINT CONVERSION
# ======================================================

def float_to_fixed(val):
    return int(np.round(val * SCALE))

def fixed_to_float(val):
    return val / SCALE

def angle_float_to_fixed(theta):
    return int(np.round(theta * ANGLE_SCALE))

def angle_fixed_to_float(val):
    return val / ANGLE_SCALE


# ======================================================
# CORDIC ANGLE LUT  (fixed-point, same scale as Z-path)
# ======================================================

def cordic_angle_table_fixed():
    """
    Standard CORDIC micro-rotation angles: arctan(2^-i) for i = 0 … N-1.
    Stored in the same Q-format as the Z accumulator.
    """
    return [int(np.round(np.arctan(2**(-i)) * ANGLE_SCALE))
            for i in range(N_STAGES)]


# ======================================================
# STANDARD CORDIC FIXED-POINT CORE (MULTIPLIERLESS)
# ======================================================

def standard_cordic_fixed(x, y, theta_fixed):
    """
    All inputs are fixed-point integers (Q1.F for x,y; ANGLE_SCALE for theta).
    Returns fixed-point (Q1.F) outputs.
    """
    xi = x
    yi = y
    zi = theta_fixed          # Z-path accumulator, same scale as angle LUT

    x_path = [xi]
    y_path = [yi]

    angles = cordic_angle_table_fixed()

    for i in range(N_STAGES):
        di     = 1 if zi >= 0 else -1
        shift  = i                        # 2^(-i) shift — correct for standard CORDIC

        x_next = xi - di * (yi >> shift)
        y_next = yi + di * (xi >> shift)
        z_next = zi - di * angles[i]

        xi, yi, zi = x_next, y_next, z_next

        x_path.append(xi)
        y_path.append(yi)

    # REMOVED MULTIPLIER: Hardware outputs are unscaled to match the Verilog RTL.
    # The standard CORDIC magnitude gain (~1.647) is preserved here.

    return xi, yi, x_path, y_path


# ======================================================
# ERROR METRICS
# ======================================================

def compute_metrics(x_ref, y_ref, x_pred, y_pred):
    diff        = np.stack([x_ref - x_pred, y_ref - y_pred], axis=1)
    mse         = np.mean(diff**2)
    rmse        = np.sqrt(mse)
    mae         = np.mean(np.abs(diff))
    max_error   = np.max(np.abs(diff))

    mag_ref     = np.hypot(x_ref,  y_ref)
    mag_pred    = np.hypot(x_pred, y_pred)
    mag_error   = np.mean(np.abs(mag_pred - mag_ref))

    angle_ref   = np.arctan2(y_ref,  x_ref)
    angle_pred  = np.arctan2(y_pred, x_pred)
    
    # FIX: Safely wraps angles across the -pi/pi boundary
    angle_diff = np.arctan2(np.sin(angle_pred - angle_ref), np.cos(angle_pred - angle_ref))
    angle_error = np.mean(np.abs(angle_diff))

    return {
        "MSE":             mse,
        "RMSE":            rmse,
        "MAE":             mae,
        "MAX_ERROR":       max_error,
        "MAG_ERROR":       mag_error, # Will be high due to ~1.647 constant gain
        "ANGLE_ERROR_RAD": angle_error,
        "ANGLE_ERROR_DEG": np.rad2deg(angle_error),
    }


# ======================================================
# MONTE CARLO TEST
# ======================================================

def monte_carlo_test():
    x_ref_list, y_ref_list = [], []
    x_hw_list,  y_hw_list  = [], []

    for _ in range(NUM_TESTS):
        x     = np.random.uniform(-1, 1)
        y     = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 0.78)   

        x_ref, y_ref = float_rotation(x, y, theta)

        x_fixed     = float_to_fixed(x)
        y_fixed     = float_to_fixed(y)
        theta_fixed = angle_float_to_fixed(theta)

        x_hw, y_hw, _, _ = standard_cordic_fixed(x_fixed, y_fixed, theta_fixed)

        x_hw = fixed_to_float(x_hw)
        y_hw = fixed_to_float(y_hw)

        x_ref_list.append(x_ref);  y_ref_list.append(y_ref)
        x_hw_list.append(x_hw);    y_hw_list.append(y_hw)

    return compute_metrics(
        np.array(x_ref_list), np.array(y_ref_list),
        np.array(x_hw_list),  np.array(y_hw_list),
    )


# ======================================================
# VISUALIZATION
# ======================================================

def plot_example():
    x, y  = 0.8, 0.2
    theta = np.deg2rad(30)

    x_ref, y_ref = float_rotation(x, y, theta)

    x_fixed     = float_to_fixed(x)
    y_fixed     = float_to_fixed(y)
    theta_fixed = angle_float_to_fixed(theta)

    x_hw, y_hw, x_path, y_path = standard_cordic_fixed(x_fixed, y_fixed, theta_fixed)

    x_hw    = fixed_to_float(x_hw)
    y_hw    = fixed_to_float(y_hw)
    x_path  = [fixed_to_float(v) for v in x_path]
    y_path  = [fixed_to_float(v) for v in y_path]

    plt.figure(figsize=(7, 7))
    plt.arrow(0, 0, x,     y,     width=0.005, color='gray', label="Input Vector")
    plt.arrow(0, 0, x_ref, y_ref, width=0.005, color='green', label="Target Rotation (Float)")
    plt.arrow(0, 0, x_hw,  y_hw,  width=0.005, color='red', label="Standard CORDIC (Unscaled)")
    plt.plot(x_path, y_path, "o--", color='orange', label="CORDIC Path")
    plt.grid(True);  plt.axis("equal")
    plt.xlabel("X");  plt.ylabel("Y")
    plt.title(f"Multiplierless Standard CORDIC (Q1.{FRAC_BITS-1}, {N_STAGES} stages)")
    plt.legend();  plt.show()


# ======================================================
# MAIN
# ======================================================

def main():
    np.random.seed(0)
    metrics = monte_carlo_test()

    print(f"\n===== MULTIPLIERLESS STANDARD CORDIC FIXED-POINT (Q1.{FRAC_BITS-1}, {N_STAGES} stages) =====\n")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}" if isinstance(v, float) else f"{k:20s}: {v}")

    plot_example()


if __name__ == "__main__":
    main()