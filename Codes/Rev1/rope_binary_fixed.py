import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================
FRAC_BITS = 7
SCALE     = 1 << FRAC_BITS

N_STAGES  = 10
ANGLE_BITS = N_STAGES 
NUM_TESTS = 10000

# ======================================================
# FLOAT REFERENCE ROTATION
# ======================================================
def float_rotation(x, y, theta):
    return (x * np.cos(theta) - y * np.sin(theta),
            x * np.sin(theta) + y * np.cos(theta))

# ======================================================
# FIXED-POINT CONVERSION
# ======================================================
def float_to_fixed(val):
    return int(np.round(val * SCALE))

def fixed_to_float(val):
    return val / SCALE

# ======================================================
# EXACT ANGLE DECOMPOSITION — Binary UD
# ======================================================
def angle_to_binary_ud(theta, n_bits=ANGLE_BITS):
    bits      = []
    remaining = theta

    for i in range(n_bits):
        # FIX: Using arctan matches the exact physical geometry of the 
        # hardware bit-shift, eliminating the hidden 2-degree error.
        alpha = np.arctan(2.0 ** (-(i + 1)))
        
        if remaining >= 0:
            bits.append(1)
            remaining -= alpha
        else:
            bits.append(0)
            remaining += alpha

    return bits

# ======================================================
# FIXED-POINT BINARY UD CORDIC (MULTIPLIERLESS)
# ======================================================
def binary_ud_cordic_fixed(x, y, bits):
    xi = x
    yi = y

    x_path = [xi]
    y_path = [yi]

    for i, b in enumerate(bits[:N_STAGES]):
        # Bidirectional rotation ensures a CONSTANT magnitude gain
        delta = 1 if b == 1 else -1 
        shift = i + 1                 

        x_next = xi - delta * (yi >> shift)
        y_next = yi + delta * (xi >> shift)

        xi, yi = x_next, y_next

        x_path.append(xi)
        y_path.append(yi)

    # NO MULTIPLIER: The hardware outputs are deliberately left unscaled. 
    # The constant magnitude gain is absorbed seamlessly by the LLM Softmax.
    
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

    # Magnitude error will be HIGH because the hardware is unscaled (Constant Gain)
    mag_error   = np.mean(np.abs(np.hypot(x_pred, y_pred) - np.hypot(x_ref, y_ref)))

    angle_ref   = np.arctan2(y_ref,  x_ref)
    angle_pred  = np.arctan2(y_pred, x_pred)
    
    # Wrap angles to handle -pi/pi boundary perfectly
    angle_diff = np.arctan2(np.sin(angle_pred - angle_ref), np.cos(angle_pred - angle_ref))
    angle_error = np.mean(np.abs(angle_diff))

    return {
        "MSE":             mse,
        "RMSE":            rmse,
        "MAE":             mae,
        "MAX_ERROR":       max_error,
        "MAG_ERROR":       mag_error,
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

        bits    = angle_to_binary_ud(theta)
        x_fixed = float_to_fixed(x)
        y_fixed = float_to_fixed(y)

        x_hw, y_hw, _, _ = binary_ud_cordic_fixed(x_fixed, y_fixed, bits)

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

    bits    = angle_to_binary_ud(theta)
    x_fixed = float_to_fixed(x)
    y_fixed = float_to_fixed(y)

    x_hw, y_hw, x_path, y_path = binary_ud_cordic_fixed(x_fixed, y_fixed, bits)

    x_hw   = fixed_to_float(x_hw)
    y_hw   = fixed_to_float(y_hw)
    x_path = [fixed_to_float(v) for v in x_path]
    y_path = [fixed_to_float(v) for v in y_path]

    plt.figure(figsize=(7, 7))
    plt.arrow(0, 0, x,     y,     width=0.005, color='gray', label="Input Vector")
    plt.arrow(0, 0, x_ref, y_ref, width=0.005, color='green', label="Target Rotation (Float)")
    plt.arrow(0, 0, x_hw,  y_hw,  width=0.005, color='red', label="Hardware Output (Unscaled)")
    
    plt.plot(x_path, y_path, "o--", color='orange', label="CORDIC Path")
    plt.grid(True);  plt.axis("equal")
    plt.xlabel("X");  plt.ylabel("Y")
    plt.title(f"Multiplierless Binary UD-CORDIC (Q1.{FRAC_BITS-1}, {N_STAGES} stages)")
    plt.legend();  plt.show()

# ======================================================
# MAIN
# ======================================================
def main():
    np.random.seed(0)
    metrics = monte_carlo_test()

    print(f"\n===== MULTIPLIERLESS BINARY UD-CORDIC (Q1.{FRAC_BITS-1}, {N_STAGES} stages) =====\n")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6e}" if isinstance(v, float) else f"{k:20s}: {v}")

    plot_example()

if __name__ == "__main__":
    main()