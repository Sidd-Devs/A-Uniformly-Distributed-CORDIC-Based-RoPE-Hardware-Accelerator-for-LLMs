import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# PARAMETERS
# ======================================================
FRAC_BITS = 7
SCALE = 1 << FRAC_BITS

N_STAGES = 5
NUM_TESTS = 10000

# ======================================================
# FLOAT REFERENCE ROTATION
# ======================================================
def float_rotation(x, y, theta):
    return (
        x*np.cos(theta) - y*np.sin(theta),
        x*np.sin(theta) + y*np.cos(theta)
    )

# ======================================================
# FIXED CONVERSION
# ======================================================
def float_to_fixed(v):
    return int(np.round(v * SCALE))

def fixed_to_float(v):
    return v / SCALE

# ======================================================
# CORRECT CSD-UD ANGLE DECOMPOSITION
# ======================================================
def angle_to_csd_ud(theta):
    digits = []
    rem = theta
    for i in range(N_STAGES):
        weight = 2.0 ** (-(i+1))
        if rem > weight/2:
            digits.append(+1)
            rem -= weight
        elif rem < -weight/2:
            digits.append(-1)
            rem += weight
        else:
            digits.append(0)
    return digits

# ======================================================
# EXACT HARDWARE CSD-UD CORDIC (NO FLOATING POINT CHEATS)
# ======================================================
def csd_ud_cordic_fixed(x, y, digits):
    xi = x
    yi = y
    x_path = [xi]
    y_path = [yi]

    for i, d in enumerate(digits):
        shift = i + 1
        if d == 1:
            # Positive rotation
            x_next = xi - (yi >> shift)
            y_next = yi + (xi >> shift)
            xi, yi = x_next, y_next
        elif d == -1:
            # Negative rotation
            x_next = xi + (yi >> shift)
            y_next = yi - (xi >> shift)
            xi, yi = x_next, y_next
            
        x_path.append(xi)
        y_path.append(yi)

    return xi, yi, x_path, y_path

# ======================================================
# ERROR METRICS
# ======================================================
def compute_metrics(x_ref,y_ref,x_pred,y_pred):
    vec_ref = np.stack([x_ref,y_ref],axis=1)
    vec_pred = np.stack([x_pred,y_pred],axis=1)
    diff = vec_ref - vec_pred

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))

    mag_ref = np.sqrt(x_ref**2 + y_ref**2)
    mag_pred = np.sqrt(x_pred**2 + y_pred**2)
    mag_error = np.mean(np.abs(mag_pred-mag_ref))

    angle_ref = np.arctan2(y_ref,x_ref)
    angle_pred = np.arctan2(y_pred,x_pred)
    
    # Wrap angles to handle -pi/pi boundary
    angle_diff = np.arctan2(np.sin(angle_pred - angle_ref), np.cos(angle_pred - angle_ref))
    angle_error = np.mean(np.abs(angle_diff))

    return {
        "MSE":mse,
        "RMSE":rmse,
        "MAE":mae,
        "MAX_ERROR":max_error,
        "MAG_ERROR":mag_error,
        "ANGLE_ERROR_RAD":angle_error,
        "ANGLE_ERROR_DEG":np.rad2deg(angle_error)
    }

# ======================================================
# MONTE CARLO TEST
# ======================================================
def monte_carlo_test():
    x_ref_list, y_ref_list = [], []
    x_hw_list, y_hw_list = [], []
    active_rot = []

    for _ in range(NUM_TESTS):  
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)
        theta = np.random.uniform(0, 0.78)

        x_ref, y_ref = float_rotation(x, y, theta)
        digits = angle_to_csd_ud(theta)

        x_fixed = float_to_fixed(x)
        y_fixed = float_to_fixed(y)

        x_hw, y_hw, _, _ = csd_ud_cordic_fixed(x_fixed, y_fixed, digits)

        x_ref_list.append(x_ref)
        y_ref_list.append(y_ref)
        x_hw_list.append(fixed_to_float(x_hw))
        y_hw_list.append(fixed_to_float(y_hw))

        active_rot.append(sum(1 for d in digits if d!=0))

    metrics = compute_metrics(
        np.array(x_ref_list), np.array(y_ref_list),
        np.array(x_hw_list), np.array(y_hw_list)
    )
    avg_active = np.mean(active_rot)
    return metrics, avg_active

# ======================================================
# VISUALIZATION
# ======================================================
def plot_example():
    x, y = 0.8, 0.2
    theta = np.deg2rad(30)

    x_ref, y_ref = float_rotation(x, y, theta)
    digits = angle_to_csd_ud(theta)

    x_fixed, y_fixed = float_to_fixed(x), float_to_fixed(y)
    x_hw, y_hw, x_path, y_path = csd_ud_cordic_fixed(x_fixed, y_fixed, digits)

    x_hw, y_hw = fixed_to_float(x_hw), fixed_to_float(y_hw)
    x_path = [fixed_to_float(v) for v in x_path]
    y_path = [fixed_to_float(v) for v in y_path]

    plt.figure(figsize=(7,7))
    plt.arrow(0, 0, x, y, width=0.005, color='gray', label="Input Vector")
    plt.arrow(0, 0, x_ref, y_ref, width=0.005, color='green', label="Target Rotation (Float)")
    plt.arrow(0, 0, x_hw, y_hw, width=0.005, color='red', label="Raw Hardware Output")

    plt.plot(x_path, y_path, "o--", color='orange', label="CORDIC Shifts")
    plt.grid(True)
    plt.axis("equal")
    plt.title("Hardware-Accurate CSD-UD CORDIC")
    plt.legend()
    plt.show()

if __name__=="__main__":
    np.random.seed(0)
    metrics, avg_active = monte_carlo_test()
    print("\n===== HARDWARE-ACCURATE CSD-UD CORDIC FIXED POINT =====\n")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")
    print(f"\nAverage Active Rotations: {avg_active:.2f} / {N_STAGES}")
    plot_example()