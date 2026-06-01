import numpy as np
import math
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
NUM_ITERATIONS = 12     # The "Sweet Spot" for Edge AI (Balance of Area vs Accuracy)
FIXED_POINT_BITS = 32   # Hardware Register Width
TEST_ANGLE = 0.75       # Radians (approx 43 degrees)
INPUT_VECTOR = (1.0, 0.5)

# --- 1. PRE-CALCULATIONS ---
# Calculate CORDIC Gain (K) based on specific iteration count
# Hardware handles this via bit-shifts (CSD), here we simulate the math.
k_val = 1.0
for i in range(NUM_ITERATIONS):
    k_val *= math.sqrt(1 + 2**(-2 * i))
CORDIC_GAIN_INVERSE = 1.0 / k_val

# --- 2. GOLDEN REFERENCE (Standard Math) ---
def rope_float(x_in, y_in, theta_rad):
    x_out = x_in * np.cos(theta_rad) - y_in * np.sin(theta_rad)
    y_out = x_in * np.sin(theta_rad) + y_in * np.cos(theta_rad)
    return x_out, y_out

# --- 3. HARDWARE SIMULATOR (UD-CORDIC) ---
def float_to_ud_bits(theta_rad, iterations):
    """ Converts angle to UD-CORDIC control bits (0 or 1) """
    bits = []
    current_angle = 0.0
    for i in range(iterations):
        ud_weight = 2.0**(-i) # The UD Assumption
        if current_angle < theta_rad:
            bits.append(1) # Rotate Positive
            current_angle += ud_weight
        else:
            bits.append(0) # Rotate Negative
            current_angle -= ud_weight
    return bits

def rope_ud_cordic(x_in, y_in, theta_rad, iterations):
    # A. Trivial Rotator (Range Reduction)
    # Hardware Stage 0: Handles large angles to bring error into check
    current_x, current_y = x_in, y_in
    remaining_theta = theta_rad
    
    if abs(remaining_theta) > 0.4:
        direction = 1 if remaining_theta > 0 else -1
        # Hard rotation by ~45 degrees (Shift 0)
        x_temp = current_x - direction * current_y
        y_temp = current_y + direction * current_x
        current_x = x_temp
        current_y = y_temp
        remaining_theta -= direction * 0.785398 
    
    # B. Fixed Point Conversion (Simulating Registers)
    scale = 2**FIXED_POINT_BITS
    x_reg = int(current_x * scale)
    y_reg = int(current_y * scale)
    
    # C. Get Control Bits for the pipeline
    control_bits = float_to_ud_bits(remaining_theta, iterations)
    
    # D. The Pipeline Loop
    for i in range(iterations):
        direction = 1 if control_bits[i] == 1 else -1
        
        # Hard-wired Shift (No Multipliers!)
        x_shifted = x_reg >> i
        y_shifted = y_reg >> i
        
        # Add/Sub Logic
        x_next = x_reg - direction * y_shifted
        y_next = y_reg + direction * x_shifted
        
        x_reg = x_next
        y_reg = y_next

    # E. Gain Correction
    x_final = (x_reg * CORDIC_GAIN_INVERSE) / scale
    y_final = (y_reg * CORDIC_GAIN_INVERSE) / scale
    
    return x_final, y_final

# --- 4. VISUALIZATION & METRICS ---
def analyze_results():
    print(f"--- RoPE Accelerator Simulation (N={NUM_ITERATIONS}) ---")
    
    x_in, y_in = INPUT_VECTOR
    
    # Run Simulations
    x_gold, y_gold = rope_float(x_in, y_in, TEST_ANGLE)
    x_hw, y_hw = rope_ud_cordic(x_in, y_in, TEST_ANGLE, NUM_ITERATIONS)
    
    # --- ERROR ANALYSIS ---
    
    # 1. Magnitude (Length) Check
    mag_gold = math.sqrt(x_gold**2 + y_gold**2)
    mag_hw = math.sqrt(x_hw**2 + y_hw**2)
    mag_error = abs(mag_gold - mag_hw)
    
    # 2. Angle Check
    angle_gold = math.atan2(y_gold, x_gold)
    angle_hw = math.atan2(y_hw, x_hw)
    angle_error_rad = abs(angle_gold - angle_hw)
    angle_error_deg = math.degrees(angle_error_rad)

    # Output Stats
    print(f"Input Vector:    ({x_in}, {y_in})")
    print(f"Target Rotation: {TEST_ANGLE} rad ({math.degrees(TEST_ANGLE):.2f} deg)")
    print("-" * 40)
    print(f"Golden Output:   ({x_gold:.5f}, {y_gold:.5f})")
    print(f"Hardware Output: ({x_hw:.5f}, {y_hw:.5f})")
    print("-" * 40)
    print(f"Magnitude Error: {mag_error:.6f}")
    print(f"Angle Error:     {angle_error_rad:.6f} rad ({angle_error_deg:.4f} deg)")
    
    # --- VERDICT ---
    print("\n--- VERDICT FOR PROJECT ---")
    if angle_error_deg < 0.5:
        print("[PASS] Angle Error is < 0.5 degrees.")
        print("       This is EXCELLENT for LLMs (RoPE).")
        print("       Neural Networks are robust to this tiny noise.")
    else:
        print("[FAIL] Error is too high. Check Range Reduction logic.")

    # --- PLOTTING ---
    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, 'ko', label='Origin') 
    plt.arrow(0, 0, x_in, y_in, head_width=0.05, fc='blue', ec='blue', label='Input')
    plt.arrow(0, 0, x_gold, y_gold, head_width=0.05, fc='green', ec='green', linestyle='--', linewidth=2, label='Golden')
    plt.arrow(0, 0, x_hw*0.99, y_hw*0.99, head_width=0.03, fc='red', ec='red', label=f'UD-CORDIC (N={NUM_ITERATIONS})')
    
    limit = max(abs(x_in), abs(y_in), abs(x_gold)) + 0.5
    plt.xlim(-limit, limit); plt.ylim(-limit, limit)
    plt.grid(True); plt.legend()
    plt.title(f"RoPE Rotation (Iterations={NUM_ITERATIONS})")
    plt.show()

if __name__ == "__main__":
    analyze_results()