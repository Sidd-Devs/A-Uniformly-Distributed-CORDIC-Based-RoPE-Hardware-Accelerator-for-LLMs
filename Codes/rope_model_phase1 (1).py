import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# PART 1: Golden Reference
# ==========================================
def get_rope_angle(dim_index, total_dims, position_index):
    # theta = m * 10000^(-2d/dim)
    theta = position_index * (10000 ** (-2 * dim_index / total_dims))
    return theta

def standard_rotation(x, y, theta):
    # Standard precision rotation
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    return x_new, y_new

# ==========================================
# PART 2: Binary Helper
# ==========================================
def float_to_fixed_binary(angle, num_bits=16):
    # Converts angle (0 to pi/2) into binary bits for alpha = 2^-i
    bits = []
    curr_angle = angle
    for i in range(1, num_bits + 1):
        alpha = 2.0 ** (-i)
        if curr_angle >= alpha:
            bits.append(1)
            curr_angle -= alpha
        else:
            bits.append(0)
    return bits

# ==========================================
# PART 3: UD-CORDIC Rotation
# ==========================================
def ud_cordic_rotation(x, y, binary_bits):
    x_curr = x
    y_curr = y
    scale_factor = 1.0
    
    for i in range(len(binary_bits)):
        # i starts at 0, representing 2^-(i+1)
        power_val = -(i + 1)
        do_rotate = binary_bits[i]
        
        if do_rotate:
            shift_val = 2.0 ** power_val
            
            # Update Scaling Factor (K)
            scale_factor *= np.sqrt(1 + shift_val**2)
            
            # Rotation: x' = x - y*2^-i
            x_next = x_curr - (y_curr * shift_val)
            y_next = y_curr + (x_curr * shift_val)
            
            x_curr = x_next
            y_curr = y_next
            
    return x_curr / scale_factor, y_curr / scale_factor

# ==========================================
# PART 4: Main Verification
# ==========================================
def main():
    print("--- Starting Phase 1: UD-CORDIC Math Verification (FIXED) ---")
    
    dim = 64
    pos_idx = 5 
    x_vec = np.random.randn(dim)
    y_vec = np.random.randn(dim)
    errors = []
    
    print(f"Testing {dim} dimensions...")
    
    for d in range(dim):
        raw_theta = get_rope_angle(d, dim, pos_idx)
        
        # --- THE FIX IS HERE ---
        # We normalize the angle to [0, pi/2] for BOTH methods.
        # This isolates the test to verify the "Shift-Add" logic works.
        # (In hardware, a Quadrant Mapper handles the rest).
        theta_test = raw_theta % (np.pi/2) 
        
        # 1. Golden Reference using the TEST angle
        x_ref, y_ref = standard_rotation(x_vec[d], y_vec[d], theta_test)
        
        # 2. CORDIC using the TEST angle
        bits = float_to_fixed_binary(theta_test, num_bits=16)
        x_my, y_my = ud_cordic_rotation(x_vec[d], y_vec[d], bits)
        
        # 3. Calculate Error
        mse = (x_ref - x_my)**2 + (y_ref - y_my)**2
        errors.append(mse)

    avg_error = np.mean(errors)
    
    print(f"\n Verification Complete.")
    print(f"   Average Mean Squared Error (MSE): {avg_error:.8f}")
    
    # We expect error < 1e-4
    if avg_error < 1e-4:
        print("   STATUS: PASS (Algorithm is accurate)")
    else:
        print(f"   STATUS: FAIL (Error {avg_error} is still too high)")

    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o', linestyle='-', color='g', label='MSE per Dimension')
    plt.title(f"Corrected UD-CORDIC Error (Avg MSE: {avg_error:.2e})")
    plt.xlabel("Dimension Index (0-63)")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
