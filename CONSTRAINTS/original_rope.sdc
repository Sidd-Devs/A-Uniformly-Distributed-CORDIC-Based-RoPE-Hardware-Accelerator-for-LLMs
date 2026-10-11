# --------------------------------------
# Timing constraint for combinational RoPE
# --------------------------------------

# Define max input-to-output delay
set_max_delay 2.0 -from [all_inputs] -to [all_outputs]

# Optional: model IO delays explicitly (not strictly required)
set_input_delay  0.2 [all_inputs]
set_output_delay 0.2 [all_outputs]
