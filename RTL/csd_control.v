module csd_control #(
    parameter STAGES = 12
)(
    input  [2*STAGES-1:0] binary_bits,
    output [3*STAGES-1:0] delta_flat   // flattened 3-bit signed deltas
);

    genvar i;
    generate
        for (i = 0; i < STAGES; i = i + 1) begin : BOOTH_REC
            
            // Radix-4 Booth Encoding requires looking at the current 2 bits, 
            // plus the MSB of the previous 2 bits (b_minus_1).
            wire b_minus_1 = (i == 0) ? 1'b0 : binary_bits[2*i - 1];
            wire b_0       = binary_bits[2*i];
            wire b_1       = binary_bits[2*i + 1];

            // Formula: -2*b1 + b0 + b_minus_1
            // This guarantees an output of -2, -1, 0, 1, or 2.
            wire signed [2:0] delta_val = -2 * $signed({1'b0, b_1}) 
                                        + $signed({1'b0, b_0}) 
                                        + $signed({1'b0, b_minus_1});

            assign delta_flat[3*i +: 3] = delta_val;

        end
    endgenerate

endmodule
