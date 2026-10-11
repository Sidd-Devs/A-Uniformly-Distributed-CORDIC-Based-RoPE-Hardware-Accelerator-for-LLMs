// ============================================================================
// Multiplier-Based RoPE Pair — Baseline for Comparison
// ============================================================================
// This module implements the RoPE rotation using direct fixed-point
// multiplications with pre-computed sin/cos values provided as inputs.
//
// Architecture:
//   Assumes sin(θ) and cos(θ) are pre-computed externally (e.g., by a
//   software driver or a separate LUT module) and provided as inputs.
//   The module performs the 2×2 rotation matrix multiplication:
//
//     y_even = cos(θ) * x_even − sin(θ) * x_odd
//     y_odd  = sin(θ) * x_even + cos(θ) * x_odd
//
//   This represents the most straightforward fixed-point RoPE implementation
//   requiring 4 multipliers and 2 adders, with no iterative computation.
//
// Parameters:
//   W — Total datapath word length (sign + integer + fractional bits)
// ============================================================================

module rope_mult_pair #(
    parameter W = 8
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_even,
    input  signed [W-1:0] x_odd,
    input  signed [W-1:0] sin_val,  // Pre-computed sin(θ) in Q1.(W-2)
    input  signed [W-1:0] cos_val,  // Pre-computed cos(θ) in Q1.(W-2)

    output reg signed [W-1:0] y_even,
    output reg signed [W-1:0] y_odd
);

    // =========================================================================
    // Rotation Matrix Multiplication
    // Four W×W signed multiplications producing 2W-bit products.
    // =========================================================================
    wire signed [2*W-1:0] prod_ce = cos_val * x_even;  // cos * x_even
    wire signed [2*W-1:0] prod_so = sin_val * x_odd;   // sin * x_odd
    wire signed [2*W-1:0] prod_se = sin_val * x_even;  // sin * x_even
    wire signed [2*W-1:0] prod_co = cos_val * x_odd;   // cos * x_odd

    // =========================================================================
    // Fixed-point truncation: Q1.F × Q1.F = Q2.2F → extract Q1.F
    // We take bits [2W-3 : W-2] to realign the fixed-point representation.
    // =========================================================================
    wire signed [W-1:0] trunc_ce = prod_ce[2*W-3:W-2];
    wire signed [W-1:0] trunc_so = prod_so[2*W-3:W-2];
    wire signed [W-1:0] trunc_se = prod_se[2*W-3:W-2];
    wire signed [W-1:0] trunc_co = prod_co[2*W-3:W-2];

    // =========================================================================
    // Pipeline register: single-cycle latency
    // =========================================================================
    always @(posedge clk) begin
        if (rst) begin
            y_even <= 0;
            y_odd  <= 0;
        end else begin
            y_even <= trunc_ce - trunc_so;  // cos*x_even - sin*x_odd
            y_odd  <= trunc_se + trunc_co;  // sin*x_even + cos*x_odd
        end
    end

endmodule
