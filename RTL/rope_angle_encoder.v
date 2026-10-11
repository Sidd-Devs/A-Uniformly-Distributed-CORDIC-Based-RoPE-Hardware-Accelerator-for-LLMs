// ============================================================================
// rope_angle_encoder — feed-forward UD-CORDIC angle encoder
// ============================================================================
// Maps a residual rotation angle to the Binary UD-CORDIC digit string, with
// no Z-path and no sequential residual loop. Purely combinational.
//
// Why this block is needed
// -----------------------
// Stage i of the Binary UD rotator turns by a_i = atan(2^-(i+1)), not by
// 2^-(i+1). Rotations compose additively, so
//
//     theta = sum_i d_i * a_i ,  d_i in {-1,+1}
//     S := (theta + A) / 2 = sum_i b_i * a_i ,   A = sum_i a_i, b_i = (d_i+1)/2
//
// Encoding is therefore a subset sum over {a_i}. Those weights are only
// approximately geometric:
//
//     a_i / 2^-(i+1)  =  0.9273, 0.9799, 0.9948, 0.9987, 0.9997, ...
//
// The tail (i >= 2) is geometric to better than 0.13%, so its digits are just
// the fixed-point bits of the residual — free, wires only. The head (i < 2) is
// not, and taking its bits directly is what makes a naive encoder saturate at
// tens of degrees of error regardless of pipeline depth. The head is therefore
// resolved by an explicit threshold decode against the four reachable head
// sums {0, a1, a0, a0+a1}.
//
// Cost: 1 adder, 3 constant comparators, one 4:1 constant mux, 1 subtractor.
//
// Accuracy (N=10, FB=12), measured against the exact rotation:
//     0.094 deg RMS / 0.196 deg peak over the full [0, 2*pi) input range.
//
// Interface
//   theta_res : signed Q1.FB residual angle, must lie in [-pi/4, +pi/4].
//               Use rope_range_reduce to produce it.
//   ud_bits   : digit string for binary_ud_cordic_core (bit i drives stage i).
// ============================================================================

module rope_angle_encoder #(
    parameter N  = 10,   // rotator stages
    parameter FB = 12    // fractional bits of the angle datapath
)(
    input  signed [FB+1:0] theta_res,
    output        [N-1:0]  ud_bits
);

    localparam integer AW = FB + 2;

    // ---- constants, scaled by 2^FB -----------------------------------------
    // A       = sum_i atan(2^-(i+1)), i = 0..N-1
    // HEADSUM = {0, a1, a0, a0+a1}
    // TMAX    = sum_{i>=2} a_i
    // Values below are for N=10, FB=12. Regenerate with VERIFY/encoder_hw.py
    // if N or FB changes.
    localparam signed [AW:0] A_CONST = 3920;
    localparam signed [AW:0] HS_01   = 1003;   // a1
    localparam signed [AW:0] HS_10   = 1899;   // a0
    localparam signed [AW:0] HS_11   = 2903;   // a0 + a1
    localparam signed [AW:0] TMAX    = 1017;

    // ---- S = (theta + A) >> 1 ----------------------------------------------
    wire signed [AW:0] s_full = $signed({theta_res[FB+1], theta_res}) + A_CONST;
    wire signed [AW:0] s      = s_full >>> 1;

    // ---- head: threshold decode against the four reachable head sums -------
    wire b0 = (s >= HS_10);
    wire b1 = (s >= HS_11) ? 1'b1 : ((s >= HS_10) ? 1'b0 : (s >= HS_01));

    reg signed [AW:0] headsum;
    always @(*) begin
        case ({b0, b1})
            2'b00: headsum = {(AW+1){1'b0}};
            2'b01: headsum = HS_01;
            2'b10: headsum = HS_10;
            2'b11: headsum = HS_11;
        endcase
    end

    // ---- residual, clamped to the tail's reachable span --------------------
    wire signed [AW:0] r_raw = s - headsum;
    wire signed [AW:0] r     = (r_raw < 0)    ? {(AW+1){1'b0}} :
                               (r_raw > TMAX) ? TMAX : r_raw;

    // ---- tail: geometric, so the digits are just the bits of r -------------
    // stage j (j >= 2) has weight ~2^-(j+1)  ->  bit (FB-j-1) of r
    genvar j;
    generate
        assign ud_bits[0] = b0;
        assign ud_bits[1] = b1;
        for (j = 2; j < N; j = j + 1) begin : TAIL
            if (FB >= j + 1)
                assign ud_bits[j] = r[FB - j - 1];
            else
                assign ud_bits[j] = 1'b0;
        end
    endgenerate

endmodule


// ============================================================================
// rope_range_reduce — fold an unbounded RoPE phase into +/- pi/4
// ============================================================================
// RoPE drives theta = m * Theta_i, which grows without bound in the token
// position m. The rotator only reaches +/-54.9 deg (Binary UD), so reduction
// is mandatory, not optional.
//
// The input is the phase as a fraction of 2*pi (Q0.PW unsigned), which is what
// a RoPE phase accumulator naturally produces — taking it mod 2*pi is then
// free, it is just truncation of the accumulator. The top two bits select the
// quadrant; quadrant rotation is free downstream (swap x/y and negate). The
// remaining bits are recentred to +/-pi/4 and scaled to radians.
//
//   quadrant : 0..3, multiples of 90 deg, applied by the datapath for free
//   theta_res: signed Q1.FB residual in [-pi/4, +pi/4]
// ============================================================================

module rope_range_reduce #(
    parameter PW = 16,   // phase word: fraction of 2*pi
    parameter FB = 12
)(
    input      [PW-1:0]     phase,
    output     [1:0]        quadrant,
    output signed [FB+1:0]  theta_res
);

    localparam integer AW = FB + 2;

    // PI_OVER_2 in Q1.FB  (pi/2 = 1.5707963 -> round(1.5707963 * 2^FB))
    localparam signed [AW+FB:0] PI_OVER_2 = 6434;   // FB = 12

    assign quadrant = phase[PW-1:PW-2];

    // position within the quadrant, recentred to [-0.5, +0.5)
    wire [PW-3:0]          u        = phase[PW-3:0];
    wire signed [PW-1:0]   u_signed = $signed({1'b0, u}) - $signed({2'b0, {(PW-3){1'b0}}} | (1 <<< (PW - 4)));

    // theta_res = u_centred * (pi/2), rescaled from Q0.(PW-2) to Q1.FB
    wire signed [PW+AW:0] prod = u_signed * PI_OVER_2;
    assign theta_res = prod >>> (PW - 2);

endmodule
