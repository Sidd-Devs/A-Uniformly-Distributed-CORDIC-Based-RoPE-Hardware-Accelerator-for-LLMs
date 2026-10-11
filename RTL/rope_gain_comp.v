// ============================================================================
// rope_gain_comp — angle-dependent gain compensation for CSD UD-CORDIC
// ============================================================================
// Why this exists
// ---------------
// A CSD UD-CORDIC stage applies x' = x - t*y, y' = y + t*x with t = v*2^-2i,
// so its gain is sqrt(1 + v^2 * 4^-i) -- a function of the DIGIT MAGNITUDE.
// Any digit set with more than one magnitude therefore has a gain that depends
// on the angle, and a fixed-magnitude set is one bit per stage, which is just
// Binary UD. Constant gain and radix-4 stage merging are mutually exclusive.
//
// Measured consequence: K ranges over [1.00, 1.93], and left uncompensated the
// perplexity penalty is +908% (LLaMA-2-7B) / +76% (Qwen2-7B) on WikiText-2.
//
// What this block does
// --------------------
// The spread is dominated by the head stages -- stage 0 contributes ~80% of it
// and stage 1 another ~7%, while every later stage is within 0.5%. Compensating
// the head alone therefore removes almost all of it:
//
//     compensated stages   residual gain spread   WikiText-2 penalty
//        0 (none)                93.4%            +908% / +76%
//        1 (H1, this block)       7.3%            +0.62% / +0.77%
//        2 (H2)                   0.47%           +0.005% / +0.008%
//
// Crucially the head gain takes only THREE distinct values, so compensation is
// a select between three CONSTANTS -- shift-add networks, not a multiplier:
//
//     digit 0     K = 1.000000   1/K = 1.000000   passthrough
//     digit +-1   K = 1.414214   1/K = 0.707107   x>>1 + x>>2 - x>>5 - x>>7 - x>>8
//     digit +-1.5 K = 1.802776   1/K = 0.554700   x>>1 + x>>5 + x>>6 + x>>7
//
// Approximation error is -7.6e-5 and -1.3e-5 respectively, far below the
// datapath LSB at any word length used here.
//
// Cost: 7 adders per datapath, 2 datapaths, plus a 3:1 mux. Compare with a
// W=9 multiplier at roughly 574 um2 (extrapolated from rope_mult_pair).
//
// MODE = 1 selects H1 (stage 0 only). MODE = 2 selects H2 (stages 0 and 1),
// which needs nine constants and is provided for the accuracy/area sweep.
// ============================================================================

module rope_gain_comp #(
    parameter W    = 12,
    parameter MODE = 1          // 1 = H1 (3 constants), 2 = H2 (9 constants)
)(
    input  signed [W-1:0] x_in,
    input  signed [W-1:0] y_in,
    input  signed [2:0]   delta0,   // stage-0 digit, csd_ud_stage encoding
    input  signed [2:0]   delta1,   // stage-1 digit, used only when MODE = 2
    output signed [W-1:0] x_out,
    output signed [W-1:0] y_out
);

    // ---- constant scalers, realised as shift-add ---------------------------
    // Each term is accumulated at full width and rounded ONCE at the end.
    // Shifting right per-term instead would truncate five times over and the
    // accumulated error swamps the constant's own approximation error
    // (measured: 3.1e-3 that way, versus 7.6e-5 here).
    //
    //   1/sqrt(2)    = 181/256   (err -7.6e-5)
    //   1/sqrt(3.25) = 142/256   (err -1.3e-5)
    localparam integer G = 8;                   // guard bits, denominator 2^G

    function signed [W+G:0] mul181;             // v * 181
        input signed [W-1:0] v;
        begin
            mul181 = ($signed(v) <<< 7) + ($signed(v) <<< 5)
                   + ($signed(v) <<< 4) + ($signed(v) <<< 2) + $signed(v);
        end
    endfunction

    function signed [W+G:0] mul142;             // v * 142
        input signed [W-1:0] v;
        begin
            mul142 = ($signed(v) <<< 7) + ($signed(v) <<< 3)
                   + ($signed(v) <<< 2) + ($signed(v) <<< 1);
        end
    endfunction

    function signed [W-1:0] rnd;                // round-to-nearest, then narrow
        input signed [W+G:0] acc;
        begin
            rnd = (acc + (1 <<< (G - 1))) >>> G;
        end
    endfunction

    function signed [W-1:0] scale_r2;
        input signed [W-1:0] v;
        begin scale_r2 = rnd(mul181(v)); end
    endfunction

    function signed [W-1:0] scale_r325;
        input signed [W-1:0] v;
        begin scale_r325 = rnd(mul142(v)); end
    endfunction

    // ---- stage-0 magnitude selects the scaler ------------------------------
    // csd_ud_stage encoding: 010 = +1.5, 001 = +1, 000 = 0, 111 = -1, 110 = -1.5
    wire is_zero = (delta0 == 3'sb000);
    wire is_one  = (delta0 == 3'sb001) || (delta0 == 3'sb111);

    reg signed [W-1:0] x_h1, y_h1;
    always @(*) begin
        if (is_zero) begin
            x_h1 = x_in;              y_h1 = y_in;
        end else if (is_one) begin
            x_h1 = scale_r2(x_in);    y_h1 = scale_r2(y_in);
        end else begin
            x_h1 = scale_r325(x_in);  y_h1 = scale_r325(y_in);
        end
    end

    generate
        if (MODE == 1) begin : H1
            assign x_out = x_h1;
            assign y_out = y_h1;
        end else begin : H2
            // Stage 1 has weight 2^-2, so its gains are sqrt(1 + v^2/16):
            //   v = 0     -> 1.000000  passthrough
            //   v = +-1   -> 1.030776  1/K = 0.970143 ~ 497/512 (err +5.6e-4)
            //   v = +-1.5 -> 1.068000  1/K = 0.936330 ~ 479/512 (err -7.8e-4)
            // 497 = 512 - 16 + 1, 479 = 512 - 32 - 1: two adders each.
            wire z1 = (delta1 == 3'sb000);
            wire o1 = (delta1 == 3'sb001) || (delta1 == 3'sb111);
            reg signed [W+9:0] acc2;
            always @(*) begin
                if (z1)      acc2 = $signed(x_h1) <<< 9;
                else if (o1) acc2 = ($signed(x_h1) <<< 9) - ($signed(x_h1) <<< 4)
                                    + $signed(x_h1);
                else         acc2 = ($signed(x_h1) <<< 9) - ($signed(x_h1) <<< 5)
                                    - $signed(x_h1);
            end
            reg signed [W+9:0] acc2y;
            always @(*) begin
                if (z1)      acc2y = $signed(y_h1) <<< 9;
                else if (o1) acc2y = ($signed(y_h1) <<< 9) - ($signed(y_h1) <<< 4)
                                     + $signed(y_h1);
                else         acc2y = ($signed(y_h1) <<< 9) - ($signed(y_h1) <<< 5)
                                     - $signed(y_h1);
            end
            wire signed [W-1:0] x_h2 = (acc2  + (1 <<< 8)) >>> 9;
            wire signed [W-1:0] y_h2 = (acc2y + (1 <<< 8)) >>> 9;
            assign x_out = x_h2;
            assign y_out = y_h2;
        end
    endgenerate

endmodule
