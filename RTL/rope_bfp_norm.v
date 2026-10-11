// ============================================================================
// rope_bfp_norm / rope_bfp_denorm — block floating-point scaling for RoPE
// ============================================================================
// Why this block is mandatory, not an optimisation
// ------------------------------------------------
// The rotator datapath is Q(1,F), which represents [-2, +2). Measured query
// and key activations in real LLMs reach |q| = 132 and |k| = 178 (Qwen2-7B,
// median ~20). A bare fixed-point rotator therefore saturates essentially
// every input, and the published perplexity figures for such a datapath
// cannot have modelled it.
//
// The fix is block floating point: one shared exponent per block of pairs,
// applied as a barrel shift in and out. The exponent travels alongside the
// mantissas and is removed after the rotation, so the rotator itself stays
// purely fixed point and unchanged.
//
// Why a SHARED exponent is the right choice here
// ----------------------------------------------
// The d_h/2 pairs of one head at one token position share a magnitude scale
// closely enough that a single exponent serves them all. Per-element
// exponents (i.e. an FP8/FP4-style element format) would add an exponent
// field, alignment and normalisation to every lane to buy range the block
// does not need. D is left parameterised so the granularity/accuracy trade
// can be swept.
//
// Headroom
// --------
// The block is scaled so the largest magnitude lands below 2^(W-4) rather
// than 2^(W-3). The extra bit covers two things at once: the CORDIC gain
// K = 1.164 applied through the pipeline, and the fact that the magnitude
// estimate below uses a one's-complement approximation (|v|-1 for negative
// v) to avoid paying a negation adder on all 2D lanes.
//
// Interface
//   vec_in / vec_out : 2*D values of W bits, lane i at [i*W +: W].
//                      Pairs are (lane 2i, lane 2i+1).
//   exp_out          : signed left-shift applied; feed to rope_bfp_denorm.
// ============================================================================

module rope_bfp_norm #(
    parameter W  = 12,      // datapath word length, Q(1, W-2)
    parameter D  = 64,      // pairs per block (d_h/2 for one head)
    parameter EW = 6        // exponent width, must cover +-W
)(
    input      [2*D*W-1:0]  vec_in,
    output     [2*D*W-1:0]  vec_out,
    output signed [EW-1:0]  exp_out
);

    localparam integer LANES  = 2 * D;
    localparam integer TARGET = W - 4;   // highest magnitude bit lands here

    // ---- magnitude estimate, one's complement (no adder per lane) ---------
    // For v >= 0 this is v; for v < 0 it is |v| - 1. Never over-estimates,
    // and the TARGET above carries the extra bit that makes that safe.
    wire [W-2:0] mag [0:LANES-1];
    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : MAG
            wire [W-1:0] v = vec_in[l*W +: W];
            assign mag[l] = v[W-1] ? ~v[W-2:0] : v[W-2:0];
        end
    endgenerate

    // ---- OR-reduce the block: one pass, no comparators --------------------
    integer j;
    reg [W-2:0] agg;
    always @(*) begin
        agg = {(W-1){1'b0}};
        for (j = 0; j < LANES; j = j + 1)
            agg = agg | mag[j];
    end

    // A one's-complement magnitude is |v|-1, so an exact power of two reports
    // its MSB one position low (|v| = 2^k gives 2^k - 1). Incrementing the
    // AGGREGATE restores it: the OR is >= 2^k - 1, so +1 carries into bit k.
    // One adder for the whole block, versus a negation on all 2D lanes.
    wire [W-1:0] agg_p1 = {1'b0, agg} + 1'b1;

    // ---- position of the highest set bit ----------------------------------
    integer k;
    reg [EW-1:0] msb;
    reg          any;
    always @(*) begin
        msb = {EW{1'b0}};
        any = 1'b0;
        for (k = 0; k < W; k = k + 1)
            if (agg_p1[k]) begin
                msb = k[EW-1:0];
                any = 1'b1;
            end
    end

    // ---- shift amount: negative means shift right -------------------------
    wire signed [EW-1:0] sh = any ? (TARGET[EW-1:0] - $signed(msb))
                                  : {EW{1'b0}};
    assign exp_out = sh;

    // ---- barrel shift every lane by the shared amount ---------------------
    generate
        for (l = 0; l < LANES; l = l + 1) begin : SHIFT
            wire signed [W-1:0] v = vec_in[l*W +: W];
            assign vec_out[l*W +: W] = (sh >= 0) ? (v <<<  sh)
                                                 : (v >>> (-sh));
        end
    endgenerate

endmodule


// ============================================================================
// rope_bfp_denorm — undo the block scaling after the rotation
// ============================================================================
// Rotation is linear, so R(theta)(alpha x) = alpha R(theta) x and the scaling
// commutes with the rotator exactly. No accuracy is lost by the round trip
// beyond the mantissa truncation the rotator applies anyway.
// ============================================================================

module rope_bfp_denorm #(
    parameter W  = 12,
    parameter D  = 64,
    parameter EW = 6
)(
    input      [2*D*W-1:0]  vec_in,
    input  signed [EW-1:0]  exp_in,
    output     [2*D*W-1:0]  vec_out
);
    localparam integer LANES = 2 * D;
    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : UNSHIFT
            wire signed [W-1:0] v = vec_in[l*W +: W];
            assign vec_out[l*W +: W] = (exp_in >= 0) ? (v >>>  exp_in)
                                                     : (v <<< (-exp_in));
        end
    endgenerate
endmodule
