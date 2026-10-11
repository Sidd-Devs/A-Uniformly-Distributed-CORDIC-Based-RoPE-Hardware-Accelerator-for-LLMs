// ============================================================================
// LUT-Based RoPE Pair — Baseline for Comparison
// ============================================================================
// This module implements the RoPE rotation using a pre-computed sin/cos
// Look-Up Table (ROM). It serves as a non-CORDIC baseline for the paper.
//
// Architecture:
//   1. A combinational ROM stores sin(θ) values for θ in [0, π/2),
//      quantized to 128 entries (ADDR_W=7). Base values are Q1.14.
//   2. cos(θ) is derived using the identity cos(θ) = sin(π/2 − θ).
//   3. Quadrant logic maps any angle in [0, 2π) to the correct sin/cos.
//   4. Four fixed-point multipliers compute the 2×2 rotation matrix product.
//   5. Two adders combine the products to produce the rotated output pair.
//
// Parameters:
//   W       — Total datapath word length (sign + integer + fractional bits)
//   ADDR_W  — Address width for the sin/cos ROM (ROM depth = 2^ADDR_W)
// ============================================================================

module rope_lut_pair #(
    parameter W      = 8,
    parameter ADDR_W = 7    // 128-entry quarter-wave LUT
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_even,
    input  signed [W-1:0] x_odd,
    input         [ADDR_W+1:0] angle_index, // Full angle index (quadrant + LUT addr)

    output reg signed [W-1:0] y_even,
    output reg signed [W-1:0] y_odd
);

    // =========================================================================
    // Quarter-wave sin ROM function (combinational)
    // Base values are pre-computed as round(sin(i*π/(2*128)) * 2^14) in Q1.14.
    // Dynamically scaled to the target word length W at elaboration time,
    // using the same approach as the standard CORDIC ATAN LUT.
    // =========================================================================
    function signed [W-1:0] get_sin;
        input integer idx;
        integer sin_base;
        begin
            case (idx)
                0: sin_base = 0;
                1: sin_base = 201;
                2: sin_base = 402;
                3: sin_base = 603;
                4: sin_base = 804;
                5: sin_base = 1005;
                6: sin_base = 1205;
                7: sin_base = 1406;
                8: sin_base = 1606;
                9: sin_base = 1806;
                10: sin_base = 2006;
                11: sin_base = 2205;
                12: sin_base = 2404;
                13: sin_base = 2603;
                14: sin_base = 2801;
                15: sin_base = 2999;
                16: sin_base = 3196;
                17: sin_base = 3393;
                18: sin_base = 3590;
                19: sin_base = 3786;
                20: sin_base = 3981;
                21: sin_base = 4176;
                22: sin_base = 4370;
                23: sin_base = 4563;
                24: sin_base = 4756;
                25: sin_base = 4948;
                26: sin_base = 5139;
                27: sin_base = 5330;
                28: sin_base = 5520;
                29: sin_base = 5708;
                30: sin_base = 5897;
                31: sin_base = 6084;
                32: sin_base = 6270;
                33: sin_base = 6455;
                34: sin_base = 6639;
                35: sin_base = 6823;
                36: sin_base = 7005;
                37: sin_base = 7186;
                38: sin_base = 7366;
                39: sin_base = 7545;
                40: sin_base = 7723;
                41: sin_base = 7900;
                42: sin_base = 8076;
                43: sin_base = 8250;
                44: sin_base = 8423;
                45: sin_base = 8595;
                46: sin_base = 8765;
                47: sin_base = 8935;
                48: sin_base = 9102;
                49: sin_base = 9269;
                50: sin_base = 9434;
                51: sin_base = 9598;
                52: sin_base = 9760;
                53: sin_base = 9921;
                54: sin_base = 10080;
                55: sin_base = 10238;
                56: sin_base = 10394;
                57: sin_base = 10549;
                58: sin_base = 10702;
                59: sin_base = 10853;
                60: sin_base = 11003;
                61: sin_base = 11151;
                62: sin_base = 11297;
                63: sin_base = 11442;
                64: sin_base = 11585;
                65: sin_base = 11727;
                66: sin_base = 11866;
                67: sin_base = 12004;
                68: sin_base = 12140;
                69: sin_base = 12274;
                70: sin_base = 12406;
                71: sin_base = 12537;
                72: sin_base = 12665;
                73: sin_base = 12792;
                74: sin_base = 12916;
                75: sin_base = 13039;
                76: sin_base = 13160;
                77: sin_base = 13279;
                78: sin_base = 13395;
                79: sin_base = 13510;
                80: sin_base = 13623;
                81: sin_base = 13733;
                82: sin_base = 13842;
                83: sin_base = 13949;
                84: sin_base = 14053;
                85: sin_base = 14155;
                86: sin_base = 14256;
                87: sin_base = 14354;
                88: sin_base = 14449;
                89: sin_base = 14543;
                90: sin_base = 14635;
                91: sin_base = 14724;
                92: sin_base = 14811;
                93: sin_base = 14896;
                94: sin_base = 14978;
                95: sin_base = 15059;
                96: sin_base = 15137;
                97: sin_base = 15213;
                98: sin_base = 15286;
                99: sin_base = 15357;
                100: sin_base = 15426;
                101: sin_base = 15493;
                102: sin_base = 15557;
                103: sin_base = 15619;
                104: sin_base = 15679;
                105: sin_base = 15736;
                106: sin_base = 15791;
                107: sin_base = 15843;
                108: sin_base = 15893;
                109: sin_base = 15941;
                110: sin_base = 15986;
                111: sin_base = 16029;
                112: sin_base = 16069;
                113: sin_base = 16107;
                114: sin_base = 16143;
                115: sin_base = 16176;
                116: sin_base = 16207;
                117: sin_base = 16235;
                118: sin_base = 16261;
                119: sin_base = 16284;
                120: sin_base = 16305;
                121: sin_base = 16324;
                122: sin_base = 16340;
                123: sin_base = 16353;
                124: sin_base = 16364;
                125: sin_base = 16373;
                126: sin_base = 16379;
                127: sin_base = 16383;
                default: sin_base = 0;
            endcase
            // Dynamic scaling: base is Q1.14 (16 bits total).
            // Target format is Q1.(W-2), so shift by (16 - W).
            if (16 > W)
                get_sin = sin_base >>> (16 - W);
            else
                get_sin = sin_base <<< (W - 16);
        end
    endfunction

    // =========================================================================
    // Quadrant decoding and sin/cos lookup
    // angle_index format: {quadrant[1:0], lut_addr[ADDR_W-1:0]}
    // =========================================================================
    wire [1:0]        quadrant = angle_index[ADDR_W+1:ADDR_W];
    wire [ADDR_W-1:0] lut_addr = angle_index[ADDR_W-1:0];

    // Complementary address for quarter-wave symmetry: cos(θ) = sin(π/2 − θ)
    wire [ADDR_W-1:0] comp_addr = {ADDR_W{1'b1}} - lut_addr; // (2^ADDR_W - 1) - addr

    reg signed [W-1:0] sin_val, cos_val;

    // Quarter-wave symmetry decoding
    //   Q0 (0 to π/2):    sin = +sin_rom[addr],   cos = +sin_rom[comp_addr]
    //   Q1 (π/2 to π):    sin = +sin_rom[comp],   cos = -sin_rom[addr]
    //   Q2 (π to 3π/2):   sin = -sin_rom[addr],   cos = -sin_rom[comp_addr]
    //   Q3 (3π/2 to 2π):  sin = -sin_rom[comp],   cos = +sin_rom[addr]
    always @(*) begin
        case (quadrant)
            2'd0: begin
                sin_val =  get_sin(lut_addr);
                cos_val =  get_sin(comp_addr);
            end
            2'd1: begin
                sin_val =  get_sin(comp_addr);
                cos_val = -get_sin(lut_addr);
            end
            2'd2: begin
                sin_val = -get_sin(lut_addr);
                cos_val = -get_sin(comp_addr);
            end
            2'd3: begin
                sin_val = -get_sin(comp_addr);
                cos_val =  get_sin(lut_addr);
            end
        endcase
    end

    // =========================================================================
    // Rotation: [y_even]   [cos  -sin] [x_even]
    //           [y_odd ] = [sin   cos] [x_odd ]
    //
    // Requires 4 multiplications and 2 additions.
    // =========================================================================
    wire signed [2*W-1:0] prod_ce = cos_val * x_even;  // cos * x_even
    wire signed [2*W-1:0] prod_so = sin_val * x_odd;   // sin * x_odd
    wire signed [2*W-1:0] prod_se = sin_val * x_even;  // sin * x_even
    wire signed [2*W-1:0] prod_co = cos_val * x_odd;   // cos * x_odd

    // Truncate back to W bits: Q1.F × Q1.F = Q2.2F → extract Q1.F
    wire signed [W-1:0] trunc_ce = prod_ce[2*W-3:W-2];
    wire signed [W-1:0] trunc_so = prod_so[2*W-3:W-2];
    wire signed [W-1:0] trunc_se = prod_se[2*W-3:W-2];
    wire signed [W-1:0] trunc_co = prod_co[2*W-3:W-2];

    // Pipeline register for output
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
