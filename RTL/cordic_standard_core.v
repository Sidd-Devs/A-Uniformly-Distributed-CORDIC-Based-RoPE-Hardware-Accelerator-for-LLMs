module cordic_standard_core #(
    parameter W = 32,
    parameter STAGES = 10
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_in,
    input  signed [W-1:0] y_in,
    input  signed [W-1:0] z_in,

    output signed [W-1:0] x_out,
    output signed [W-1:0] y_out
);

    // =========================================================
    // DYNAMIC LUT: Base values are Q1.14. 
    // We shift them down dynamically based on the actual W.
    // Assuming format is Q2.(W-2) -> fractional bits = W - 2.
    // Shift amount = 14 - (W - 2) = 16 - W
    // =========================================================
    function signed [W-1:0] get_atan;
        input integer idx;
        integer base_val;
        begin
            case (idx)
                0: base_val = 6434; // atan(2^-0)  in Q1.14 = 0.7854 rad
                1: base_val = 3798; // atan(2^-1)  = 0.4636 rad
                2: base_val = 2007; // atan(2^-2)  = 0.2450 rad
                3: base_val = 1019; // atan(2^-3)  = 0.1244 rad
                4: base_val = 511;  // atan(2^-4)  = 0.0624 rad
                5: base_val = 256;  // atan(2^-5)  = 0.0312 rad
                6: base_val = 128;  // atan(2^-6)  = 0.0156 rad
                7: base_val = 64;   // atan(2^-7)  = 0.0078 rad
                8: base_val = 32;   // atan(2^-8)  = 0.0039 rad
                9: base_val = 16;   // atan(2^-9)  = 0.0020 rad
                10: base_val = 8;   // atan(2^-10) = 0.00098 rad
                11: base_val = 4;   // atan(2^-11) = 0.00049 rad
                12: base_val = 2;   // atan(2^-12) = 0.00024 rad
                13: base_val = 1;   // atan(2^-13) = 0.00012 rad
                default: base_val = 0;
            endcase
            // Shift down if W < 16, Shift up if W > 16
            if (16 > W)
                get_atan = base_val >>> (16 - W);
            else
                get_atan = base_val <<< (W - 16);
        end
    endfunction

    // Pipeline wire arrays
    wire signed [W-1:0] x_pipe [0:STAGES];
    wire signed [W-1:0] y_pipe [0:STAGES];
    wire signed [W-1:0] z_pipe [0:STAGES];

    // Pipeline register arrays for sequential logic
    reg signed [W-1:0] x_reg [0:STAGES];
    reg signed [W-1:0] y_reg [0:STAGES];
    reg signed [W-1:0] z_reg [0:STAGES];

    assign x_pipe[0] = x_in;
    assign y_pipe[0] = y_in;
    assign z_pipe[0] = z_in;

    genvar i;
    generate
        for (i = 0; i < STAGES; i = i + 1) begin : STAGE
            
            // Wire the output of the stage
            wire signed [W-1:0] x_stage_out;
            wire signed [W-1:0] y_stage_out;
            wire signed [W-1:0] z_stage_out;

            // Instantiate the unused stage module properly!
            cordic_standard_stage #(
                .W(W),
                .I(i),
                .ATAN(get_atan(i)) // Dynamically scaled ATAN
            ) stage_inst (
                .x_in(x_pipe[i]),
                .y_in(y_pipe[i]),
                .z_in(z_pipe[i]),
                .x_out(x_stage_out),
                .y_out(y_stage_out),
                .z_out(z_stage_out)
            );

            // Register the outputs for a clean pipeline
            always @(posedge clk) begin
                if (rst) begin
                    x_reg[i+1] <= 0;
                    y_reg[i+1] <= 0;
                    z_reg[i+1] <= 0;
                end else begin
                    x_reg[i+1] <= x_stage_out;
                    y_reg[i+1] <= y_stage_out;
                    z_reg[i+1] <= z_stage_out;
                end
            end
            
            // Connect registers to the next pipe stage
            assign x_pipe[i+1] = x_reg[i+1];
            assign y_pipe[i+1] = y_reg[i+1];
            assign z_pipe[i+1] = z_reg[i+1];
            
        end
    endgenerate

    assign x_out = x_pipe[STAGES];
    assign y_out = y_pipe[STAGES];

endmodule
