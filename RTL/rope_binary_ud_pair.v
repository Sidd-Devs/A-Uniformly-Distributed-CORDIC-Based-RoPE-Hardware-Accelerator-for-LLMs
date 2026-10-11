module rope_binary_ud_pair #(
    parameter W = 8,
    parameter STAGES = 10
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_even,
    input  signed [W-1:0] x_odd,

    input  [STAGES-1:0] ud_bits,

    output signed [W-1:0] y_even,
    output signed [W-1:0] y_odd
);

    // Rotate (x_even, x_odd)
    binary_ud_cordic_core #(
        .W(W),
        .STAGES(STAGES)
    ) rotator (
        .clk(clk),
        .rst(rst),
        .x_in(x_even),
        .y_in(x_odd),
        .ud_bits_in(ud_bits),  // <--- FIXED: Now matches the core's new port name
        .x_out(y_even),
        .y_out(y_odd)
    );

endmodule
