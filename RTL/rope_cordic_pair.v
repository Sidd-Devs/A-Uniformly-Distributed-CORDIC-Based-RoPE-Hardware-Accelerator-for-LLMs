module rope_cordic_pair #(
    parameter W = 32,
    parameter STAGES = 10
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_even,
    input  signed [W-1:0] x_odd,
    input  signed [W-1:0] theta,

    output signed [W-1:0] y_even,
    output signed [W-1:0] y_odd
);

    cordic_standard_core #(
        .W(W),
        .STAGES(STAGES)
    ) rot (
        .clk(clk),
        .rst(rst),
        .x_in(x_even),
        .y_in(x_odd),
        .z_in(theta),
        .x_out(y_even),
        .y_out(y_odd)
    );

endmodule

