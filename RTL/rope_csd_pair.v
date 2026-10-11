module rope_csd_pair #(
    parameter WIDTH  = 9,
    parameter STAGES = 12
)(
    input clk,
    input rst,

    input  signed [WIDTH-1:0] x_in,
    input  signed [WIDTH-1:0] y_in,
    input  [2*STAGES-1:0] angle_bits,

    output signed [WIDTH-1:0] x_out,
    output signed [WIDTH-1:0] y_out
);

    csd_ud_cordic_core #(
        .WIDTH(WIDTH),
        .STAGES(STAGES)
    ) core (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .y_in(y_in),
        .angle_bits(angle_bits),
        .x_out(x_out),
        .y_out(y_out)
    );

endmodule
