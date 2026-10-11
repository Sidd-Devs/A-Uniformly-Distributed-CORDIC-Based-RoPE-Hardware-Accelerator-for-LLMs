module cordic_standard_stage #(
    parameter W = 32,
    parameter I = 0,
    parameter signed [W-1:0] ATAN = 0
)(
    input  signed [W-1:0] x_in,
    input  signed [W-1:0] y_in,
    input  signed [W-1:0] z_in,

    output signed [W-1:0] x_out,
    output signed [W-1:0] y_out,
    output signed [W-1:0] z_out
);

    wire d = z_in[W-1]; // sign(z)

    assign x_out = d ?
        (x_in + (y_in >>> I)) :
        (x_in - (y_in >>> I));

    assign y_out = d ?
        (y_in - (x_in >>> I)) :
        (y_in + (x_in >>> I));

    assign z_out = d ?
        (z_in + ATAN) :
        (z_in - ATAN);

endmodule

