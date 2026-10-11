`timescale 1ns/1ps
// End-to-end: rope_angle_encoder -> binary_ud_cordic_core.
// Dumps (theta_q, x_out, y_out) so Python can measure the true rotation error
// of the synthesized datapath. This is the check that closes T1.5.
module tb_e2e;

    parameter W      = 16;
    parameter STAGES = 10;
    parameter FB     = 12;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg signed [FB+1:0] theta_res;
    wire [STAGES-1:0]   ud_bits;

    rope_angle_encoder #(.N(STAGES), .FB(FB)) enc (
        .theta_res(theta_res), .ud_bits(ud_bits));

    reg  signed [W-1:0] x_in, y_in;
    wire signed [W-1:0] x_out, y_out;

    binary_ud_cordic_core #(.W(W), .STAGES(STAGES)) core (
        .clk(clk), .rst(rst),
        .x_in(x_in), .y_in(y_in),
        .ud_bits_in(ud_bits),
        .x_out(x_out), .y_out(y_out));

    integer fd, k, lim, tq;
    integer NPTS;

    initial begin
        NPTS = 400;
        lim  = (3217 * (1 << FB)) / 4096;   // pi/4 in Q1.FB
        fd = $fopen("e2e_out.txt", "w");
        x_in = 0; y_in = 0; theta_res = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (k = 0; k <= NPTS; k = k + 1) begin
            tq = -lim + (2 * lim * k) / NPTS;
            theta_res = tq[FB+1:0];
            // unit vector along +x, Q1.(W-2)
            x_in = (1 <<< (W - 2));
            y_in = 0;
            repeat (STAGES + 2) @(posedge clk);
            $fwrite(fd, "%0d %0d %0d\n", tq, x_out, y_out);
        end
        $fclose(fd);
        $display("e2e: wrote %0d points (W=%0d STAGES=%0d)", NPTS + 1, W, STAGES);
        $finish;
    end
endmodule
