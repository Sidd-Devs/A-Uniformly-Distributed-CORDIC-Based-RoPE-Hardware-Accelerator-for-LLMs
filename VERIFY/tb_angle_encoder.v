`timescale 1ns/1ps
// Self-checking testbench: rope_angle_encoder vs. the bit-exact golden model.
module tb_angle_encoder;

    localparam N  = 10;
    localparam FB = 12;

    reg  signed [FB+1:0] theta_res;
    wire        [N-1:0]  ud_bits;

    rope_angle_encoder #(.N(N), .FB(FB)) dut (
        .theta_res(theta_res),
        .ud_bits(ud_bits)
    );

    integer fd, code, errors, total;
    reg [31:0] tq, expect_bits;

    initial begin
        errors = 0;
        total  = 0;
        fd = $fopen("golden_vectors.txt", "r");
        if (fd == 0) begin
            $display("FATAL: cannot open golden_vectors.txt");
            $finish;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%h %h\n", tq, expect_bits);
            if (code == 2) begin
                theta_res = tq[FB+1:0];
                #1;
                total = total + 1;
                if (ud_bits !== expect_bits[N-1:0]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("MISMATCH theta=%04h  got=%03h  exp=%03h",
                                 tq, ud_bits, expect_bits[N-1:0]);
                end
            end
        end
        $fclose(fd);
        $display("");
        $display("=========================================");
        $display(" vectors checked : %0d", total);
        $display(" mismatches      : %0d", errors);
        if (errors == 0) $display(" RESULT          : PASS");
        else             $display(" RESULT          : FAIL");
        $display("=========================================");
        $finish;
    end
endmodule
