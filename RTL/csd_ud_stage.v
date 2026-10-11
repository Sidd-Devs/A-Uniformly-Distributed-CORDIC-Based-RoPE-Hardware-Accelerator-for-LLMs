module csd_ud_stage #(
    parameter WIDTH = 9,
    parameter SHIFT1 = 0,
    parameter SHIFT2 = 1
)(
    input clk,
    input rst,

    input  signed [WIDTH-1:0] x_in,
    input  signed [WIDTH-1:0] y_in,
    input  signed [2:0] delta,

    output reg signed [WIDTH-1:0] x_out,
    output reg signed [WIDTH-1:0] y_out
);

    wire signed [WIDTH-1:0] x_s1 = x_in >>> SHIFT1;
    wire signed [WIDTH-1:0] x_s2 = x_in >>> SHIFT2;
    wire signed [WIDTH-1:0] y_s1 = y_in >>> SHIFT1;
    wire signed [WIDTH-1:0] y_s2 = y_in >>> SHIFT2;

    reg signed [WIDTH-1:0] x_comb;
    reg signed [WIDTH-1:0] y_comb;

    // Combinatorial adder selection based on Booth/CSD digit
    // Delta values: +2, +1, 0, -1, -2 (3-bit signed)
    always @(*) begin
        case (delta)
            3'sb010: begin  // +2
                x_comb =  x_s1 + x_s2;
                y_comb =  y_s1 + y_s2;
            end
            3'sb001: begin  // +1
                x_comb =  x_s1;
                y_comb =  y_s1;
            end
            3'sb000: begin  // 0 (no rotation)
                x_comb = 0;
                y_comb = 0;
            end
            3'sb111: begin  // -1
                x_comb = -x_s1;
                y_comb = -y_s1;
            end
            3'sb110: begin  // -2
                x_comb = -(x_s1 + x_s2);
                y_comb = -(y_s1 + y_s2);
            end
            default: begin
                x_comb = 0;
                y_comb = 0;
            end
        endcase
    end

    // Synchronous state update
    always @(posedge clk) begin
        if (rst) begin
            x_out <= 0;
            y_out <= 0;
        end else begin
            // Cross-coupled addition/subtraction for rotation
            x_out <= x_in - y_comb;
            y_out <= y_in + x_comb;
        end
    end

endmodule
