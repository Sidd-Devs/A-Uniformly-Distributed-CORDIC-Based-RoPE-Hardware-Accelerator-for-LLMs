module csd_ud_cordic_core #(
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

    wire [3*STAGES-1:0] delta_flat;

    // 1. Combinatorial Booth/CSD Encoding
    csd_control #(.STAGES(STAGES)) ctrl (
        .binary_bits(angle_bits),
        .delta_flat(delta_flat)
    );

    wire signed [WIDTH-1:0] x [0:STAGES];
    wire signed [WIDTH-1:0] y [0:STAGES];

    assign x[0] = x_in;
    assign y[0] = y_in;

    // 2. The 2D Pipeline Shift Register for Delta values
    reg signed [2:0] delta_pipe [0:STAGES-1][0:STAGES-1];
    integer j, k;

    always @(posedge clk) begin
        if (rst) begin
            for (j = 0; j < STAGES; j = j + 1) begin
                for (k = 0; k < STAGES; k = k + 1) begin
                    delta_pipe[j][k] <= 3'd0;
                end
            end
        end else begin
            // Load the combinatorial output into the first pipeline stage
            for (k = 0; k < STAGES; k = k + 1) begin
                delta_pipe[0][k] <= delta_flat[3*k +: 3];
            end

            // Shift the deltas down the pipeline
            for (j = 0; j < STAGES-1; j = j + 1) begin
                for (k = 0; k < STAGES; k = k + 1) begin
                    delta_pipe[j+1][k] <= delta_pipe[j][k];
                end
            end
        end
    end

    // 3. Data Pipeline Generation
    genvar i;
    generate
        for (i = 0; i < STAGES; i = i + 1) begin : STAGE
            
            // Extract the specific delta that has been delayed by 'i' clock cycles
            wire signed [2:0] delta_i_delayed = delta_pipe[i][i];

            csd_ud_stage #(
                .WIDTH(WIDTH),
                .SHIFT1(2*i),
                .SHIFT2(2*i+1)
            ) stage_i (
                .clk(clk),
                .rst(rst),
                .x_in(x[i]),
                .y_in(y[i]),
                .delta(delta_i_delayed), 
                .x_out(x[i+1]),
                .y_out(y[i+1])
            );
        end
    endgenerate

    assign x_out = x[STAGES];
    assign y_out = y[STAGES];

endmodule
