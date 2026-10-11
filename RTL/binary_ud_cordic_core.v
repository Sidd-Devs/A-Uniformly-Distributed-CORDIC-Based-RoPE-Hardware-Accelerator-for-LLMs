module binary_ud_cordic_core #(
    parameter W = 8,
    parameter STAGES = 10
)(
    input  clk,
    input  rst,

    input  signed [W-1:0] x_in,
    input  signed [W-1:0] y_in,

    // 1 bit per stage: 1 = Positive Rotation, 0 = Negative Rotation
    input  [STAGES-1:0] ud_bits_in,

    output signed [W-1:0] x_out,
    output signed [W-1:0] y_out
);

    reg signed [W-1:0] x [0:STAGES];
    reg signed [W-1:0] y [0:STAGES];
    
    // Pipeline shift registers for the rotation bits
    // ud_pipeline[stage][bit_index]
    reg [STAGES-1:0] ud_pipeline [0:STAGES-1];

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= STAGES; i = i + 1) begin
                x[i] <= {W{1'b0}};
                y[i] <= {W{1'b0}};
            end
            for (i = 0; i < STAGES; i = i + 1) begin
                ud_pipeline[i] <= {STAGES{1'b0}};
            end
        end else begin
            // 1. Feed input data into the first stage
            x[0] <= x_in;
            y[0] <= y_in;
            
            // Feed input bits into the first stage of the bit pipeline
            ud_pipeline[0] <= ud_bits_in;

            // 2. Process pipeline stages
            for (i = 0; i < STAGES; i = i + 1) begin
                
                // Pass the remaining bits down the pipeline alongside the data
                if (i < STAGES - 1) begin
                    ud_pipeline[i+1] <= ud_pipeline[i];
                end

                // Binary UD-CORDIC: 1 = Add/Sub, 0 = Sub/Add (NEVER skip)
                // Note: We use ud_pipeline[i][i] to grab the bit that traveled with this data
                if (ud_pipeline[i][i]) begin
                    x[i+1] <= x[i] - (y[i] >>> (i+1));
                    y[i+1] <= y[i] + (x[i] >>> (i+1));
                end else begin
                    x[i+1] <= x[i] + (y[i] >>> (i+1));
                    y[i+1] <= y[i] - (x[i] >>> (i+1));
                end
            end
        end
    end

    assign x_out = x[STAGES];
    assign y_out = y[STAGES];

endmodule
