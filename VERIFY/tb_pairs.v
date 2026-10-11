`timescale 1ns/1ps
// ============================================================================
// T1.6 — self-checking testbenches for the five rope_*_pair modules.
//
// Golden vectors come from VERIFY/gen_pair_vectors.py, which models each RTL
// bit-accurately (same truncation, same wrap, same floor-shift semantics) and
// parses the sin/atan tables out of the RTL itself so the two cannot drift.
//
// Each vector is applied and held for SETTLE cycles before sampling, so the
// checks do not depend on assumed pipeline latency -- a module whose depth
// changes still passes, while a module whose ARITHMETIC changes fails.
//
// Build one at a time, selecting the top:
//   iverilog -g2005 -s tb_mult -o t.vvp tb_pairs.v ../RTL/rope_mult_pair.v
// ============================================================================

// ---------------------------------------------------------------- multiplier
module tb_mult;
  localparam W=12, SETTLE=6;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg signed [W-1:0] xe,xo,sv,cv; wire signed [W-1:0] ye,yo;
  rope_mult_pair #(.W(W)) dut(.clk(clk),.rst(rst),.x_even(xe),.x_odd(xo),
                              .sin_val(sv),.cos_val(cv),.y_even(ye),.y_odd(yo));
  integer fd,c,n,bad; reg [31:0] a,b,s,k,ee,eo;
  initial begin
    n=0; bad=0; fd=$fopen("gv_mult.txt","r");
    if (fd==0) begin $display("FATAL: gv_mult.txt missing"); $finish; end
    repeat(4) @(posedge clk); rst=0;
    while (!$feof(fd)) begin
      c=$fscanf(fd,"%h %h %h %h %h %h\n",a,b,s,k,ee,eo);
      if (c==6) begin
        xe=a[W-1:0]; xo=b[W-1:0]; sv=s[W-1:0]; cv=k[W-1:0];
        repeat(SETTLE) @(posedge clk);
        n=n+1;
        if (ye!==$signed(ee[W-1:0]) || yo!==$signed(eo[W-1:0])) begin
          bad=bad+1;
          if (bad<=5) $display("  MISMATCH #%0d got(%0d,%0d) exp(%0d,%0d)",
                               n,ye,yo,$signed(ee[W-1:0]),$signed(eo[W-1:0]));
        end
      end
    end
    $fclose(fd);
    $display("tb_mult      %0d vectors, %0d mismatches : %s",n,bad,bad?"FAIL":"PASS");
    $finish;
  end
endmodule

// ----------------------------------------------------------------------- LUT
module tb_lut;
  localparam W=12, AW=7, SETTLE=6;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg signed [W-1:0] xe,xo; reg [AW+1:0] ai; wire signed [W-1:0] ye,yo;
  rope_lut_pair #(.W(W),.ADDR_W(AW)) dut(.clk(clk),.rst(rst),.x_even(xe),
                 .x_odd(xo),.angle_index(ai),.y_even(ye),.y_odd(yo));
  integer fd,c,n,bad; reg [31:0] a,b,g,ee,eo;
  initial begin
    n=0; bad=0; fd=$fopen("gv_lut.txt","r");
    if (fd==0) begin $display("FATAL: gv_lut.txt missing"); $finish; end
    repeat(4) @(posedge clk); rst=0;
    while (!$feof(fd)) begin
      c=$fscanf(fd,"%h %h %h %h %h\n",a,b,g,ee,eo);
      if (c==5) begin
        xe=a[W-1:0]; xo=b[W-1:0]; ai=g[AW+1:0];
        repeat(SETTLE) @(posedge clk);
        n=n+1;
        if (ye!==$signed(ee[W-1:0]) || yo!==$signed(eo[W-1:0])) begin
          bad=bad+1;
          if (bad<=5) $display("  MISMATCH #%0d ai=%03h got(%0d,%0d) exp(%0d,%0d)",
                               n,ai,ye,yo,$signed(ee[W-1:0]),$signed(eo[W-1:0]));
        end
      end
    end
    $fclose(fd);
    $display("tb_lut       %0d vectors, %0d mismatches : %s",n,bad,bad?"FAIL":"PASS");
    $finish;
  end
endmodule

// ----------------------------------------------------------------- Binary UD
module tb_binary;
  localparam W=12, N=10, SETTLE=N+4;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg signed [W-1:0] xe,xo; reg [N-1:0] ud; wire signed [W-1:0] ye,yo;
  rope_binary_ud_pair #(.W(W),.STAGES(N)) dut(.clk(clk),.rst(rst),.x_even(xe),
                       .x_odd(xo),.ud_bits(ud),.y_even(ye),.y_odd(yo));
  integer fd,c,n,bad; reg [31:0] a,b,u,ee,eo;
  initial begin
    n=0; bad=0; fd=$fopen("gv_binary.txt","r");
    if (fd==0) begin $display("FATAL: gv_binary.txt missing"); $finish; end
    repeat(4) @(posedge clk); rst=0;
    while (!$feof(fd)) begin
      c=$fscanf(fd,"%h %h %h %h %h\n",a,b,u,ee,eo);
      if (c==5) begin
        xe=a[W-1:0]; xo=b[W-1:0]; ud=u[N-1:0];
        repeat(SETTLE) @(posedge clk);
        n=n+1;
        if (ye!==$signed(ee[W-1:0]) || yo!==$signed(eo[W-1:0])) begin
          bad=bad+1;
          if (bad<=5) $display("  MISMATCH #%0d ud=%03h got(%0d,%0d) exp(%0d,%0d)",
                               n,ud,ye,yo,$signed(ee[W-1:0]),$signed(eo[W-1:0]));
        end
      end
    end
    $fclose(fd);
    $display("tb_binary    %0d vectors, %0d mismatches : %s",n,bad,bad?"FAIL":"PASS");
    $finish;
  end
endmodule

// ----------------------------------------------------------------- merged UD
module tb_csd;
  localparam W=12, N=5, SETTLE=N+6;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg signed [W-1:0] xe,xo; reg [2*N-1:0] ab; wire signed [W-1:0] ye,yo;
  rope_csd_pair #(.WIDTH(W),.STAGES(N)) dut(.clk(clk),.rst(rst),.x_in(xe),
                 .y_in(xo),.angle_bits(ab),.x_out(ye),.y_out(yo));
  integer fd,c,n,bad; reg [31:0] a,b,g,ee,eo;
  initial begin
    n=0; bad=0; fd=$fopen("gv_csd.txt","r");
    if (fd==0) begin $display("FATAL: gv_csd.txt missing"); $finish; end
    repeat(4) @(posedge clk); rst=0;
    while (!$feof(fd)) begin
      c=$fscanf(fd,"%h %h %h %h %h\n",a,b,g,ee,eo);
      if (c==5) begin
        xe=a[W-1:0]; xo=b[W-1:0]; ab=g[2*N-1:0];
        repeat(SETTLE) @(posedge clk);
        n=n+1;
        if (ye!==$signed(ee[W-1:0]) || yo!==$signed(eo[W-1:0])) begin
          bad=bad+1;
          if (bad<=5) $display("  MISMATCH #%0d ab=%03h got(%0d,%0d) exp(%0d,%0d)",
                               n,ab,ye,yo,$signed(ee[W-1:0]),$signed(eo[W-1:0]));
        end
      end
    end
    $fclose(fd);
    $display("tb_csd       %0d vectors, %0d mismatches : %s",n,bad,bad?"FAIL":"PASS");
    $finish;
  end
endmodule

// ------------------------------------------------------------ standard CORDIC
module tb_cordic;
  localparam W=12, N=10, SETTLE=N+4;
  reg clk=0, rst=1; always #5 clk=~clk;
  reg signed [W-1:0] xe,xo,th; wire signed [W-1:0] ye,yo;
  rope_cordic_pair #(.W(W),.STAGES(N)) dut(.clk(clk),.rst(rst),.x_even(xe),
                    .x_odd(xo),.theta(th),.y_even(ye),.y_odd(yo));
  integer fd,c,n,bad; reg [31:0] a,b,t,ee,eo;
  initial begin
    n=0; bad=0; fd=$fopen("gv_cordic.txt","r");
    if (fd==0) begin $display("FATAL: gv_cordic.txt missing"); $finish; end
    repeat(4) @(posedge clk); rst=0;
    while (!$feof(fd)) begin
      c=$fscanf(fd,"%h %h %h %h %h\n",a,b,t,ee,eo);
      if (c==5) begin
        xe=a[W-1:0]; xo=b[W-1:0]; th=t[W-1:0];
        repeat(SETTLE) @(posedge clk);
        n=n+1;
        if (ye!==$signed(ee[W-1:0]) || yo!==$signed(eo[W-1:0])) begin
          bad=bad+1;
          if (bad<=5) $display("  MISMATCH #%0d th=%0d got(%0d,%0d) exp(%0d,%0d)",
                               n,$signed(th),ye,yo,$signed(ee[W-1:0]),$signed(eo[W-1:0]));
        end
      end
    end
    $fclose(fd);
    $display("tb_cordic    %0d vectors, %0d mismatches : %s",n,bad,bad?"FAIL":"PASS");
    $finish;
  end
endmodule
