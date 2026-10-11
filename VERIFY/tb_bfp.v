`timescale 1ns/1ps
// Self-checking: normalise -> (rotate is a no-op here) -> denormalise must be
// lossless within the shift, and the normalised block must stay in range.
module tb_bfp;
  localparam W=12, D=4, EW=6, LANES=2*D, TARGET=W-4;
  reg  [LANES*W-1:0] vin;
  wire [LANES*W-1:0] vnorm, vback;
  wire signed [EW-1:0] e;

  rope_bfp_norm   #(.W(W),.D(D),.EW(EW)) u1(.vec_in(vin),.vec_out(vnorm),.exp_out(e));
  rope_bfp_denorm #(.W(W),.D(D),.EW(EW)) u2(.vec_in(vnorm),.exp_in(e),.vec_out(vback));

  integer t,i,errs,rng; reg signed [W-1:0] v,n,b; integer maxmag,lim;
  initial begin
    errs=0; lim = 1<<(W-3);           // must stay below 2^(W-3) after scaling
    for (t=0;t<300;t=t+1) begin
      // sweep magnitudes from tiny to full-scale
      rng = (t%10==0) ? 3 : (1 << (1 + (t % (W-1))));
      for (i=0;i<LANES;i=i+1) begin
        v = ($random % rng);
        vin[i*W +: W] = v;
      end
      #1;
      maxmag=0;
      for (i=0;i<LANES;i=i+1) begin
        n = vnorm[i*W +: W];
        b = vback[i*W +: W];
        v = vin[i*W +: W];
        if ((n<0 ? -n : n) > maxmag) maxmag = (n<0 ? -n : n);
        // round trip must not corrupt when shifting left then right
        if (e >= 0 && b !== v) begin
          errs=errs+1;
          if (errs<5) $display("  ROUNDTRIP t=%0d lane=%0d in=%0d out=%0d e=%0d",t,i,v,b,e);
        end
      end
      if (maxmag >= lim) begin
        errs=errs+1;
        if (errs<8) $display("  RANGE t=%0d maxmag=%0d limit=%0d e=%0d",t,maxmag,lim,e);
      end
    end
    $display("");
    $display("  vectors: 300 x %0d lanes   errors: %0d", LANES, errs);
    $display("  RESULT: %s", (errs==0)?"PASS":"FAIL");
    $display("  (range check: |normalised| < 2^%0d, leaving headroom for K=1.164)", W-3);
    $finish;
  end
endmodule
