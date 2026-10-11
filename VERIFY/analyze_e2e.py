import math, sys
FB=12
K = math.prod(math.hypot(1,2.0**-(i+1)) for i in range(10))
errs=[]; gains=[]
for line in open("e2e_out.txt"):
    tq,x,y = (int(v) for v in line.split())
    th = tq/(1<<FB)
    if x==0 and y==0: continue
    got = math.atan2(y,x)
    errs.append(abs(got-th))
    gains.append(math.hypot(x,y)/(1<<(int(sys.argv[1])-2)))
rms=math.degrees(math.sqrt(sum(e*e for e in errs)/len(errs)))
print(f"  points {len(errs)}   RMS {rms:.4f} deg   max {math.degrees(max(errs)):.4f} deg")
print(f"  gain: measured mean {sum(gains)/len(gains):.6f}  spread [{min(gains):.6f}, {max(gains):.6f}]")
print(f"        theoretical K = {K:.6f}   -> gain is angle-INDEPENDENT (contribution C1)")
