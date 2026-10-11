"""T2.2 / R4-8 — latency, throughput and energy from the synthesis data.

Reviewer 4: "Add throughput/latency/energy metrics, not only area and power."

All inputs are measured post-syn_opt numbers under the same 2.0 ns constraint.
  F_max      = 1 / (T_constraint - slack)
  throughput = F_max  (fully pipelined, one rotation per cycle)
  latency    = pipeline_depth / F_max
  energy/rot = reported power / 500 MHz   (the constraint the power was taken at)
"""
T_NS = 2.0
F_REPORT = 500e6

# name: (W, depth, area_um2, power_mW, slack_ps)
D = {
 "W = 9  (paper's old operating point)": {
    "Mult RoPE*":        (9, 1, 1953, 0.810, 1086),
    "LUT RoPE":          (9, 1, 2339, 1.770,  536),
    "Std CORDIC N=10":   (9,10, 3292, 3.062, 1381),
    "Merged UD N=5":     (9, 5, 2330, 1.117, 1347),
 },
 "W = 14 (measured requirement, F=12)": {
    "Mult RoPE*":        (14, 1, 4185, 2.092,  600),
    "LUT RoPE":          (14, 1, 5375, 4.544,   26),
    "Binary UD N=10":    (14,10, 4346, 2.800, None),
    "Merged UD N=5":     (14, 5, 3467, 1.765, 1105),
 },
}

def row(name, w, depth, area, power, slack):
    if slack is None:
        fmax = lat = float("nan")
    else:
        fmax = 1e3 / (T_NS * 1e3 - slack)      # GHz
        lat = depth / fmax                      # ns
    e = power * 1e-3 / F_REPORT * 1e12          # pJ per rotation
    return (name, area, power, depth, fmax, lat, e)

for group, designs in D.items():
    print("=" * 78); print(group); print("=" * 78)
    print(f"{'design':<20}{'area':>7}{'power':>8}{'depth':>7}"
          f"{'F_max':>9}{'latency':>9}{'energy':>9}")
    print(f"{'':<20}{'um2':>7}{'mW':>8}{'cyc':>7}{'GHz':>9}{'ns':>9}{'pJ/rot':>9}")
    print("-" * 78)
    out = [row(n, *v) for n, v in designs.items()]
    for n, a, p, d, f, l, e in out:
        fs = "  n/a" if f != f else f"{f:>9.3f}"
        ls = "  n/a" if l != l else f"{l:>9.2f}"
        print(f"{n:<20}{a:>7}{p:>8.3f}{d:>7}{fs}{ls}{e:>9.2f}")
    best = min(out, key=lambda r: r[6])
    print(f"\n  lowest energy: {best[0]} at {best[6]:.2f} pJ/rotation")
    print()

print("* Mult RoPE consumes sin/cos it does not generate; the other three are")
print("  complete. See the shared angle front-end in Section III.\n")

print("=" * 78)
print("Energy per token, one LLaMA-2-7B-class model (d_h=128, 32 heads, 32 layers)")
print("=" * 78)
rot = (128 // 2) * 32 * 2 * 32
print(f"  rotations per token = (128/2 pairs) x 32 heads x 2 (q,k) x 32 layers"
      f" = {rot:,}\n")
for n, v in D["W = 14 (measured requirement, F=12)"].items():
    e = v[3] * 1e-3 / F_REPORT * 1e12
    print(f"  {n:<20} {e*rot/1e3:>8.2f} nJ/token")
